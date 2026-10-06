import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from loguru import logger as log

from modelforge.potential.utils import Dense

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.neighbors import PairlistData


class FukuiEquilibration(nn.Module):
    """
    Equilibrates per-atom charge to match per-system target totals, using learnable, atom-wise Fukui weights
    rather than a uniform correction.  This is adapted from MACE Polar, but is fundamentally the same as
    aimnet2-NSE approach, but applied to a single channel

    Parameters
    ----------
    number_of_per_atom_features : int
        Dimensionality of the atomic embedding used to predict the Fukui
        weights.
    hidden_dim : int
        Hidden layer size of the small Fukui-weight-predicting MLP.
    """

    def __init__(self, number_of_per_atom_features: int, hidden_dim: int = 32):
        super().__init__()
        # Predicts  non-negative "softness" weights per atom
        self.fukui_mlp = nn.Sequential(
            Dense(
                number_of_per_atom_features,
                hidden_dim,
                activation_function=nn.SiLU(),
            ),
            Dense(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    @staticmethod
    def _per_system_sum(
        values: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        n_systems: int,
    ) -> torch.Tensor:
        out = torch.zeros(
            n_systems, values.shape[-1], device=values.device, dtype=values.dtype
        )
        out.index_add_(0, atomic_subsystem_indices, values)
        return out

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        per_atom_charge: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        per_system_total_charge: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
         Parameters
         ----------
         atomic_embedding : (n_atoms, F) -- used to predict Fukui weights
         per_atom_charge : (n_atoms, 1)
         per_system_total_charge : (n_systems, 1)
         atomic_subsystem_indices : (n_atoms,)

         Returns
         -------
        dictionary containing "per_atom_charge"
        """

        device = atomic_embedding.device
        n_systems = (
            int(atomic_subsystem_indices.max().item()) + 1
            if atomic_subsystem_indices.numel() > 0
            else 0
        )

        # Non-negative, per-atom Fukui weights for each channel.
        raw_weights = self.softplus(self.fukui_mlp(atomic_embedding)) + 1e-6
        w_q = raw_weights[:, 0:1]

        # Normalize per system so each channel's weights form a partition
        # of unity over the atoms of that system (a proper Fukui function).
        def _normalize(weights: torch.Tensor) -> torch.Tensor:
            sums = self._per_system_sum(weights, atomic_subsystem_indices, n_systems)
            sums = sums[atomic_subsystem_indices]
            return weights / sums.clamp_min(1e-8)

        f_q = _normalize(w_q)

        # Current per-system totals of the predicted populations.
        q_sum = self._per_system_sum(
            per_atom_charge, atomic_subsystem_indices, n_systems
        )

        target_charge = per_system_total_charge.reshape(-1, 1).to(per_atom_charge.dtype)

        # q_l = q_{l-1} + f_q * (Q_target - Q_pred).
        # Here, Q is the total per system charge, and q is the partial charge.
        # f_q is the learned matrix that distributes the difference between
        # the labeled total charge and the predicted total charge to all atoms in the system

        residual_q = target_charge - q_sum

        # Redistribute residuals according to the *learned* Fukui weights
        # instead of uniformly across all atoms.

        corrected_charge = per_atom_charge + f_q * residual_q[atomic_subsystem_indices]

        return {"per_atom_charge": corrected_charge}


class AimNet2Core(torch.nn.Module):

    # per-atom properties computed by the core (charges are equilibrated);
    # readout heads must not overwrite them
    core_output_properties = frozenset(
        {"per_atom_scalar_representation", "per_atom_charge"}
    )

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_vector_features: int,
        number_of_interaction_modules: int,
        interaction_module_hidden_layers: List[List[int]],
        output_module_hidden_layers: List[int],
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[str],
        predicted_dim: List[int],
        maximum_interaction_radius: float,
        charge_equilibration_scheme: str,
    ) -> None:
        """
        Core architecture of the AimNet2 model for molecular property
        prediction.

        Parameters
        ----------
        featurization : Dict[str, Dict[str, int]]
            Configuration dictionary specifying feature details for atomic
            embeddings.
        number_of_radial_basis_functions : int
            Number of radial basis functions used in the radial symmetry
            function.
        number_of_vector_features : int
            Number of vector features used in the model, which determines the
            dimensionality of the angular symmetry functions.
        number_of_interaction_modules : int
            Number of interaction modules in the model, determining the depth of
            message passing.
        interaction_module_hidden_layers : List[List[int]]
            List of hidden layer sizes for each interaction module. Note, this requires at least one value to be defined
            as this specifies the output size of the first layer in the interaction module and input size of the final layer
        output_module_hidden_layers : List[int]
            List of hidden layer sizes for the output modules, which are used to
            compute the final predictions from the atomic embeddings.
        activation_function_parameter : Dict[str, str]
            Configuration of activation functions used across the model.
        predicted_properties : List[str]
            List of properties that the model is predicting (e.g., energy,
            forces).
        predicted_dim : List[int]
            The dimensionality of each predicted property.
        maximum_interaction_radius : float
            The cutoff radius for atomic interactions in the model.
        charge_equilibration_scheme : str
            The scheme used for charge equilibration in the model, which can "default" or "fukui"
        """

        super().__init__()

        log.debug("Initializing the AimNet2 architecture.")
        self.model_name = "aimnet2"
        self.activation_function = activation_function_parameter["activation_function"]

        # Initialize representation block
        self.representation_module = AIMNet2Representation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization,
        )
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )

        # scale by 1/sqrt(F_atom * G), the fan-in of the einsum contraction,
        # so vector features start on the same scale as radial features
        self.agh = nn.Parameter(
            torch.randn(
                number_of_per_atom_features,  # F_atom
                number_of_radial_basis_functions,  # G
                number_of_vector_features,  # H
            )
            / math.sqrt(number_of_per_atom_features * number_of_radial_basis_functions)
        )
        # shape(nr_of_angular_symmetry_functions,nr_of_radial_symmetry_functions,nr_of_vector_features)

        # Define interaction modules for message passing
        self.interaction_modules = torch.nn.ModuleList(
            [
                AIMNet2InteractionModule(
                    number_of_per_atom_features=number_of_per_atom_features,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    number_of_vector_features=number_of_vector_features,
                    hidden_layers=interaction_module_hidden_layers[i],
                    activation_function=self.activation_function,
                    is_first_module=(i == 0),
                )
                for i in range(number_of_interaction_modules)
            ]
        )
        # Define output layers to calculate per-atom predictions
        overlap = self.core_output_properties.intersection(predicted_properties)
        if overlap:
            raise ValueError(
                f"predicted_properties {sorted(overlap)} are computed by the "
                "AimNet2 core (equilibrated charges) and will not be overwritten "
                "by a readout head. Remove them from predicted_properties; the "
                "core always returns them."
            )
        self.output_layers = nn.ModuleDict()
        for property, dim in zip(predicted_properties, predicted_dim):
            self.output_layers[property] = mlp_init(
                n_in_features=number_of_per_atom_features,
                n_out_features=int(dim),
                activation_function=self.activation_function,
                hidden_layers=output_module_hidden_layers,
            )
        self.charge_equilibration_scheme = charge_equilibration_scheme

        if self.charge_equilibration_scheme == "default":
            from modelforge.potential.processing import ChargeConservation

            self.charge_conservation = ChargeConservation()
        elif self.charge_equilibration_scheme == "fukui":
            self.fukui_equilibration = FukuiEquilibration(
                number_of_per_atom_features, 32
            )

    def compute_properties(
        self,
        data: NNPInput,
        pairlist: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the requested properties for a given input batch.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist: PairlistData
            The output from the pairlist module.
        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom scalar representations and atomic subsystem
            indices.
        """

        rep = self.representation_module(data, pairlist)
        atomic_embedding = rep["atomic_embedding"]
        r_ij, d_ij, f_ij, f_cutoff = (
            pairlist.r_ij,
            pairlist.d_ij,
            rep["f_ij"],
            rep["f_cutoff"],
        )
        # Scalar Gaussian expansion for radial terms
        gs = f_ij * f_cutoff  # Shape: (number_of_pairs, G)
        # Unit direction vectors
        u_ij = r_ij / d_ij
        # Compute gv with shape (number_of_pairs, 3, G)
        gv = u_ij.unsqueeze(-1) * gs.unsqueeze(1)  # Broadcasting over G

        # Atomic embedding "a" Eqn. (3)
        partial_charges = torch.zeros(
            (atomic_embedding.shape[0], 1),
            dtype=atomic_embedding.dtype,
            device=atomic_embedding.device,
        )

        # Perform message passing using interaction modules
        for i, interaction in enumerate(self.interaction_modules):

            delta_a, delta_q, f = interaction(
                atomic_embedding,
                partial_charges,
                pairlist.pair_indices,
                gs,
                gv,
                self.agh,
            )

            # Update atomic embeddings
            atomic_embedding = atomic_embedding + delta_a

            # Apply scaling factor `f` to `delta_q`
            scaled_delta_q = f * delta_q

            # Update partial charges
            if i == 0:
                partial_charges = scaled_delta_q  # Initialize charges
            else:
                partial_charges = partial_charges + scaled_delta_q  # Incremental update

            if self.charge_equilibration_scheme == "default":
                partial_charges = self.charge_conservation(
                    {
                        "per_atom_charge": partial_charges,
                        "per_system_total_charge": data.per_system_total_charge.to(
                            dtype=atomic_embedding.dtype
                        ),
                        "atomic_subsystem_indices": data.atomic_subsystem_indices.to(
                            dtype=torch.int64
                        ),
                    }
                )["per_atom_charge"]
            elif self.charge_equilibration_scheme == "fukui":
                partial_charges = self.fukui_equilibration(
                    atomic_embedding=atomic_embedding,
                    per_atom_charge=partial_charges,
                    atomic_subsystem_indices=data.atomic_subsystem_indices.to(
                        dtype=torch.int64
                    ),
                    per_system_total_charge=data.per_system_total_charge.to(
                        dtype=atomic_embedding.dtype
                    ),
                )["per_atom_charge"]
        # check that none of the tensors are NaN
        if torch.isnan(atomic_embedding).any():
            raise ValueError("NaN values detected in atomic embeddings.")
        if torch.isnan(partial_charges).any():
            raise ValueError("NaN values detected in partial charges.")

        return {
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
            "per_atom_charge": partial_charges,
        }

    def forward(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the forward pass through the network.

        Parameters
        ----------
        data : NNPInput
            Contains input data for the batch obtained directly from the
            dataset, including atomic numbers, positions, and other relevant
            fields.
        pairlist_output : PairListData
            Contains the indices for the selected pairs and their associated
            distances and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and other properties from the
            forward pass.
        """
        # perform the forward pass implemented in the subclass
        results = self.compute_properties(data, pairlist_output)
        atomic_embedding = results["per_atom_scalar_representation"]

        # The way this is setup, we will always return per_atom_charge and electrostatic_energy
        # as those come directly from the compute_properties method.

        # Compute all specified outputs
        for output_name, output_layer in self.output_layers.items():
            output = output_layer(atomic_embedding)
            results[output_name] = output

        return results


def mlp_init(
    n_in_features: int,
    n_out_features: int,
    activation_function: nn.Module,
    hidden_layers: List[int],
) -> nn.Sequential:
    """
    Helper function to create a sequence of dense layers with an activation function.


    Parameters
    ----------
    n_in_features : int
        Number of input features for the first layer.
    n_out_features : int
        Number of output features for the last layer.
    activation_function : nn.Module
        Activation function to be applied after each hidden layer.
    hidden_layers : List[int]
        List of integers specifying the number of features for each hidden layer.
        First element is the number of output features of the first layer,
        and the last element is the number of input features of the final layer.
        If only a single value is provided, it will be used as the output size of the first layer
        and input size of the final layer (no intermediate layers will be created).
    """

    # Single MLP producing combined outputs
    num_of_layers = len(hidden_layers)

    mlp = nn.Sequential()

    mlp.append(
        Dense(
            in_features=n_in_features,
            out_features=hidden_layers[0],
            activation_function=activation_function,
        )
    )
    for i in range(1, num_of_layers):
        mlp.append(
            Dense(
                in_features=hidden_layers[i - 1],
                out_features=hidden_layers[i],
                activation_function=activation_function,
            )
        )
    mlp.append(
        Dense(
            in_features=hidden_layers[-1],
            out_features=n_out_features,  # delta_q, f, delta_a
        )
    )
    return mlp


class AIMNet2InteractionModule(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_vector_features: int,
        hidden_layers: List[int],
        activation_function: nn.Module,
        is_first_module: bool = False,
    ):
        super().__init__()
        self.is_first_module = is_first_module
        self.number_of_per_atom_features = number_of_per_atom_features
        self.number_of_vector_features = number_of_vector_features
        self.gs_to_fatom = Dense(
            number_of_radial_basis_functions, number_of_per_atom_features, bias=False
        )

        if not self.is_first_module:
            self.number_of_input_features = (
                number_of_per_atom_features  # radial_contributions_emb
                + number_of_vector_features  # vector_contributions_emb
                + number_of_per_atom_features  # radial_contributions_charge
                + number_of_vector_features  # vector_contributions_charge
            )
        else:
            self.number_of_input_features = (
                number_of_per_atom_features  # radial_contributions_emb
                + number_of_vector_features  # vector_contributions_emb
            )

        # Single MLP producing combined outputs
        self.mlp = mlp_init(
            n_in_features=self.number_of_input_features,
            n_out_features=self.number_of_per_atom_features + 2,
            activation_function=activation_function,
            hidden_layers=hidden_layers,
        )

    def calculate_radial_contributions(
        self,
        gs: Tensor,
        a_j: Tensor,
        number_of_atoms: int,
        idx_i: Tensor,
    ) -> Tensor:
        """
        Compute radial contributions for each atom based on pair interactions.

        Parameters
        ----------
        gs : Tensor
            Radial symmetry functions with shape (number_of_pairs, G).
        a_j : Tensor
            Neighbour (atom j) features for each pair with shape (number_of_pairs, F_atom) or (number_of_pairs, 1).
        number_of_atoms : int
            Total number of atoms in the system.
        idx_i : Tensor
            Index of the central atom for each pair, with shape (number_of_pairs,).

        Returns
        -------
        Tensor
            Radial contributions aggregated per atom, with shape (number_of_atoms, F_atom).
        """
        # Map gs to shape (number_of_pairs, F_atom)
        mapped_gs = self.gs_to_fatom(gs)  # Shape: (number_of_pairs, F_atom)

        # Compute avf_s using element-wise multiplication
        avf_s = a_j * mapped_gs  # Shape: (number_of_pairs, F_atom)

        # Initialize tensor to accumulate radial contributions
        radial_contributions = torch.zeros(
            (number_of_atoms, avf_s.shape[-1]),
            device=avf_s.device,
            dtype=avf_s.dtype,
        )
        # Aggregate per atom
        radial_contributions.index_add_(0, idx_i, avf_s)

        return radial_contributions

    def calculate_vector_contributions(
        self,
        gv: Tensor,
        a_j: Tensor,
        idx_i: Tensor,
        agh: Tensor,
        number_of_atoms: int,
        device: torch.device,
    ) -> Tensor:
        """
        Compute vector (angular) contributions for each atom based on pair interactions.

        Parameters
        ----------
        gv : Tensor
            Vector symmetry functions with shape (number_of_pairs, 3, G).
        a_j : Tensor
            Neighbour (atom j) features for each pair with shape (number_of_pairs, F_atom).
        idx_i : Tensor
            Index of the central atom for each pair, with shape (number_of_pairs,).
        agh : Tensor
            Transformation tensor with shape (F_atom, G, H).
        number_of_atoms : int
            Total number of atoms in the system.
        device : torch.device
            The device to perform computations on.

        Returns
        -------
        Tensor
            Vector contributions aggregated per atom, with shape (number_of_atoms, H).
        """
        # Compute per-pair vector contributions
        # avf_v: (number_of_pairs, H, 3)
        avf_v = torch.einsum("pa, pdg, agh -> phd", a_j, gv, agh)

        # Initialize tensor to accumulate vector contributions per atom
        avf_v_sum = torch.zeros(
            (number_of_atoms, avf_v.shape[1], avf_v.shape[2]),
            device=device,
            dtype=avf_v.dtype,
        )
        # Aggregate per atom by summing the vectors
        avf_v_sum.index_add_(0, idx_i, avf_v)  # Shape: (number_of_atoms, H, 3)

        # Smoothed norm over the vector components: sqrt(|v|^2 + eps^2) - eps.
        # Rotation invariant, smooth at v = 0 (e.g., symmetric sites), with
        # curvature capped at 1/eps for stable force training.
        eps = 1e-4 # sits between the round-off at symmetric sites (~1e-8) and typical features (RMS about 0.24)
        vector_contributions = (
            torch.sqrt(avf_v_sum.pow(2).sum(dim=-1) + eps**2) - eps
        )  # Shape: (number_of_atoms, H)

        # raise an error if we have NaN values in the vector contribution as this will cause problems later on
        if torch.isnan(vector_contributions).any():
            raise ValueError("NaN values detected in vector contributions.")
        return vector_contributions

    def calculate_contributions(
        self,
        atomic_embedding: Tensor,
        pair_indices: Tensor,
        gs: Tensor,
        gv: Tensor,
        agh: Tensor,
        calculate_vector_contributions: bool,
    ) -> Tuple[Tensor, Tensor]:
        # central atom i = pair_indices[0] receives the sum over its
        # neighbours j = pair_indices[1], using the neighbours' features
        idx_i, idx_j = pair_indices[0], pair_indices[1]
        a_j = atomic_embedding[idx_j]  # Shape: (number_of_pairs, F_atom)

        radial_contributions = self.calculate_radial_contributions(
            gs,
            a_j,
            atomic_embedding.shape[0],
            idx_i,
        )

        if calculate_vector_contributions:
            vector_contributions = self.calculate_vector_contributions(
                gv,
                a_j,
                idx_i,
                agh,
                number_of_atoms=atomic_embedding.shape[0],
                device=atomic_embedding.device,
            )
        else:
            # Return zeros with shape (number_of_atoms, number_of_vector_features)
            vector_contributions = torch.zeros(
                (atomic_embedding.shape[0], self.number_of_vector_features),
                dtype=atomic_embedding.dtype,
                device=atomic_embedding.device,
            )

        return radial_contributions, vector_contributions

    def forward(
        self,
        atomic_embedding: Tensor,
        partial_charges: Tensor,
        pair_indices: Tensor,
        gs: Tensor,
        gv: Tensor,
        agh: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Calculate contributions from embeddings
        radial_contributions_emb, vector_contributions_emb = (
            self.calculate_contributions(
                atomic_embedding,
                pair_indices,
                gs,
                gv,
                agh,
                calculate_vector_contributions=True,
            )
        )

        if not self.is_first_module:
            # Calculate contributions from charges
            radial_contributions_charge, vector_contributions_charge = (
                self.calculate_contributions(
                    partial_charges,
                    pair_indices,
                    gs,
                    gv,
                    agh,
                    calculate_vector_contributions=False,
                )
            )
            # Combine messages
            combined_message = torch.cat(
                [
                    radial_contributions_emb,  # (N, F_atom)
                    vector_contributions_emb,  # (N, H)
                    radial_contributions_charge,  # (N, 1)
                    vector_contributions_charge,  # (N, H)
                ],
                dim=1,
            )
        else:
            combined_message = torch.cat(
                [
                    radial_contributions_emb,  # (N, F_atom)
                    vector_contributions_emb,  # (N, H)
                ],
                dim=1,
            )

        # Pass combined message through single MLP
        out = self.mlp(combined_message)

        # Split the output tensor into delta_q, f, and delta_a
        delta_q, f, delta_a = torch.split(
            out, [1, 1, self.number_of_per_atom_features], dim=1
        )

        return delta_a, delta_q, f


class AIMNet2Representation(nn.Module):
    def __init__(
        self,
        radial_cutoff: float,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        """
        Initialize the AIMNet2 representation layer.

        Parameters
        ----------
        radial_cutoff : float
            The cutoff distance for the radial symmetry function in nanometer.
        number_of_radial_basis_functions : int
            Number of radial basis functions to use.
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for the featurization process.
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        # Initialize cutoff module
        from modelforge.potential import CosineAttenuationFunction
        from modelforge.potential.featurization import FeaturizeInput

        self.featurize_input = FeaturizeInput(featurization_config)
        self.cutoff_module = CosineAttenuationFunction(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: float, number_of_radial_basis_functions: int
    ):
        from modelforge.potential import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        data : NNPInput
            The input data including atomic positions and numbers.
        pairlist_output : PairlistData
            Pairwise distances between atoms and pair indices.

        Returns
        -------
        Dict[str, torch.Tensor]
            The radial basis functions and atomic embeddings.
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(pairlist_output.d_ij)
        # Apply cutoff function to radial basis
        f_cutoff = self.cutoff_module(pairlist_output.d_ij)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(
                data
            ),  # add per-atom properties and embedding
        }

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from loguru import logger as log

from modelforge.potential.utils import Dense

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.neighbors import PairlistData


class AimNet2Core(torch.nn.Module):
    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_vector_features: int,
        number_of_interaction_modules: int,
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[str],
        predicted_dim: List[int],
        maximum_interaction_radius: float,
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
        number_of_interaction_modules : int
            Number of interaction modules in the model, determining the depth of
            message passing.
        activation_function_parameter : Dict[str, str]
            Configuration of activation functions used across the model.
        predicted_properties : List[str]
            List of properties that the model is predicting (e.g., energy,
            forces).
        predicted_dim : List[int]
            The dimensionality of each predicted property.
        maximum_interaction_radius : float
            The cutoff radius for atomic interactions in the model.
        """

        super().__init__()

        log.debug("Initializing the AimNet2 architecture.")

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

        self.agh = nn.Parameter(
            torch.randn(
                number_of_per_atom_features,  # F_atom
                number_of_radial_basis_functions,  # G
                number_of_vector_features,  # H
            )
        )
        # shape(nr_of_angular_symmetry_functions,nr_of_radial_symmetry_functions,nr_of_vector_features)

        # Define interaction modules for message passing
        self.interaction_modules = torch.nn.ModuleList(
            [
                AIMNet2InteractionModule(
                    number_of_per_atom_features=number_of_per_atom_features,
                    number_of_vector_features=number_of_vector_features,
                    activation_function=self.activation_function,
                    is_first_module=(i == 0),
                )
                for i in range(number_of_interaction_modules)
            ]
        )
        # Define output layers to calculate per-atom predictions
        self.output_layers = nn.ModuleDict()
        for property, dim in zip(predicted_properties, predicted_dim):
            self.output_layers[property] = nn.Sequential(
                Dense(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                ),
                Dense(
                    number_of_per_atom_features,
                    int(dim),
                ),
            )
        from modelforge.potential.processing import ChargeConservation

        self.charge_conservation = ChargeConservation()

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
            (atomic_embedding.shape[0], 1), device=atomic_embedding.device
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

            partial_charges = self.charge_conservation(
                {
                    "per_atom_charge": partial_charges,
                    "per_system_total_charge": data.per_system_total_charge.to(
                        dtype=torch.float32
                    ),
                    "atomic_subsystem_indices": data.atomic_subsystem_indices.to(
                        dtype=torch.int64
                    ),
                }
            )["per_atom_charge"]

        return {
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
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
        pairlist_output : PairListOutputs
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

        # Compute all specified outputs
        for output_name, output_layer in self.output_layers.items():
            output = output_layer(atomic_embedding)
            results[output_name] = output

        return results


import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AIMNet2InteractionModule(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_vector_features: int,
        activation_function: nn.Module,
        is_first_module: bool = False,
    ):
        super().__init__()
        self.is_first_module = is_first_module
        self.number_of_per_atom_features = number_of_per_atom_features
        self.number_of_vector_features = number_of_vector_features

        if not self.is_first_module:
            self.number_of_input_features = (
                number_of_per_atom_features  # radial_contributions_emb
                + number_of_vector_features  # vector_contributions_emb
                + 1  # radial_contributions_charge (from charges)
                + number_of_vector_features  # vector_contributions_charge
            )
        else:
            self.number_of_input_features = (
                number_of_per_atom_features  # radial_contributions_emb
                + number_of_vector_features  # vector_contributions_emb
            )

        # Single MLP producing combined outputs
        self.mlp = nn.Sequential(
            Dense(
                in_features=self.number_of_input_features,
                out_features=128,
                activation_function=activation_function,
            ),
            Dense(
                in_features=128,
                out_features=128,
                activation_function=activation_function,
            ),
            Dense(
                in_features=128,
                out_features=number_of_per_atom_features + 2,  # delta_q, f, delta_a
            ),
        )

    def calculate_radial_contributions(
        self,
        gs: Tensor,
        a_j: Tensor,
        number_of_atoms: int,
        idx_j: Tensor,
    ) -> Tensor:
        """
        Compute radial contributions for each atom based on pair interactions.

        Parameters
        ----------
        gs : Tensor
            Radial symmetry functions with shape (number_of_pairs, G).
        a_j : Tensor
            Atomic features for each pair with shape (number_of_pairs,
            F_atom).
        number_of_atoms : int
            Total number of atoms in the system.
        idx_j : Tensor
            Indices mapping each pair to an atom, with shape
            (number_of_pairs,).

        Returns
        -------
        Tensor
            Radial contributions aggregated per atom, with shape
            (number_of_atoms, F_atom).
        """
        # Compute radial contributions
        avf_s = gs.unsqueeze(-1) * a_j.unsqueeze(1)  # (number_of_pairs, G, F_atom)

        # Sum over G (if necessary)
        avf_s = avf_s.sum(dim=1)  # Adjust if needed

        # Initialize tensor to accumulate radial contributions
        radial_contributions = torch.zeros(
            (number_of_atoms, avf_s.shape[-1]),
            device=avf_s.device,
            dtype=avf_s.dtype,
        )
        radial_contributions.index_add_(0, idx_j, avf_s)

        return radial_contributions

    def calculate_vector_contributions(
        self,
        gv: Tensor,
        a_j: Tensor,
        idx_j: Tensor,
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
            Atomic features for each pair with shape (number_of_pairs, F_atom).
        idx_j : Tensor
            Indices mapping each pair to an atom, with shape
            (number_of_pairs,).
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
        # Compute vector contributions using adjusted Einstein summation
        avf_v = torch.einsum("pa, pdg, agh -> phd", a_j, gv, agh)
        # avf_v: Shape (number_of_pairs, H, 3)

        # Compute squared sum over vector components (d)
        avf_v_squared = torch.sum(avf_v.pow(2), dim=-1)  # Shape: (number_of_pairs, H)

        # Initialize the output tensor and aggregate per atom
        vector_contributions = torch.zeros(
            (number_of_atoms, avf_v_squared.shape[-1]),
            device=device,
            dtype=avf_v_squared.dtype,
        )
        vector_contributions.index_add_(0, idx_j, avf_v_squared)

        return vector_contributions

    def calculate_contributions(
        self,
        atomic_embedding: Tensor,
        pair_indices: Tensor,
        gs: Tensor,
        gv: Tensor,
        agh: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        idx_j = pair_indices[1]
        a_j = atomic_embedding[idx_j]  # Shape: (number_of_pairs, F_atom)

        radial_contributions = self.calculate_radial_contributions(
            gs,
            a_j,
            atomic_embedding.shape[0],
            idx_j,
        )

        if agh is not None:
            vector_contributions = self.calculate_vector_contributions(
                gv,
                a_j,
                idx_j,
                agh,
                number_of_atoms=atomic_embedding.shape[0],
                device=atomic_embedding.device,
            )
        else:
            # Return zeros with shape (number_of_atoms, number_of_vector_features)
            vector_contributions = torch.zeros(
                (atomic_embedding.shape[0], self.number_of_vector_features),
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
                    agh=None,  # No `agh` for charges
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

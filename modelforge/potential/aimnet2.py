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

        # Define interaction modules for message passing
        self.interaction_modules = torch.nn.ModuleList(
            [
                AIMNet2InteractionModule(
                    number_of_input_features=(
                        2 * (number_of_per_atom_features + 1)
                        if i > 0
                        else number_of_per_atom_features + 1
                    ),
                    number_of_per_atom_features=number_of_per_atom_features,
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
        r_ij, d_ij, f_ij, f_cutoff = (
            rep["r_ij"],
            rep["d_ij"],
            rep["f_ij"],
            rep["f_cutoff"],
        )
        # Scalar Gaussian expansion for radial terms
        f_ij_cutoff = f_ij * f_cutoff
        # Unit direction vectors u_ij = r_ij / d_ij (transpose for correct broadcasting)
        u_ij = r_ij.transpose(-1, -2).contiguous() / d_ij.unsqueeze(-2)
        gv = f_ij_cutoff.unsqueeze(-2) * u_ij.unsqueeze(
            -3
        )  # Single basis vector interaction

        # Atomic embedding "a" Eqn. (3)
        partial_charges = torch.zeros(
            (atomic_embedding.shape[0], 1), device=atomic_embedding.device
        )

        # Perform message passing using interaction modules
        for interaction in self.interaction_modules:

            delta_a, delta_q = interaction(
                atomic_embedding,
                pairlist.pair_indices,
                f_ij_cutoff,
                u_ij,
                partial_charges,
            )

            # Update atomic embeddings and partial charges
            atomic_embedding = atomic_embedding + delta_a
            partial_charges = partial_charges + delta_q

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



class AIMNet2InteractionModule(nn.Module):
    def __init__(
        self,
        number_of_input_features: int,
        number_of_per_atom_features: int,
        activation_function: nn.Module,
        is_first_module: bool = False,
    ):
        super().__init__()
        self.is_first_module = is_first_module
        self.shared_layers = nn.Sequential(
            Dense(
                in_features=number_of_input_features,
                out_features=128,
                activation_function=activation_function,
            ),
            Dense(
                in_features=128,
                out_features=64,
                activation_function=activation_function,
            ),
        )
        self.delta_a_mlp = nn.Sequential(
            self.shared_layers,
            Dense(
                in_features=64, out_features=32, activation_function=activation_function
            ),
            Dense(in_features=32, out_features=number_of_per_atom_features),
        )
        self.delta_q_mlp = nn.Sequential(
            self.shared_layers,
            Dense(
                in_features=64, out_features=32, activation_function=activation_function
            ),
            Dense(in_features=32, out_features=1),
        )

    def calculate_contributions(
        self,
        per_atom_feature_tensor,
        pair_indices,
        f_ij_cutoff,
        u_ij,
        use_charge_layer=False,
    ):
        idx_j = pair_indices[1]
        proto = f_ij_cutoff * per_atom_feature_tensor[idx_j]
        radial_contributions = torch.zeros(
            (per_atom_feature_tensor.shape[0], proto.shape[-1]),
            device=per_atom_feature_tensor.device,
            dtype=per_atom_feature_tensor.dtype,
        )
        radial_contributions.index_add_(0, idx_j, proto)
        vector_prot_step1 = u_ij.unsqueeze(-1) * f_ij_cutoff.unsqueeze(-2)
        vector_prot_step2 = vector_prot_step1 * per_atom_feature_tensor[
            idx_j
        ].unsqueeze(1)
        vector_prot_step2 = vector_prot_step2.sum(dim=-1)
        vector_contributions = torch.zeros(
            per_atom_feature_tensor.shape[0],
            3,
            device=per_atom_feature_tensor.device,
            dtype=vector_prot_step2.dtype,
        )
        vector_contributions.index_add_(0, idx_j, vector_prot_step2)
        return radial_contributions, torch.norm(vector_contributions, p=2, dim=1)

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        partial_charges: torch.Tensor,
        pair_indices: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        radial_contributions_emb, vector_contributions_emb = (
            self.calculate_contributions(
                atomic_embedding,
                pair_indices,
                f_ij_cutoff,
                r_ij,
                use_charge_layer=False,
            )
        )
        if not self.is_first_module:
            radial_contributions_charge, vector_contributions_charge = (
                self.calculate_contributions(
                    partial_charges,
                    pair_indices,
                    f_ij_cutoff,
                    r_ij,
                    use_charge_layer=True,
                )
            )
            combined_message = torch.cat(
                [
                    torch.cat(
                        [
                            radial_contributions_emb,
                            vector_contributions_emb.unsqueeze(1),
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            radial_contributions_charge,
                            vector_contributions_charge.unsqueeze(1),
                        ],
                        dim=1,
                    ),
                ],
                dim=1,
            )
        else:
            combined_message = torch.cat(
                [radial_contributions_emb, vector_contributions_emb.unsqueeze(1)], dim=1
            )
        delta_a = self.delta_a_mlp(combined_message)
        delta_q = self.delta_q_mlp(combined_message)
        return delta_a, delta_q


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

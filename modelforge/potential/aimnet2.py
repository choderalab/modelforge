from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData, Dense
from .models import NNPInput, BaseNetwork, CoreNetwork


@dataclass
class AIMNet2NeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for AIMNet2-based neural network potentials, including the necessary
    geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff
    (`f_cutoff`) to accurately represent atomistic systems for energy predictions.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A 2D tensor of shape [2, num_pairs], indicating the indices of atom pairs within a molecule or system.
    d_ij : torch.Tensor
        A 1D tensor containing the distances between each pair of atoms identified in `pair_indices`. Shape: [num_pairs, 1].
    r_ij : torch.Tensor
        A 2D tensor of shape [num_pairs, 3], representing the displacement vectors between each pair of atoms.
    number_of_atoms : int
        A integer indicating the number of atoms in the batch.
    positions : torch.Tensor
        A 2D tensor of shape [num_atoms, 3], representing the XYZ coordinates of each atom within the system.
    atomic_numbers : torch.Tensor
        A 1D tensor containing atomic numbers for each atom, used to identify the type of each atom in the system(s).
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule, useful for systems involving multiple
        molecules or distinct subsystems.
    total_charge : torch.Tensor
        A tensor with the total charge of each system or molecule. Shape: [num_systems], where each entry corresponds
        to a distinct system or molecule.
    atomic_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived from atomic numbers.
        Shape: [num_atoms, embedding_dim], where `embedding_dim` is the dimensionality of the embedding vectors.
    f_ij : Optional[torch.Tensor]
        A tensor representing the radial symmetry function expansion of distances between atom pairs, capturing the
        local chemical environment. Shape: [num_pairs, num_features], where `num_features` is the dimensionality of
        the radial symmetry function expansion. This field will be populated after initialization.
    f_cutoff : Optional[torch.Tensor]
        A tensor representing the cosine cutoff function applied to the radial symmetry function expansion, ensuring
        that atom pair contributions diminish smoothly to zero at the cutoff radius. Shape: [num_pairs]. This field
        will be populated after initialization.


    """

    f_ij: Optional[torch.Tensor] = field(default=None)
    f_cutoff: Optional[torch.Tensor] = field(default=None)


from typing import Union, Type


class AIMNet2Core(CoreNetwork):
    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        activation_function: Type[torch.nn.Module],
        maximum_interaction_radius: unit.Quantity,
    ) -> None:
        """
        Initialize the AIMNet2Core class.

        Parameters
        ----------
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for the featurization process.
        number_of_radial_basis_functions : int
            Number of radial basis functions used.
        number_of_interaction_modules : int
            Number of interaction modules used in the network.
        activation_function : Type[torch.nn.Module]
            Activation function to be used in the network.
        maximum_interaction_radius : openff.units.unit.Quantity
            The maximum interaction radius for the network.
        """

        log.debug("Initializing the AimNet2 architecture.")

        super().__init__(activation_function)

        # Initialize representation block
        self.representation_module = AIMNet2Representation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )
        number_of_per_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )

        self.interaction_modules = torch.nn.ModuleList(
            [
                AIMNet2Interaction(
                    FirstMessageModule(number_of_per_atom_features),
                    number_of_input_features=number_of_per_atom_features + 1,
                    number_of_per_atom_features=number_of_per_atom_features,
                    activation_function=activation_function,
                ),
                AIMNet2Interaction(
                    SubsequentMessageModule(number_of_per_atom_features),
                    number_of_input_features=2 * (number_of_per_atom_features + 1),
                    number_of_per_atom_features=number_of_per_atom_features,
                    activation_function=activation_function,
                ),
                AIMNet2Interaction(
                    SubsequentMessageModule(number_of_per_atom_features),
                    number_of_input_features=2 * (number_of_per_atom_features + 1),
                    number_of_per_atom_features=number_of_per_atom_features,
                    activation_function=activation_function,
                ),
                AIMNet2Interaction(
                    SubsequentMessageModule(number_of_per_atom_features),
                    number_of_input_features=2 * (number_of_per_atom_features + 1),
                    number_of_per_atom_features=number_of_per_atom_features,
                    activation_function=activation_function,
                ),
            ]
        )
        # output layer to obtain per-atom energies
        self.energy_layer = nn.Sequential(
            Dense(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation_function=self.activation_function,
            ),
            Dense(
                number_of_per_atom_features,
                1,
            ),
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output
    ) -> AIMNet2NeuralNetworkData:
        """
        Prepare the model-specific input.

        Parameters
        ----------
        data : NNPInput
            The input data containing atomic details.
        pairlist_output : object
            The output of pair list calculations containing pair indices, distances, and displacement vectors.

        Returns
        -------
        AIMNet2NeuralNetworkData
            The structured input data for AIMNet2.
        """

        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = AIMNet2NeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
        )

        return nnp_input

    def compute_properties(
        self, data: AIMNet2NeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the requested properties for a given input batch.

        Parameters
        ----------
        data : AIMNet2NeuralNetworkData
            The input data structured for the AIMNet2 network.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated energies and forces.
        """

        representation = self.representation_module(data)

        data.f_ij = representation["f_ij"]
        data.f_cutoff = representation["f_cutoff"]
        f_ij_cutoff = torch.mul(data.f_ij, data.f_cutoff)
        # Atomic embedding "a" Eqn. (3)
        atomic_embedding = representation["atomic_embedding"]
        partial_point_charges = torch.zeros(
            (atomic_embedding.shape[0], 1), device=atomic_embedding.device
        )

        # Generate message passing output
        for i in range(len(self.interaction_modules)):

            interaction = self.interaction_modules[i]
            delta_a, delta_q = interaction(
                atomic_embedding,
                partial_point_charges,
                data.pair_indices,
                f_ij_cutoff,
                data.r_ij,
            )

            # Update atomic embeddings and partial charges
            atomic_embedding = atomic_embedding + delta_a
            partial_point_charges = partial_point_charges + delta_q

        E_i = self.energy_layer(atomic_embedding).squeeze(1)

        return {
            "per_atom_energy": E_i,
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


class BaseMessageModule(nn.Module):
    def __init__(self, number_of_per_atom_features: int):
        """
        Initialize the BaseMessageModule.

        Parameters
        ----------
        number_of_per_atom_features : int
            The number of features per atom.
        """
        super().__init__()
        self.number_of_per_atom_features = number_of_per_atom_features

        # Separate linear layers for embeddings and charges
        self.linear_transform_embeddings = nn.Linear(
            number_of_per_atom_features, number_of_per_atom_features
        )
        self.linear_transform_charges = nn.Linear(
            1, number_of_per_atom_features
        )  # For partial charges

    def calculate_contributions(
        self,
        per_atom_feature_tensor: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
        linear_transform: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the radial and vector contributions for the given features.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            The feature tensor (either atomic embeddings or repeated partial charges).
        pairlist : torch.Tensor
            The list of atom pairs.
        f_ij_cutoff : torch.Tensor
            The cutoff function applied to the radial symmetry functions.
        r_ij : torch.Tensor
            The displacement vectors between atom pairs.
        linear_transform : nn.Module
            The linear transformation to apply to the features.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Radial and vector contributions.
        """

        idx_j = pairlist[1]

        # Calculate the unit vector u_ij
        r_ij_norm = torch.norm(r_ij, dim=1, keepdim=True)  # Shape: (num_atom_pairs, 1)
        u_ij = r_ij / r_ij_norm  # Shape: (num_atom_pairs, 3)

        # Step 1: Radial Contributions Calculation (Equation 4)
        proto_v_r_a = (
            f_ij_cutoff * per_atom_feature_tensor[idx_j]
        )  # Shape: (num_atom_pairs, nr_of_features)

        # Initialize tensor to accumulate radial contributions for each atom
        radial_contributions = torch.zeros(
            (per_atom_feature_tensor.shape[0], self.number_of_per_atom_features),
            device=per_atom_feature_tensor.device,
            dtype=per_atom_feature_tensor.dtype,
        )  # Shape: (num_of_atoms, nr_of_features)

        # Accumulate the radial contributions using index_add_
        radial_contributions.index_add_(0, idx_j, proto_v_r_a)

        # Step 2: Vector Contributions Calculation (Equation 5)
        # First, calculate the directional component by multiplying g_ij with u_ij
        vector_prot_step1 = u_ij.unsqueeze(-1) * f_ij_cutoff.unsqueeze(
            -2
        )  # Shape: (num_atom_pairs, 3, nr_of_features)

        # Next, multiply this result by the input of atom j
        vector_prot_step2 = vector_prot_step1 * per_atom_feature_tensor[
            idx_j
        ].unsqueeze(
            1
        )  # Shape: (num_atom_pairs, 3, nr_of_features)

        # Sum over the last dimension (nr_of_features) to reduce it
        vector_prot_step2 = vector_prot_step2.sum(dim=-1)  # Shape: (num_atom_pairs, 3)

        # Initialize tensor to accumulate vector contributions for each atom
        vector_contributions = torch.zeros(
            per_atom_feature_tensor.shape[0], 3
        )  # Shape: (num_of_atoms, 3)

        # Accumulate the vector contributions using index_add_
        vector_contributions.index_add_(0, idx_j, vector_prot_step2)

        # Step 3: Compute the Euclidean Norm for each atom
        vector_norms = torch.norm(
            vector_contributions, p=2, dim=1
        )  # Shape: (num_of_atoms,)

        return radial_contributions, vector_norms


class FirstMessageModule(BaseMessageModule):
    def __init__(self, number_of_per_atom_features: int):
        """
        Initialize the FirstMessageModule.

        Parameters
        ----------
        number_of_per_atom_features : int
            The number of features per atom.
        """
        super().__init__(number_of_per_atom_features)
        self.linear_transform_embeddings = nn.Linear(
            number_of_per_atom_features, number_of_per_atom_features
        )

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        partial_charges: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the message module.

        Parameters
        ----------
        atomic_embedding : torch.Tensor
            The embedding of each atom.
        partial_charges : torch.Tensor
            The partial charges of each atom.
        pairlist : torch.Tensor
            The list of atom pairs.
        f_ij_cutoff : torch.Tensor
            The cutoff function applied to the radial symmetry functions.
        r_ij : torch.Tensor
            The displacement vectors between atom pairs.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Updated atomic embeddings and partial charges.
        """

        # Step 1: Calculate radial and vector contributions for atomic embeddings (Equation 4 and 5)
        radial_contributions_emb, vector_contributions_emb = (
            self.calculate_contributions(
                atomic_embedding,
                pairlist,
                f_ij_cutoff,
                r_ij,
                self.linear_transform_embeddings,
            )
        )

        # Step 3: Combine contributions
        feature_vector = torch.cat(
            [radial_contributions_emb, vector_contributions_emb.unsqueeze(-1)], dim=1
        )

        return feature_vector


class SubsequentMessageModule(BaseMessageModule):
    def __init__(self, number_of_per_atom_features: int):
        """
        Initialize the SubsequentMessageModule.

        Parameters
        ----------
        number_of_per_atom_features : int
            The number of features per atom.
        """
        super().__init__(number_of_per_atom_features)
        self.linear_transform_embeddings = nn.Linear(
            number_of_per_atom_features, number_of_per_atom_features
        )
        self.linear_transform_charges = nn.Linear(
            1, number_of_per_atom_features
        )  # For partial charges

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        partial_charges: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the message module.

        Parameters
        ----------
        atomic_embedding : torch.Tensor
            The embedding of each atom.
        partial_charges : torch.Tensor
            The partial charges of each atom.
        pairlist : torch.Tensor
            The list of atom pairs.
        f_ij_cutoff : torch.Tensor
            The cutoff function applied to the radial symmetry functions.
        r_ij : torch.Tensor
            The displacement vectors between atom pairs.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Updated atomic embeddings and partial charges.
        """

        # Step 1: Calculate radial and vector contributions for atomic embeddings (Equation 4 and 5)
        a = 7
        radial_contributions_emb, vector_contributions_emb = (
            self.calculate_contributions(
                atomic_embedding,
                pairlist,
                f_ij_cutoff,
                r_ij,
                self.linear_transform_embeddings,
            )
        )
        a = 7
        # Step 2: Calculate radial and vector contributions for partial charges (Equation 4 and 5)
        radial_contributions_charge, vector_contributions_charge = (
            self.calculate_contributions(
                partial_charges,
                pairlist,
                f_ij_cutoff,
                r_ij,
                self.linear_transform_charges,
            )
        )

        # Step 3: Combine contributions
        feature_vector_emb = torch.cat(
            [radial_contributions_emb, vector_contributions_emb.unsqueeze(1)], dim=1
        )
        feature_vector_charge = torch.cat(
            [radial_contributions_charge, vector_contributions_charge.unsqueeze(1)],
            dim=1,
        )

        return torch.cat([feature_vector_emb, feature_vector_charge], dim=1)


class AIMNet2Interaction(nn.Module):
    def __init__(
        self,
        message_module: Type[torch.nn.Module],
        number_of_input_features: int,
        number_of_per_atom_features: int,
        activation_function: Type[torch.nn.Module],
    ):
        """
        Initialize the AIMNet2Interaction module.

        Parameters
        ----------
        message_module : nn.Module
            The message passing module to be used.
        number_of_per_atom_features : int
            The number of features per atom.

        """
        super().__init__()
        self.message_module = message_module
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
                in_features=64,
                out_features=32,
                activation_function=activation_function,
            ),
            Dense(
                in_features=32,
                out_features=number_of_per_atom_features,
            ),
        )
        self.delta_q_mlp = nn.Sequential(
            self.shared_layers,
            Dense(
                in_features=64,
                out_features=32,
                activation_function=activation_function,
            ),
            Dense(
                in_features=32,
                out_features=1,
            ),
        )

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
        partial_point_charges: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the AIMNet2Interaction module.

        Parameters
        ----------
        atomic_embedding : torch.Tensor
            The embedding of each atom.
        pairlist : torch.Tensor
            The list of atom pairs.
        f_ij_cutoff : torch.Tensor
            The cutoff function applied to the radial symmetry functions.
        r_ij : torch.Tensor
            The displacement vectors between atom pairs.
        partial_point_charges : Optional[torch.Tensor], optional
            The partial point charges for atoms, by default None.

        Returns
        -------
        torch.Tensor
            The result of the message passing.
        """
        combined_message = self.message_module(
            atomic_embedding, pairlist, f_ij_cutoff, r_ij, partial_point_charges
        )

        delta_a = self.delta_a_mlp(combined_message)
        delta_q = self.delta_q_mlp(combined_message)

        return delta_a, delta_q


class AIMNet2Representation(nn.Module):
    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Union[List[str], int]],
    ):
        """
        Initialize the AIMNet2 representation layer.

        Parameters
        ----------
        radial_cutoff : openff.units.unit.Quantity
            The cutoff distance for the radial symmetry function.
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
        from modelforge.potential import CosineAttenuationFunction, FeaturizeInput

        self.featurize_input = FeaturizeInput(featurization_config)
        self.cutoff_module = CosineAttenuationFunction(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: unit.Quantity, number_of_radial_basis_functions: int
    ):
        from .utils import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, data: Type[AIMNet2NeuralNetworkData]) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        d_ij : Pairwise distances between atoms; shape [n_pairs, 1]

        Returns
        -------
        Radial basis functions for pairs of atoms; shape [n_pairs, 1, number_of_radial_basis_functions]
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(
            data.d_ij
        )  # shape (n_pairs, 1, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(data.d_ij)  # shape (n_pairs, 1)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(
                data
            ),  # add per-atom properties and embedding
        }


from typing import List


class AIMNet2(BaseNetwork):
    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        maximum_interaction_radius: Union[unit.Quantity, str],
        activation_function_parameter: Dict,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        potential_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the AIMNet2 network.

        # NOTE: set correct reference

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_per_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        number_of_interaction_modules : int, default=2
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """
        self.only_unique_pairs = False  # NOTE: need to be set before super().__init__
        from modelforge.utils.units import _convert_str_to_unit

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            potential_seed=potential_seed,
        )

        activation_function = activation_function_parameter["activation_function"]

        self.core_module = AIMNet2Core(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            activation_function=activation_function,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
        )

    def _config_prior(self):
        pass

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData
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
        Initialize the AIMNet2 class.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        number_of_interaction_modules : int, default=3
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """

        log.debug("Initializing the AimNet2 architecture.")

        super().__init__(activation_function)

        # Initialize representation block
        self.representation_module = AIMNet2Representation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )
        number_of_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )

        self.interaction_modules_first_pass = AIMNet2Interaction(
            FirstMessageModule(number_of_atom_features)
        )
        # Initialize the module as a SubsequentMessageModule
        self.interaction_modules_subsequent_pass = AIMNet2Interaction(
            SubsequentMessageModule(number_of_atom_features)
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output
    ) -> AIMNet2NeuralNetworkData:

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
        data : NamedTuple

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated energies; shape (nr_systems,).
        """

        representation = self.representation_module(data)

        data.f_ij = representation["f_ij"]
        data.f_cutoff = representation["f_cutoff"]
        f_ij_cutoff = torch.mul(data.f_ij, data.f_cutoff)
        # Atomic embedding "a" Eqn. (3)
        atomic_embedding = representation["atomic_embedding"]

        result_first = self.interaction_modules_first_pass(
            atomic_embedding, data.pair_indices, f_ij_cutoff, data.r_ij
        )

        result_second = self.interaction_modules_first_pass(
            atomic_embedding, data.pair_indices, f_ij_cutoff, data.r_ij
        )

        a = 7
        # first pass


import torch
import torch.nn as nn
from typing import Optional


class BaseMessageModule(nn.Module):
    def __init__(self, number_of_atom_features: int):
        super().__init__()
        self.number_of_atomic_features = number_of_atom_features
        self.linear = nn.Linear(number_of_atom_features, number_of_atom_features)

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
        partial_point_charges: Optional[torch.Tensor] = None,
    ):
        # Step 1: Calculate weighted embeddings
        weighted_embeddings = self.calculate_weighted_embeddings(
            atomic_embedding, pairlist, f_ij_cutoff, partial_point_charges
        )

        # Step 2: Calculate radial contributions (Equation 4)
        radial_contributions = self.calculate_radial_contributions(
            atomic_embedding, pairlist, weighted_embeddings
        )

        # Step 3: Calculate vector contributions (Equation 5)
        vector_norms = self.calculate_vector_contributions(
            pairlist, weighted_embeddings, r_ij
        )

        # Step 4: Combine radial and vector contributions to form the final message
        combined_message = self.combine_messages(vector_norms, radial_contributions)

        return combined_message

    def calculate_radial_contributions(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        weighted_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the radial contributions (Equation 4)."""
        idx_i = pairlist[0]
        radial_contributions = torch.zeros_like(
            atomic_embedding
        )  # Shape: (num_atoms, 16)
        radial_contributions.index_add_(
            0, idx_i, weighted_embeddings[:, : self.number_of_atomic_features]
        )
        return radial_contributions

    def calculate_vector_contributions(
        self,
        pairlist: torch.Tensor,
        weighted_embeddings: torch.Tensor,
        r_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the vector contributions and norms (Equation 5)."""
        idx_i = pairlist[0]

        # Calculate the unit vector u_ij
        r_ij_norm = torch.norm(r_ij, dim=1, keepdim=True)  # Shape: (num_atom_pairs, 1)
        u_ij = r_ij / r_ij_norm  # Shape: (num_atom_pairs, 3)

        # Multiply u_ij with weighted embeddings
        u_weighted_embeddings = u_ij.unsqueeze(-1) * weighted_embeddings.unsqueeze(-2)

        # Apply the linear transformation using nn.Linear
        transformed_embeddings = self.apply_linear_transformation(u_weighted_embeddings)

        # Sum over j to get the contributions for each i
        vector_contributions = torch.zeros(
            (weighted_embeddings.shape[0], 3, weighted_embeddings.shape[-1]),
            device=weighted_embeddings.device,
        )
        vector_contributions.index_add_(0, idx_i, transformed_embeddings)

        # Calculate the norm of the resulting vectors for each atom
        vector_norms = torch.norm(vector_contributions, dim=1)  # Shape: (num_atoms, 16)

        return vector_norms

    def apply_linear_transformation(
        self, u_weighted_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply the linear transformation to the weighted embeddings."""
        u_weighted_embeddings_flat = u_weighted_embeddings.view(
            -1, u_weighted_embeddings.shape[-1]
        )
        transformed_embeddings_flat = self.linear(u_weighted_embeddings_flat)
        transformed_embeddings = transformed_embeddings_flat.view(
            u_weighted_embeddings.shape
        )
        return transformed_embeddings

    def combine_messages(
        self, vector_norms: torch.Tensor, radial_contributions: torch.Tensor
    ) -> torch.Tensor:
        """Combine radial and vector contributions."""
        combined_message = torch.cat([vector_norms, radial_contributions], dim=-1)
        return combined_message

    def calculate_weighted_embeddings(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        partial_point_charges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses")


class FirstMessageModule(BaseMessageModule):
    def calculate_weighted_embeddings(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        partial_point_charges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the weighted embeddings for the first message round."""
        idx_j = pairlist[1]
        a_j = atomic_embedding[idx_j]  # Shape: (num_atom_pairs, 16)
        weighted_embeddings = f_ij_cutoff * a_j  # Element-wise multiplication
        return weighted_embeddings


class SubsequentMessageModule(BaseMessageModule):
    def __init__(self, number_of_atom_features: int):
        super().__init__(number_of_atom_features + 1)  # Adjust for partial charge

    def calculate_weighted_embeddings(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        partial_point_charges: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the weighted embeddings for subsequent message rounds."""
        idx_j = pairlist[1]
        a_j = atomic_embedding[idx_j]
        q_j = partial_point_charges[idx_j].unsqueeze(-1)  # Shape: (num_atom_pairs, 1)
        combined_features = torch.cat(
            [a_j, q_j], dim=-1
        )  # Combine embeddings with charges
        weighted_embeddings = f_ij_cutoff * combined_features
        return weighted_embeddings


class AIMNet2Interaction(nn.Module):
    def __init__(self, message_module: nn.Module):
        super().__init__()
        self.message_module = message_module

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
        r_ij: torch.Tensor,
        partial_point_charges: Optional[torch.Tensor] = None,
    ):
        return self.message_module(
            atomic_embedding, pairlist, f_ij_cutoff, r_ij, partial_point_charges
        )


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
        Radial Basis Function Module
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
        number_of_atom_features : int, default=64
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

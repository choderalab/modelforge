"""
SchNet neural network potential for modeling quantum interactions.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from loguru import logger as log

from modelforge.utils.prop import NNPInput
from modelforge.potential.neighbors import PairlistData


class SchNetCore(torch.nn.Module):
    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        maximum_interaction_radius: float,
        number_of_filters: int,
        activation_function_parameter: Dict[str, str],
        shared_interactions: bool,
        predicted_properties: List[str],
        predicted_dim: List[int],
        potential_seed: int = -1,
    ) -> None:
        """
        Core SchNet architecture for modeling quantum interactions between atoms.

        Parameters
        ----------
        featurization : Dict[str, Dict[str, int]]
            Configuration for atom featurization, including number of features per atom.
        number_of_radial_basis_functions : int
            Number of radial basis functions for the SchNet representation.
        number_of_interaction_modules : int
            Number of interaction modules to use.
        maximum_interaction_radius : float
            Maximum distance for interactions.
        number_of_filters : int
            Number of filters for interaction layers.
        activation_function_parameter : Dict[str, str]
            Dictionary containing the activation function to use.
        shared_interactions : bool
            Whether to share weights across all interaction modules.
        predicted_properties : List[str]
            List of properties to predict.
        predicted_dim : List[int]
            List of dimensions for each predicted property.
        potential_seed : int, optional
            Seed for random number generation, by default -1.
        """

        from modelforge.utils.misc import seed_random_number

        if potential_seed != -1:
            seed_random_number(potential_seed)

        super().__init__()
        self.activation_function = activation_function_parameter["activation_function"]

        log.debug("Initializing the SchNet architecture.")
        from modelforge.potential.utils import DenseWithCustomDist

        # Set the number of filters and atom features
        self.number_of_filters = number_of_filters or int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )

        # Initialize representation block for SchNet
        self.schnet_representation_module = SchNETRepresentation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization,
        )
        # Initialize interaction blocks, sharing or not based on config
        if shared_interactions:
            self.interaction_modules = nn.ModuleList(
                [
                    SchNETInteractionModule(
                        number_of_per_atom_features,
                        self.number_of_filters,
                        number_of_radial_basis_functions,
                        activation_function=self.activation_function,
                    )
                ]
                * number_of_interaction_modules
            )

        else:
            self.interaction_modules = nn.ModuleList(
                [
                    SchNETInteractionModule(
                        number_of_per_atom_features,
                        self.number_of_filters,
                        number_of_radial_basis_functions,
                        activation_function=self.activation_function,
                    )
                    for _ in range(number_of_interaction_modules)
                ]
            )

        # Initialize output layers based on predicted properties
        self.output_layers = nn.ModuleDict()
        for property, dim in zip(predicted_properties, predicted_dim):
            self.output_layers[property] = nn.Sequential(
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                ),
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    int(dim),
                ),
            )

    def compute_properties(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties based on the input data and pair list.

        Parameters
        ----------
        data : NNPInput
            Input data including atomic numbers, positions, etc.
        pairlist_output: PairlistData
            Output from the pairlist module, containing pair indices and
            distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the computed properties for each atom.
        """
        # Compute the atomic representation
        representation = self.schnet_representation_module(data, pairlist_output)
        atomic_embedding = representation["atomic_embedding"]
        f_ij = representation["f_ij"]
        f_cutoff = representation["f_cutoff"]

        # Apply interaction modules to update the atomic embedding
        for interaction in self.interaction_modules:
            atomic_embedding = atomic_embedding + interaction(
                atomic_embedding,
                pairlist_output,
                f_ij,
                f_cutoff,
            )

        return {
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the SchNet model.

        Parameters
        ----------
        data : NNPInput
            Input data including atomic numbers, positions, and relevant fields.
        pairlist_output : PairlistData
            Pair indices and distances from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of calculated properties from the forward pass.
        """
        # Compute properties using the core method
        results = self.compute_properties(data, pairlist_output)
        atomic_embedding = results["per_atom_scalar_representation"]

        # Apply output layers to the atomic embedding
        for output_name, output_layer in self.output_layers.items():
            results[output_name] = output_layer(atomic_embedding)

        return results


class SchNETInteractionModule(nn.Module):

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_filters: int,
        number_of_radial_basis_functions: int,
        activation_function: torch.nn.Module,
    ) -> None:
        """
        SchNet interaction module to compute interaction terms based on atomic
        distances and features.

        Parameters
        ----------
        number_of_per_atom_features : int
            Number of atom features, defines the dimensionality of the
            embedding.
        number_of_filters : int
            Number of filters, defines the dimensionality of the intermediate
            features.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        activation_function : torch.nn.Module
            The activation function to use in the interaction module.
        """

        super().__init__()
        from .utils import DenseWithCustomDist

        assert (
            number_of_radial_basis_functions > 4
        ), "Number of radial basis functions must be larger than 10."
        assert number_of_filters > 1, "Number of filters must be larger than 1."
        assert (
            number_of_per_atom_features > 10
        ), "Number of atom basis must be larger than 10."

        self.number_of_per_atom_features = (
            number_of_per_atom_features  # Initialize parameters
        )

        # Define input, filter, and output layers
        self.intput_to_feature = DenseWithCustomDist(
            number_of_per_atom_features,
            number_of_filters,
            bias=False,
        )
        self.feature_to_output = nn.Sequential(
            DenseWithCustomDist(
                number_of_filters,
                number_of_per_atom_features,
                activation_function=activation_function,
            ),
            DenseWithCustomDist(
                number_of_per_atom_features,
                number_of_per_atom_features,
            ),
        )
        self.filter_network = nn.Sequential(
            DenseWithCustomDist(
                number_of_radial_basis_functions,
                number_of_filters,
                activation_function=activation_function,
            ),
            DenseWithCustomDist(
                number_of_filters,
                number_of_filters,
            ),
        )

    def forward(
        self,
        atomic_embedding: torch.Tensor,
        pairlist: PairlistData,
        f_ij: torch.Tensor,
        f_ij_cutoff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        atomic_embedding : torch.Tensor
            Input feature tensor for atoms (output of embedding).
        pairlist : PairlistData
            List of atom pairs.
        f_ij : torch.Tensor, shape [n_pairs, number_of_radial_basis_functions]
            Radial basis functions for pairs of atoms.
        f_ij_cutoff : torch.Tensor, shape [n_pairs, 1]
            Cutoff values for the pairs.

        Returns
        -------
        torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Updated feature tensor after interaction block.
        """
        idx_i, idx_j = pairlist.pair_indices[0], pairlist.pair_indices[1]

        # Transform atomic embedding to filter space
        atomic_embedding = self.intput_to_feature(atomic_embedding)

        # Generate interaction filters based on radial basis functions
        W_ij = self.filter_network(f_ij.squeeze(1))
        W_ij = W_ij * f_ij_cutoff  # Shape: [n_pairs, number_of_filters]

        # Perform continuous-filter convolution
        x_j = atomic_embedding[idx_j]
        x_ij = x_j * W_ij  # Element-wise multiplication

        out = torch.zeros_like(atomic_embedding).scatter_add_(
            0, idx_i.unsqueeze(-1).expand_as(x_ij), x_ij
        )  # Aggregate per-atom pair to per-atom

        return self.feature_to_output(out)  # Output the updated atomic features


class SchNETRepresentation(nn.Module):

    def __init__(
        self,
        radial_cutoff: float,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        """
        SchNet representation module to generate the radial symmetry
        representation of pairwise distances.

        Parameters
        ----------
        radial_cutoff : float
            The cutoff distance for interactions in nanometer.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        featurization_config : Dict[str, Dict[str, int]]
            Configuration for atom featurization.
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
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate the radial symmetry representation of pairwise
        distances.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic numbers and positions.
        pairlist_output : PairlistData
            Output from the pairlist module, containing pair indices and distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing radial basis functions, cutoff values, and atomic embeddings.
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(pairlist_output.d_ij)

        # Apply cutoff function to distances
        f_cutoff = self.cutoff_module(pairlist_output.d_ij)  # shape (n_pairs, 1)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(data),
        }

"""
SchNet neural network potential for modeling quantum interactions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union, List

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData
from .models import PairListOutputs, NNPInput, BaseNetwork, CoreNetwork


class SchNet:
    def __init__(self) -> None:
        pass


@dataclass
class SchnetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for SchNet-based neuraletwork potentials, including the necessary
    geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff
    (`f_cutoff`) to accurately represent atomistic systems for energy predictions.


    Note that only the arguments not present in the baseclass are described here.

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

    atomic_embedding: Optional[torch.Tensor] = field(default=None)
    f_ij: Optional[torch.Tensor] = field(default=None)
    f_cutoff: Optional[torch.Tensor] = field(default=None)


class SchNetCore(CoreNetwork):
    def __init__(
        self,
        featurization_config: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_filters: int,
        shared_interactions: bool,
        activation_function: Type[torch.nn.Module],
        maximum_interaction_radius: unit.Quantity,
    ) -> None:

        super().__init__(activation_function)

        log.debug("Initializing the SchNet architecture.")
        from modelforge.potential.utils import DenseWithCustomDist

        self.number_of_filters = number_of_filters or int(
            featurization_config["number_of_per_atom_features"]
        )
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        number_of_per_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )

        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )
        # Initialize interaction blocks
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

        # output layer to obtain per-atom energies
        self.energy_layer = nn.Sequential(
            DenseWithCustomDist(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation_function=self.activation_function,
            ),
            DenseWithCustomDist(
                number_of_per_atom_features,
                1,
            ),
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: Dict[str, PairListOutputs]
    ) -> SchnetNeuralNetworkData:
        """
        Prepare the input data for the SchNet model.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist_output : Dict[str, PairListOutputs]
            The pairlist output(s).

        Returns
        -------
        SchnetNeuralNetworkData
            The prepared input data for the SchNet model.
        """
        number_of_atoms = data.atomic_numbers.shape[0]

        # Note, pairlist_output is a Dict where the key corresponds to the name of the cutoff parameter
        # e.g. "maximum_interaction_radius"

        pairlist_output = pairlist_output["maximum_interaction_radius"]

        nnp_input = SchnetNeuralNetworkData(
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
        self, data: SchnetNeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the properties for a given input batch.

        Parameters
        ----------
        data : SchnetNeuralNetworkData
            The input data for the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated properties.
        """
        # Compute the representation for each atom (transform to radial basis set, multiply by cutoff and add embedding)
        representation = self.schnet_representation_module(data)
        atomic_embedding = representation["atomic_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                atomic_embedding,
                data.pair_indices,
                representation["f_ij"],
                representation["f_cutoff"],
            )
            atomic_embedding = (
                atomic_embedding + v
            )  # Update per atom features given the environment

        E_i = self.energy_layer(atomic_embedding).squeeze(1)

        return {
            "per_atom_energy": E_i,
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

    def forward(
        self, data: NNPInput, pairlist_output: PairListOutputs
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
        # perform model specific modifications
        nnp_input = self._model_specific_input_preparation(data, pairlist_output)
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(nnp_input)
        # add atomic numbers to the output
        outputs["atomic_numbers"] = data.atomic_numbers

        return outputs


class SchNETInteractionModule(nn.Module):
    """
    SchNet interaction module to compute interaction terms based on atomic distances and features.

    Parameters
    ----------
    number_of_per_atom_features : int
        Number of atom features, defines the dimensionality of the embedding.
    number_of_filters : int
        Number of filters, defines the dimensionality of the intermediate features.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    activation_function: Type[torch.nn.Module]
        The activation function to use in the interaction module.
    """

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_filters: int,
        number_of_radial_basis_functions: int,
        activation_function: Type[torch.nn.Module],
    ) -> None:

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
        self.intput_to_feature = DenseWithCustomDist(
            number_of_per_atom_features,
            number_of_filters,
            bias=False,
            activation_function=None,
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
                activation_function=None,
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
                activation_function=None,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        pairlist: torch.Tensor,  # shape [n_pairs, 2]
        f_ij: torch.Tensor,  # shape [n_pairs, number_of_radial_basis_functions]
        f_ij_cutoff: torch.Tensor,  # shape [n_pairs, 1]
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Input feature tensor for atoms (output of embedding).
        pairlist : torch.Tensor, shape [n_pairs, 2]
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
        idx_i, idx_j = pairlist[0], pairlist[1]

        # Map input features to the filter space
        x = self.intput_to_feature(x)

        # Generate interaction filters based on radial basis functions
        W_ij = self.filter_network(f_ij.squeeze(1))  # FIXME
        W_ij = W_ij * f_ij_cutoff

        # Perform continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * W_ij  # (nr_of_atom_pairs, nr_atom_basis)
        out = torch.zeros_like(x)
        out.scatter_add_(
            0, idx_i.unsqueeze(-1).expand_as(x_ij), x_ij
        )  # from per_atom_pair to _per_atom

        return self.feature_to_output(out)  # shape: (nr_of_atoms, 1)


class SchNETRepresentation(nn.Module):
    """
    SchNet representation module to generate the radial symmetry representation of pairwise distances.

    Parameters
    ----------
    radial_cutoff : unit.Quantity
        The cutoff distance for interactions.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    featurization_config : Dict[str, Union[List[str], int]]
        Configuration for atom featurization.
    """

    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Union[List[str], int]],
    ):
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

    def forward(self, data: Type[SchnetNeuralNetworkData]) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        data : SchnetNeuralNetworkData

        Returns
        -------
        Dict[str, torch.Tensor]
            Radial basis functions, cutoff values for pairs of atoms and atomic embedding.
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(
            data.d_ij
        )  # shape (n_pairs, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(data.d_ij)  # shape (n_pairs, 1)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(
                data
            ),  # add per-atom properties and embedding
        }


from typing import List, Union

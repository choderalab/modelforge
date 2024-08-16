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
    """
    Core network class for the SchNet neural network potential.

    Parameters
    ----------
    featurization_config : Dict[str, Union[List[str], int]]
        Configuration for featurization, including the number of per-atom features and the maximum atomic number to be embedded.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    number_of_interaction_modules : int
        Number of interaction modules.
    number_of_filters : int
        Number of filters, defines the dimensionality of the intermediate features.
    shared_interactions : bool
        Whether to share interaction parameters across all interaction modules.
    maximum_interaction_radius : openff.units.unit.Quantity
        The cutoff distance for interactions.
    activation_function : Type[torch.nn.Module]
        Activation function to use.
    """

    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_filters: int,
        shared_interactions: bool,
        activation_function: Type[torch.nn.Module],
        maximum_interaction_radius: unit.Quantity,
        predicted_properties: List[Dict[str, str]],
    ) -> None:

        log.debug("Initializing the SchNet architecture.")
        from modelforge.potential.utils import DenseWithCustomDist

        super().__init__(activation_function)
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
        # Intialize interaction blocks
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

        # Initialize output layers based on configuration
        self.output_layers = nn.ModuleDict()
        for property in predicted_properties:
            output_name = property["name"]
            output_type = property["type"]
            output_dimension = (
                1 if output_type == "scalar" else 3
            )  # vector means 3D output

            self.output_layers[output_name] = nn.Sequential(
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                ),
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    output_dimension,
                ),
            )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: PairListOutputs
    ) -> SchnetNeuralNetworkData:
        """
        Prepare the input data for the SchNet model.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist_output : PairListOutputs
            The pairlist output.

        Returns
        -------
        SchnetNeuralNetworkData
            The prepared input data for the SchNet model.
        """
        number_of_atoms = data.atomic_numbers.shape[0]

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
        self, data: Type[SchnetNeuralNetworkData]
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

        results = {
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

        # Compute all specified outputs
        for output_name, output_layer in self.output_layers.items():
            results[output_name] = output_layer(atomic_embedding).squeeze(-1)

        return results


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
from modelforge.utils.units import _convert_str_to_unit
from modelforge.utils.io import import_
from modelforge.potential.utils import shared_config_prior


class SchNet(BaseNetwork):
    """
    SchNet network for modeling quantum interactions.

    Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller: SchNet: A continuous-filter
    convolutional neural network for modeling quantum interactions.

    Parameters
    ----------
    featurization : Dict[str, Union[List[str], int]]
        Configuration for atom featurization.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    number_of_interaction_modules : int
        Number of interaction modules.
    maximum_interaction_radius : Union[unit.Quantity, str]
        The cutoff distance for interactions.
    number_of_filters : int
        Number of filters.
    shared_interactions : bool
        Whether to use shared interactions.
    activation_function_parameter : Dict
        Dict that contains keys: activation_function_name [str], activation_function_arguments [Dict],
        and activation_function [Type[torch.nn.Module]].
    postprocessing_parameter : Dict[str, Dict[str, bool]]
        Configuration for postprocessing parameters.
    dataset_statistic : Optional[Dict[str, float]], default=None
        Statistics of the dataset.
    potential_seed : Optional[int], optional
        Seed for the random number generator, default None.
    """

    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        maximum_interaction_radius: Union[unit.Quantity, str],
        number_of_filters: int,
        activation_function_parameter: Dict,
        shared_interactions: bool,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        predicted_properties: List[Dict[str, str]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        potential_seed: Optional[int] = None,
    ) -> None:

        self.only_unique_pairs = False  # NOTE: need to be set before super().__init__

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            potential_seed=potential_seed,
        )

        activation_function = activation_function_parameter["activation_function"]

        self.core_module = SchNetCore(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_filters=number_of_filters,
            shared_interactions=shared_interactions,
            activation_function=activation_function,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            predicted_properties=predicted_properties,
        )

    def _config_prior(self):
        """
        Configure the SchNet model hyperparameter prior distribution.

        Returns
        -------
        dict
            The prior distribution of hyperparameters.
        """
        log.info("Configuring SchNet model hyperparameter prior distribution")

        from ray import tune

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "maximum_interaction_radius": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "number_of_filters": tune.randint(32, 128),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior

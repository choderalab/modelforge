from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple
from .models import BaseNetwork, CoreNetwork

import torch
from loguru import logger as log
from torch import nn

from modelforge.utils.prop import SpeciesAEV

if TYPE_CHECKING:
    from modelforge.dataset.dataset import NNPInput
    from .models import PairListOutputs


def triu_index(num_species: int) -> torch.Tensor:
    """
    Generate a tensor representing the upper triangular indices for species pairs.

    Parameters
    ----------
    num_species : int
        The number of species in the system.

    Returns
    -------
    torch.Tensor
        A tensor containing the pair indices.
    """
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index

    return ret


from modelforge.potential.utils import NeuralNetworkData

ATOMIC_NUMBER_TO_INDEX_MAP = {
    1: 0,  # H
    6: 1,  # C
    7: 2,  # N
    8: 3,  # O
    9: 4,  # F
    16: 5,  # S
    17: 6,  # Cl
}


@dataclass
class AniNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs for ANI neural network potentials, designed to facilitate the efficient representation of atomic systems for energy computation and property prediction.

    Attributes
    ----------
    atomic_numbers : torch.Tensor
        A 1D tensor containing the atomic numbers for atoms, used for identifying the atom types within the model. Shape: [num_atoms].

    """

    atom_index: torch.Tensor


from openff.units import unit


class ANIRepresentation(nn.Module):
    """
    Compute the Atomic Environment Vectors (AEVs) for the ANI architecture.

    Parameters
    ----------
    radial_max_distance : unit.Quantity
        The maximum distance for radial symmetry functions.
    radial_min_distance : unit.Quantity
        The minimum distance for radial symmetry functions.
    number_of_radial_basis_functions : int
        The number of radial basis functions.
    maximum_interaction_radius_for_angular_features : unit.Quantity
        The maximum interaction radius for angular features.
    minimum_interaction_radius_for_angular_features : unit.Quantity
        The minimum interaction radius for angular features.
    angular_dist_divisions : int
        The number of angular distance divisions.
    angle_sections : int
        The number of angle sections.
    nr_of_supported_elements : int, optional
        The number of supported elements, by default 7.
    """

    def __init__(
        self,
        radial_max_distance: unit.Quantity,
        radial_min_distanc: unit.Quantity,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius_for_angular_features: unit.Quantity,
        minimum_interaction_radius_for_angular_features: unit.Quantity,
        angular_dist_divisions: int,
        angle_sections: int,
        nr_of_supported_elements: int = 7,
    ):

        super().__init__()
        from modelforge.potential.utils import CosineAttenuationFunction

        self.maximum_interaction_radius_for_angular_features = (
            maximum_interaction_radius_for_angular_features
        )
        self.nr_of_supported_elements = nr_of_supported_elements

        self.cutoff_module = CosineAttenuationFunction(radial_max_distance)

        self.radial_symmetry_functions = self._setup_radial_symmetry_functions(
            radial_max_distance, radial_min_distanc, number_of_radial_basis_functions
        )
        self.angular_symmetry_functions = self._setup_angular_symmetry_functions(
            maximum_interaction_radius_for_angular_features,
            minimum_interaction_radius_for_angular_features,
            angular_dist_divisions,
            angle_sections,
        )
        # generate indices
        from modelforge.potential.utils import triple_by_molecule

        self.triple_by_molecule = triple_by_molecule
        self.register_buffer("triu_index", triu_index(self.nr_of_supported_elements))

    def _setup_radial_symmetry_functions(
        self,
        max_distance: unit.Quantity,
        min_distance: unit.Quantity,
        number_of_radial_basis_functions: int,
    ):
        from .utils import AniRadialBasisFunction

        radial_symmetry_function = AniRadialBasisFunction(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def _setup_angular_symmetry_functions(
        self,
        max_distance: unit.Quantity,
        min_distance: unit.Quantity,
        angular_dist_divisions,
        angle_sections,
    ):
        from .utils import AngularSymmetryFunction

        # set up modelforge angular features
        return AngularSymmetryFunction(
            max_distance,
            min_distance,
            angular_dist_divisions,
            angle_sections,
            dtype=torch.float32,
        )

    def forward(self, data: AniNeuralNetworkData) -> SpeciesAEV:
        """
        Forward pass to compute Atomic Environment Vectors (AEVs).

        Parameters
        ----------
        data : AniNeuralNetworkData
            The input data for the ANI model.

        Returns
        -------
        SpeciesAEV
            The computed atomic environment vectors (AEVs) for each species.
        """

        # ----------------- Radial symmetry vector ---------------- #
        # compute radial aev

        radial_feature_vector = self.radial_symmetry_functions(data.d_ij)
        # Apply cutoff to radial features
        rcut_ij = self.cutoff_module(data.d_ij)
        radial_feature_vector = radial_feature_vector * rcut_ij

        # Process output to prepare for angular symmetry vector
        postprocessed_radial_aev_and_additional_data = self._postprocess_radial_aev(
            radial_feature_vector, data=data
        )
        processed_radial_feature_vector = postprocessed_radial_aev_and_additional_data[
            "radial_aev"
        ]

        # ----------------- Angular symmetry vector ---------------- #
        # Compute angular AEV
        angular_data = self._preprocess_angular_aev(
            postprocessed_radial_aev_and_additional_data
        )
        # calculate angular aev
        angular_feature_vector = self.angular_symmetry_functions(
            angular_data["angular_r_ij"]
        )
        # postprocess
        angular_data["angular_feature_vector"] = angular_feature_vector
        processed_angular_feature_vector = self._postprocess_angular_aev(
            data, angular_data
        )
        # Concatenate radial and angular features
        aevs = torch.cat(
            [processed_radial_feature_vector, processed_angular_feature_vector], dim=-1
        )

        return SpeciesAEV(data.atom_index, aevs)

    def _postprocess_angular_aev(
        self, data: AniNeuralNetworkData, angular_data: Dict[str, torch.Tensor]
    ):
        """
        Postprocess the angular AEVs.

        Parameters
        ----------
        data : AniNeuralNetworkData
            The input data for the ANI model.
        angular_data : Dict[str, torch.Tensor]
            The angular data including species and displacement vectors.

        Returns
        -------
        torch.Tensor
            The processed angular AEVs.
        """
        angular_sublength = self.angular_symmetry_functions.angular_sublength
        angular_length = (
            (self.nr_of_supported_elements * (self.nr_of_supported_elements + 1))
            // 2
            * angular_sublength
        )

        num_species_pairs = angular_length // angular_sublength

        number_of_atoms = data.number_of_atoms
        # compute angular aev
        central_atom_index = angular_data["central_atom_index"]
        angular_species12 = angular_data["angular_species12"]
        angular_r_ij = angular_data["angular_r_ij"]

        angular_terms_ = angular_data["angular_feature_vector"]
        # Initialize tensor to store angular AEVs

        angular_aev = angular_terms_.new_zeros(
            (number_of_atoms * num_species_pairs, angular_sublength)
        )

        index = (
            central_atom_index * num_species_pairs
            + self.triu_index[angular_species12[0], angular_species12[1]]
        )
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(number_of_atoms, angular_length)
        return angular_aev

    def _postprocess_radial_aev(
        self,
        radial_feature_vector: torch.Tensor,
        data: AniNeuralNetworkData,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess the radial AEVs.

        Parameters
        ----------
        radial_feature_vector : torch.Tensor
            The radial feature vectors.
        data : AniNeuralNetworkData
            The input data for the ANI model.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the radial AEVs and additional data.
        """
        radial_feature_vector = radial_feature_vector.squeeze(1)
        number_of_atoms = data.number_of_atoms
        radial_sublength = (
            self.radial_symmetry_functions.number_of_radial_basis_functions
        )
        radial_length = radial_sublength * self.nr_of_supported_elements

        # Initialize tensor to store radial AEVs
        radial_aev = radial_feature_vector.new_zeros(
            (
                number_of_atoms * self.nr_of_supported_elements,
                radial_sublength,
            )
        )
        atom_index12 = data.pair_indices
        species = data.atom_index
        species12 = species[atom_index12]

        index12 = atom_index12 * self.nr_of_supported_elements + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_feature_vector)
        radial_aev.index_add_(0, index12[1], radial_feature_vector)

        radial_aev = radial_aev.reshape(number_of_atoms, radial_length)

        # compute new neighbors with radial_cutoff
        distances = data.d_ij.T.flatten()
        even_closer_indices = (
            (
                distances
                <= self.maximum_interaction_radius_for_angular_features.to(
                    unit.nanometer
                ).m
            )
            .nonzero()
            .flatten()
        )
        r_ij = data.r_ij
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        r_ij_small = r_ij.index_select(0, even_closer_indices)

        return {
            "radial_aev": radial_aev,
            "atom_index12": atom_index12,
            "species12": species12,
            "r_ij": r_ij_small,
        }

    def _preprocess_angular_aev(self, data: Dict[str, torch.Tensor]):
        """
        Preprocess the angular AEVs.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The data dictionary containing radial AEVs and additional data.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the preprocessed angular AEV data.
        """
        atom_index12 = data["atom_index12"]
        species12 = data["species12"]
        r_ij = data["r_ij"]

        # compute angular aev
        central_atom_index, pair_index12, sign12 = self.triple_by_molecule(atom_index12)
        species12_small = species12[:, pair_index12]

        r_ij12 = r_ij.index_select(0, pair_index12.view(-1)).view(
            2, -1, 3
        ) * sign12.unsqueeze(-1)
        species12_ = torch.where(
            torch.eq(sign12, 1), species12_small[1], species12_small[0]
        )
        return {
            "angular_r_ij": r_ij12,
            "central_atom_index": central_atom_index,
            "angular_species12": species12_,
        }


class ANIInteraction(nn.Module):
    """
    Atomic neural network interaction module for ANI.

    Parameters
    ----------
    aev_dim : int
        The dimensionality of the AEVs.
    activation_function_class : torch.nn.Module
        The activation function class to use.
    """

    def __init__(self, *, aev_dim: int, activation_function_class: torch.nn.Module):

        super().__init__()
        # define atomic neural network
        atomic_neural_networks = self.intialize_atomic_neural_network(
            aev_dim, activation_function_class
        )
        self.atomic_networks = nn.ModuleList(
            [
                atomic_neural_networks[element]
                for element in ["H", "C", "O", "N", "S", "F", "Cl"]
            ]
        )

    def intialize_atomic_neural_network(
        self, aev_dim: int, activation_function_class: torch.nn.Module
    ) -> Dict[str, nn.Module]:
        """
        Initialize the atomic neural networks for each element.

        Parameters
        ----------
        aev_dim : int
            The dimensionality of the AEVs.
        activation_function_class : torch.nn.Module
            The activation function class to use.

        Returns
        -------
        Dict[str, nn.Module]
            A dictionary mapping element symbols to their corresponding neural networks.
        """

        def create_network(layers):
            """
            Create a neural network with the specified layers.

            Parameters
            ----------
            layers : List[int]
                A list of integers specifying the number of units in each layer.

            Returns
            -------
            nn.Sequential
                The created neural network.
            """
            network_layers = []
            input_dim = aev_dim
            for units in layers:
                network_layers.append(nn.Linear(input_dim, units))
                network_layers.append(activation_function_class(0.1))
                input_dim = units
            network_layers.append(nn.Linear(input_dim, 1))
            return nn.Sequential(*network_layers)

        return {
            element: create_network(layers)
            for element, layers in {
                "H": [256, 192, 160],
                "C": [224, 192, 160],
                "N": [192, 160, 128],
                "O": [192, 160, 128],
                "S": [160, 128, 96],
                "F": [160, 128, 96],
                "Cl": [160, 128, 96],
            }.items()
        }

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass to compute atomic energies from AEVs.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the species tensor and the AEV tensor.

        Returns
        -------
        torch.Tensor
            The computed atomic energies.
        """
        species, aev = input
        output = aev.new_zeros(species.shape)

        for i, model in enumerate(self.atomic_networks):
            mask = torch.eq(species, i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output[midx] = model(input_).flatten()

        return output.view_as(species)


class ANI2xCore(CoreNetwork):
    """
    Core network class for the ANI2x neural network potential.

    Parameters
    ----------
    maximum_interaction_radius : unit.Quantity
        The maximum radial distance for the radial basis functions.
    minimum_interaction_radius : unit.Quantity
        The minimum radial distance for the radial basis functions.
    number_of_radial_basis_functions : int
        The number of radial basis functions to use.
    maximum_interaction_radius_for_angular_features : unit.Quantity
        The maximum angular distance for the angular basis functions.
    minimum_interaction_radius_for_angular_features : unit.Quantity
        The minimum angular distance for the angular basis functions.
    activation_function : str
        The name of the activation function to use.
    angular_dist_divisions : int
        The number of divisions for the angular distance.
    angle_sections : int
        The number of angle sections to use.
    """

    def __init__(
        self,
        *,
        maximum_interaction_radius: unit.Quantity,
        minimum_interaction_radius: unit.Quantity,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius_for_angular_features: unit.Quantity,
        minimum_interaction_radius_for_angular_features: unit.Quantity,
        activation_function: str,
        angular_dist_divisions: int,
        angle_sections: int,
    ) -> None:
        # number of elements in ANI2x
        self.num_species = 7

        log.debug("Initializing the ANI2x architecture.")
        super().__init__(activation_function)

        # Initialize representation block
        self.ani_representation_module = ANIRepresentation(
            maximum_interaction_radius,
            minimum_interaction_radius,
            number_of_radial_basis_functions,
            maximum_interaction_radius_for_angular_features,
            minimum_interaction_radius_for_angular_features,
            angular_dist_divisions,
            angle_sections,
        )
        # The length of radial aev
        self.radial_length = self.num_species * number_of_radial_basis_functions
        # The length of angular aev
        self.angular_length = (
            (self.num_species * (self.num_species + 1))
            // 2
            * self.ani_representation_module.angular_symmetry_functions.angular_sublength
        )

        # The length of full aev
        self.aev_length = self.radial_length + self.angular_length

        # Initialize interaction blocks
        self.interaction_modules = ANIInteraction(
            aev_dim=self.aev_length,
            activation_function_class=self.activation_function_class,
        )

        # ----- ATOMIC NUMBER LOOKUP --------
        # Create a tensor for direct lookup. The size of this tensor will be
        # # the max atomic number in map. Initialize with a default value (e.g., -1 for not found).

        maximum_atomic_number = max(ATOMIC_NUMBER_TO_INDEX_MAP.keys())
        lookup_tensor = torch.full((maximum_atomic_number + 1,), -1, dtype=torch.long)

        # Populate the lookup tensor with indices from your map
        for atomic_number, index in ATOMIC_NUMBER_TO_INDEX_MAP.items():
            lookup_tensor[atomic_number] = index

        self.register_buffer("lookup_tensor", lookup_tensor)

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> AniNeuralNetworkData:
        """
        Prepare the model-specific input data for the ANI2x model.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist_output : PairListOutputs
            The pairlist output.

        Returns
        -------
        AniNeuralNetworkData
            The prepared input data for the ANI2x model.
        """
        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_data = AniNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atom_index=self.lookup_tensor[data.atomic_numbers.long()],
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
        )

        return nnp_data

    def compute_properties(self, data: AniNeuralNetworkData) -> Dict[str, torch.Tensor]:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        data : AniNeuralNetworkData
            The input data for the ANI model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated energies.
        """

        # compute the representation (atomic environment vectors) for each atom
        representation = self.ani_representation_module(data)
        # compute the atomic energies
        E_i = self.interaction_modules(representation)

        return {
            "per_atom_energy": E_i,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


from typing import Union, Optional, Dict


class ANI2x(BaseNetwork):
    """
    ANI2x network class for the ANI2x neural network potential.

    Parameters
    ----------
    maximum_interaction_radius : Union[unit.Quantity, str]
        The maximum radial distance for the radial basis functions.
    minimum_interaction_radius : Union[unit.Quantity, str]
        The minimum radial distance for the radial basis functions.
    number_of_radial_basis_functions : int
        The number of radial basis functions to use.
    maximum_interaction_radius_for_angular_features : Union[unit.Quantity, str]
        The maximum angular distance for the angular basis functions.
    minimum_interaction_radius_for_angular_features : Union[unit.Quantity, str]
        The minimum angular distance for the angular basis functions.
    angular_dist_divisions : int
        The number of divisions for the angular distance.
    angle_sections : int
        The number of angle sections to use.
    activation_function : str
        The name of the activation function to use.
    postprocessing_parameter : Dict[str, Dict[str, bool]]
        Configuration for postprocessing parameters.
    dataset_statistic : Optional[Dict[str, float]], optional
        Statistics of the dataset, by default None.
    """

    def __init__(
        self,
        maximum_interaction_radius: Union[unit.Quantity, str],
        minimum_interaction_radius: Union[unit.Quantity, str],
        number_of_radial_basis_functions: int,
        maximum_interaction_radius_for_angular_features: Union[unit.Quantity, str],
        minimum_interaction_radius_for_angular_features: Union[unit.Quantity, str],
        angular_dist_divisions: int,
        angle_sections: int,
        activation_function: str,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:

        from modelforge.utils.units import _convert_str_to_unit

        self.only_unique_pairs = True  # NOTE: need to be set before super().__init__

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
        )

        self.core_module = ANI2xCore(
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            minimum_interaction_radius=_convert_str_to_unit(minimum_interaction_radius),
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            maximum_interaction_radius_for_angular_features=_convert_str_to_unit(
                maximum_interaction_radius_for_angular_features
            ),
            minimum_interaction_radius_for_angular_features=_convert_str_to_unit(
                minimum_interaction_radius_for_angular_features
            ),
            activation_function=activation_function,
            angular_dist_divisions=angular_dist_divisions,
            angle_sections=angle_sections,
        )

    def _config_prior(self):
        log.info("Configuring ANI2x model hyperparameter prior distribution")
        from modelforge.utils.io import import_

        tune = import_("ray").tune
        # from ray import tune

        from modelforge.train.utils import shared_config_prior

        prior = {
            "radial_max_distance": tune.uniform(5, 10),
            "radial_min_distance": tune.uniform(0.6, 1.4),
            "number_of_radial_basis_functions": tune.randint(12, 20),
            "maximum_interaction_radius_for_angular_features": tune.uniform(2.5, 4.5),
            "minimum_interaction_radius_for_angular_features": tune.uniform(0.6, 1.4),
            "angle_sections": tune.randint(3, 8),
        }
        prior.update(shared_config_prior())
        return prior

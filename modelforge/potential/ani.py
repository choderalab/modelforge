from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple
from .models import InputPreparation, BaseNetwork, CoreNetwork

import torch
from loguru import logger as log
from torch import nn

from modelforge.utils.prop import SpeciesAEV

if TYPE_CHECKING:
    from modelforge.dataset.dataset import NNPInput
    from .models import PairListOutputs


def triu_index(num_species: int) -> torch.Tensor:
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
    A dataclass to structure the inputs for ANI neural network potentials, designed to
    facilitate the efficient representation of atomic systems for energy computation and
    property prediction.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A 2D tensor indicating the indices of atom pairs. Shape: [2, num_pairs].
    d_ij : torch.Tensor
        A 1D tensor containing distances between each pair of atoms. Shape: [num_pairs, 1].
    r_ij : torch.Tensor
        A 2D tensor representing displacement vectors between atom pairs. Shape: [num_pairs, 3].
    number_of_atoms : int
        An integer indicating the number of atoms in the batch.
    positions : torch.Tensor
        A 2D tensor representing the XYZ coordinates of each atom. Shape: [num_atoms, 3].
    atom_index : torch.Tensor
        A 1D tensor containing atomic numbers for each atom in the system(s). Shape: [num_atoms].
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule. Shape: [num_atoms].
    total_charge : torch.Tensor
        An tensor with the total charge of each system or molecule. Shape: [num_systems].
    atomic_numbers : torch.Tensor
        A 1D tensor containing the atomic numbers for atoms, used for identifying the atom types within the model. Shape: [num_atoms].

    Notes
    -----
    The `AniNeuralNetworkInput` dataclass encapsulates essential inputs required by the
    ANI neural network model to predict system energies and properties accurately. It
    includes atomic positions, types, and connectivity information, crucial for representing
    atomistic systems in detail.

    Examples
    --------
    >>> ani_input = AniNeuralNetworkData(
    ...     pair_indices=torch.tensor([[0, 1], [0, 2], [1, 2]]).T,  # Transpose for correct shape
    ...     d_ij=torch.tensor([[1.0], [1.0], [1.0]]),  # Distances between pairs
    ...     r_ij=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # Displacement vectors
    ...     number_of_atoms=4,  # Total number of atoms
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ...     atom_index=torch.tensor([1, 6, 6, 8]),  # Atomic numbers for H, C, C, O
    ...     atomic_subsystem_indices=torch.tensor([0, 0, 0, 0]),  # All atoms belong to the same molecule
    ...     total_charge=torch.tensor([0.0]),  # Assuming the molecule is neutral
    ...     atomic_numbers=torch.tensor([1, 6, 6, 8])  # Repeated for completeness
    ... )
    """

    atom_index: torch.Tensor


from openff.units import unit


class ANIRepresentation(nn.Module):
    # calculate the atomic environment vectors
    # used for the ANI architecture of NNPs

    def __init__(
        self,
        radial_max_distance: unit.Quantity,
        radial_min_distanc: unit.Quantity,
        number_of_radial_basis_functions: int,
        angular_max_distance: unit.Quantity,
        angular_min_distance: unit.Quantity,
        angular_dist_divisions: int,
        angle_sections: int,
        nr_of_supported_elements: int = 7,
    ):
        # radial symmetry functions

        super().__init__()
        from modelforge.potential.utils import CosineCutoff

        self.angular_max_distance = angular_max_distance
        self.nr_of_supported_elements = nr_of_supported_elements

        self.cutoff_module = CosineCutoff(radial_max_distance)

        self.radial_symmetry_functions = self._setup_radial_symmetry_functions(
            radial_max_distance, radial_min_distanc, number_of_radial_basis_functions
        )
        self.angular_symmetry_functions = self._setup_angular_symmetry_functions(
            angular_max_distance,
            angular_min_distance,
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
        # calculate the atomic environment vectors
        # used for the ANI architecture of NNPs

        # ----------------- Radial symmetry vector ---------------- #
        # compute radial aev

        radial_feature_vector = self.radial_symmetry_functions(data.d_ij)
        # cutoff
        rcut_ij = self.cutoff_module(data.d_ij)
        radial_feature_vector = radial_feature_vector * rcut_ij

        # process output to prepare for agular symmetry vector
        postprocessed_radial_aev_and_additional_data = self._postprocess_radial_aev(
            radial_feature_vector, data=data
        )
        processed_radial_feature_vector = postprocessed_radial_aev_and_additional_data[
            "radial_aev"
        ]

        # ----------------- Angular symmetry vector ---------------- #
        # preprocess
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
        aevs = torch.cat(
            [processed_radial_feature_vector, processed_angular_feature_vector], dim=-1
        )

        return SpeciesAEV(data.atom_index, aevs)

    def _postprocess_angular_aev(
        self, data: AniNeuralNetworkData, angular_data: Dict[str, torch.Tensor]
    ):
        # postprocess the angular aev
        # used for the ANI architecture of NNPs
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
    ) -> Dict[str, torch.tensor]:
        radial_feature_vector = radial_feature_vector.squeeze(1)
        number_of_atoms = data.number_of_atoms
        radial_sublength = self.radial_symmetry_functions.number_of_radial_basis_functions
        radial_length = radial_sublength * self.nr_of_supported_elements

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
            (distances <= self.angular_max_distance.to(unit.nanometer).m)
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
    def __init__(self, aev_dim: int):
        super().__init__()
        # define atomic neural network
        atomic_neural_networks = self.intialize_atomic_neural_network(aev_dim)
        H_network = atomic_neural_networks["H"]
        C_network = atomic_neural_networks["C"]
        O_network = atomic_neural_networks["O"]
        N_network = atomic_neural_networks["N"]
        S_network = atomic_neural_networks["S"]
        F_network = atomic_neural_networks["F"]
        Cl_network = atomic_neural_networks["Cl"]
        self.atomic_networks = nn.ModuleList(
            [
                H_network,
                C_network,
                O_network,
                N_network,
                S_network,
                F_network,
                Cl_network,
            ]
        )

    def intialize_atomic_neural_network(self, aev_dim: int) -> Dict[str, nn.Module]:
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.CELU(0.1),
            torch.nn.Linear(256, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 1),
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.CELU(0.1),
            torch.nn.Linear(224, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 1),
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 1),
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.CELU(0.1),
            torch.nn.Linear(192, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 1),
        )

        S_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        F_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        Cl_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        return {
            "H": H_network,
            "C": C_network,
            "N": N_network,
            "O": O_network,
            "S": S_network,
            "F": F_network,
            "Cl": Cl_network,
        }

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
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
    def __init__(
        self,
        radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
        radial_min_distanc: unit.Quantity = 0.8 * unit.angstrom,
        number_of_radial_basis_functions: int = 16,
        angular_max_distance: unit.Quantity = 3.5 * unit.angstrom,
        angular_min_distance: unit.Quantity = 0.8 * unit.angstrom,
        angular_dist_divisions: int = 8,
        angle_sections: int = 4,
    ) -> None:
        """
        Initialize the ANI NNP architecture.

        Parameters
        ----------
        """
        # number of elements in ANI2x
        self.num_species = 7

        log.debug("Initializing ANI model.")
        super().__init__()

        # Initialize representation block
        self.ani_representation_module = ANIRepresentation(
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
            angular_max_distance,
            angular_min_distance,
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

        # Intialize interaction blocks
        self.interaction_modules = ANIInteraction(self.aev_length)

        # ----- ATOMIC NUMBER LOOKUP --------
        # Create a tensor for direct lookup. The size of this tensor will be
        # # the max atomic number in map. Initialize with a default value (e.g., -1 for not found).

        max_atomic_number = max(ATOMIC_NUMBER_TO_INDEX_MAP.keys())
        lookup_tensor = torch.full((max_atomic_number + 1,), -1, dtype=torch.long)

        # Populate the lookup tensor with indices from your map
        for atomic_number, index in ATOMIC_NUMBER_TO_INDEX_MAP.items():
            lookup_tensor[atomic_number] = index

        self.register_buffer("lookup_tensor", lookup_tensor)

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> AniNeuralNetworkData:
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
        data : AniNeuralNetworkInput
        - pairlist:  shape (n_pairs, 2)
        - r_ij:  shape (n_pairs, 1)
        - d_ij:  shape (n_pairs, 3)
        - positions:  shape (nr_of_atoms_per_molecules, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # compute the representation (atomic environment vectors) for each atom
        representation = self.ani_representation_module(data)
        # compute the atomic energies
        E_i = self.interaction_modules(representation)

        return {
            "E_i": E_i,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


from typing import Union, Optional, List, Dict


class ANI2x(BaseNetwork):
    def __init__(
        self,
        radial_max_distance: Union[unit.Quantity, str],
        radial_min_distance: Union[unit.Quantity, str],
        number_of_radial_basis_functions: int,
        angular_max_distance: Union[unit.Quantity, str],
        angular_min_distance: Union[unit.Quantity, str],
        angular_dist_divisions: int,
        angle_sections: int,
        processing_operation: List[Dict[str, str]],
        readout_operation: List[Dict[str, str]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(
            processing_operation=processing_operation,
            dataset_statistic=dataset_statistic,
            readout_operation=readout_operation,
        )

        from modelforge.utils.units import _convert

        self.core_module = ANI2xCore(
            _convert(radial_max_distance),
            _convert(radial_min_distance),
            number_of_radial_basis_functions,
            _convert(angular_max_distance),
            _convert(angular_min_distance),
            angular_dist_divisions,
            angle_sections,
        )
        self.only_unique_pairs = True  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=_convert(radial_max_distance),
            only_unique_pairs=self.only_unique_pairs,
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
            "angular_max_distance": tune.uniform(2.5, 4.5),
            "angular_min_distance": tune.uniform(0.6, 1.4),
            "angle_sections": tune.randint(3, 8),
        }
        prior.update(shared_config_prior())
        return prior

    def combine_per_atom_properties(
        self, values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return values

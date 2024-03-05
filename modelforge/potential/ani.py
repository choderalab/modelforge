import torch
from torch import nn
from loguru import logger as log
from modelforge.potential.models import BaseNNP
from modelforge.potential.postprocessing import PostprocessingPipeline, NoPostprocess
from typing import Dict, NamedTuple, Tuple
from openff.units import unit


class SpeciesEnergies(NamedTuple):
    species: torch.Tensor
    energies: torch.Tensor


class SpeciesAEV(NamedTuple):
    species: torch.Tensor
    aevs: torch.Tensor


def triu_index(num_species: int) -> torch.Tensor:
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index
    return ret


class ANIRepresentation(nn.Module):
    # calculate the atomic environment vectors
    # used for the ANI architecture of NNPs

    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        angular_cutoff: unit.Quantity,
        nr_of_supported_elements: int = 7,
        device: torch.device = torch.device("cpu"),
    ):
        # radial symmetry functions

        super().__init__()
        self.radial_cutoff = radial_cutoff
        self.angular_cutoff = angular_cutoff
        self.nr_of_supported_elements = nr_of_supported_elements

        self.radial_symmetry_functions = self._setup_radial_symmetry_functions(
            self.radial_cutoff
        )
        self.angular_symmetry_functions = self._setup_angular_symmetry_functions(
            self.angular_cutoff
        )
        # generate indices
        from modelforge.potential.utils import triple_by_molecule

        self.triple_by_molecule = triple_by_molecule
        self.register_buffer(
            "triu_index",
            triu_index(self.nr_of_supported_elements).to(device=device),
        )

    def _setup_radial_symmetry_functions(self, radial_cutoff: unit.Quantity):
        from openff.units import unit
        from .utils import RadialSymmetryFunction

        # ANI constants
        radial_start = 0.8 * unit.angstrom
        radial_dist_divisions = 16

        radial_symmetry_function = RadialSymmetryFunction(
            radial_dist_divisions,
            radial_cutoff,
            radial_start,
            ani_style=True,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def _setup_angular_symmetry_functions(self, angular_cutoff: unit.Quantity):
        from .utils import AngularSymmetryFunction
        from openff.units import unit

        # ANI constants for angular features
        angular_start = 0.8 * unit.angstrom
        angular_dist_divisions = 8
        angle_sections = 4

        # set up modelforge angular features
        return AngularSymmetryFunction(
            angular_cutoff,
            angular_start,
            angular_dist_divisions,
            angle_sections,
            dtype=torch.float32,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):

        # calculate the atomic environment vectors
        # used for the ANI architecture of NNPs

        # ----------------- Radial symmetry vector ---------------- #
        # compute radial aev
        radial_feature_vector = self.radial_symmetry_functions(inputs["d_ij"])
        # process output to prepare for agular symmetry vector
        postprocessed_radial_aev_and_additional_data = self._postprocess_radial_aev(
            radial_feature_vector, inputs=inputs
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
            inputs, angular_data
        )
        aevs = torch.cat(
            [processed_radial_feature_vector, processed_angular_feature_vector], dim=-1
        )

        return SpeciesAEV(inputs["atomic_numbers"], aevs)

    def _postprocess_angular_aev(
        self, inputs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
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

        number_of_atoms_in_batch = inputs["number_of_atoms_in_batch"]
        # compute angular aev
        central_atom_index = data["central_atom_index"]
        angular_species12 = data["angular_species12"]
        angular_r_ij = data["angular_r_ij"]

        angular_terms_ = data["angular_feature_vector"]

        angular_aev = angular_terms_.new_zeros(
            (number_of_atoms_in_batch * num_species_pairs, angular_sublength)
        )
        index = (
            central_atom_index * num_species_pairs
            + self.triu_index[angular_species12[0], angular_species12[1]]
        )
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(number_of_atoms_in_batch, angular_length)
        return angular_aev

    def _postprocess_radial_aev(
        self,
        radial_feature_vector,
        inputs: Dict[str, torch.Tensor],
    ):

        radial_feature_vector = radial_feature_vector.squeeze(1)
        number_of_atoms_in_batch = inputs["number_of_atoms_in_batch"]
        radial_sublength = self.radial_symmetry_functions.radial_sublength
        radial_length = radial_sublength * self.nr_of_supported_elements
        radial_aev = radial_feature_vector.new_zeros(
            (
                number_of_atoms_in_batch * self.nr_of_supported_elements,
                radial_sublength,
            )
        )
        atom_index12 = inputs["pair_indices"]
        species = inputs["atomic_numbers"]
        species12 = species[atom_index12]

        index12 = atom_index12 * self.nr_of_supported_elements + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_feature_vector)
        radial_aev.index_add_(0, index12[1], radial_feature_vector)

        radial_aev = radial_aev.reshape(number_of_atoms_in_batch, radial_length)

        # compute new neighbors with radial_cutoff
        distances = inputs["d_ij"].T.flatten()
        even_closer_indices = (
            (distances <= self.angular_cutoff.to(unit.nanometer).m).nonzero().flatten()
        )
        r_ij = inputs["r_ij"]
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
        species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])
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
        self.H_network = atomic_neural_networks["H"]
        self.C_network = atomic_neural_networks["C"]
        self.O_network = atomic_neural_networks["O"]
        self.N_network = atomic_neural_networks["N"]
        # self.S_network = atomic_neural_networks["S"]
        # self.F_network = atomic_neural_networks["F"]
        # self.Cl_network = atomic_neural_networks["Cl"]
        self.atomic_networks = [
            self.H_network,
            self.C_network,
            self.O_network,
            self.N_network,
        ]

    def intialize_atomic_neural_network(self, aev_dim: int) -> Dict[str, nn.Module]:

        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.CELU(0.1),
            torch.nn.Linear(160, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 144),
            torch.nn.CELU(0.1),
            torch.nn.Linear(144, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.CELU(0.1),
            torch.nn.Linear(128, 112),
            torch.nn.CELU(0.1),
            torch.nn.Linear(112, 96),
            torch.nn.CELU(0.1),
            torch.nn.Linear(96, 1),
        )

        return {"H": H_network, "C": C_network, "N": N_network, "O": O_network}

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):

        species, aev = input
        output = aev.new_zeros(species.shape)

        for i, model in enumerate(self.atomic_networks):
            mask = species == i
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, model(input_).flatten())

        return output.view_as(species)


class ANI2x(BaseNNP):

    def __init__(
        self,
        postprocessing: PostprocessingPipeline = PostprocessingPipeline(
            [NoPostprocess({})]
        ),
        radial_cutoff: unit.Quantity = 5.3 * unit.angstrom,
        angular_cutoff: unit.Quantity = 3.5 * unit.angstrom,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize the ANi NNP architeture.

        Parameters
        ----------
        """
        # number of elements in ANI2x
        self.num_species = 7
        self.only_unique_pairs = True

        log.debug("Initializing ANI model.")
        super().__init__(
            radial_cutoff=radial_cutoff,
            angular_cutoff=angular_cutoff,
            postprocessing=postprocessing,
        )

        # Initialize representation block
        self.ani_representation_module = ANIRepresentation(
            radial_cutoff, angular_cutoff, device=device
        )
        # The length of radial aev
        self.radial_length = (
            self.num_species
            * self.ani_representation_module.radial_symmetry_functions.radial_sublength
        )
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

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

    def _model_specific_input_preparation(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        return inputs

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
        - pairlist:  shape (n_pairs, 2)
        - r_ij:  shape (n_pairs, 1)
        - d_ij:  shape (n_pairs, 3)
        - positions:  shape (nr_of_atoms_per_molecules, 3)
        - atomic_embedding:  shape (nr_of_atoms_in_systems, nr_atom_basis)


        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # compute the representation (atomic environment vectors) for each atom
        representation = self.ani_representation_module(inputs)
        # compute the atomic energies
        per_species_energies = self.interaction_modules(representation)

        return {
            "scalar_representation": per_species_energies,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:

        per_species_energies = inputs["scalar_representation"]
        atomic_subsystem_indices = inputs["atomic_subsystem_indices"]
        # output tensor for the sums, size based on the number of unique values in atomic_subsystem_indices
        energy_per_molecule = torch.zeros(
            atomic_subsystem_indices.max() + 1, dtype=per_species_energies.dtype
        )

        # use index_add_ to sum values in per_species_energies according to indices in atomic_subsystem_indices
        energy_per_molecule.index_add_(
            0, atomic_subsystem_indices, per_species_energies
        )
        return energy_per_molecule

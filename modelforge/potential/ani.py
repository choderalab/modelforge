import torch
from torch import nn
from loguru import logger as log
from modelforge.potential.models import BaseNNP
from modelforge.potential.postprocessing import PostprocessingPipeline, NoPostprocess
from typing import Dict
from openff.units import unit


class ANIRepresentation(nn.Module):
    # calculate the atomic environment vectors
    # used for the ANI architecture of NNPs

    def __init__(self, radial_cutoff: unit.Quantity, angular_cutoff: unit.Quantity):
        # radial symmetry functions

        super().__init__()
        self.radial_cutoff = radial_cutoff
        self.angular_cutoff = angular_cutoff

        self.radial_symmetry_functions = self._setup_radial_symmetry_functions(
            self.radial_cutoff
        )
        self.angular_symmetry_functions = self._setup_angular_symmetry_functions(
            self.angular_cutoff
        )

    def _setup_radial_symmetry_functions(self, radial_cutoff: unit.Quantity):
        from openff.units import unit
        from .utils import RadialSymmetryFunction

        # ANI constants
        radial_start = 0.8 * unit.angstrom
        radial_dist_divisions = 8

        radial_symmetry_function = RadialSymmetryFunction(
            radial_dist_divisions,
            radial_cutoff,
            radial_start,
            ani_style=True,
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
        )


class ANIInteraction(nn.Module):

    def __init__(self):
        pass


class ANI2x(BaseNNP):

    def __init__(
        self,
        postprocessing: PostprocessingPipeline = PostprocessingPipeline(
            [NoPostprocess({})]
        ),
        radial_cutoff: unit.Quantity = 5.3 * unit.angstrom,
        angular_cutoff: unit.Quantity = 3.5 * unit.angstrom,
    ) -> None:
        """
        Initialize the ANi NNP architeture.

        Parameters
        ----------
        nr_filters : int, optional
            Number of filters; defines the dimensionality of the intermediate features (default is 2).
        """

        log.debug("Initializing ANI model.")
        super().__init__(
            radial_cutoff=radial_cutoff,
            angular_cutoff=angular_cutoff,
            postprocessing=postprocessing,
        )

        # Initialize representation block
        self.ani_representation_module = ANIRepresentation(
            radial_cutoff, angular_cutoff
        )
        # Intialize interaction blocks
        self.interaction_modules = ANIInteraction()

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        inputs = self._prepare_inputs(inputs)
        inputs = self._model_specific_input_preparation(inputs)
        return inputs

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        # reformat for input
        species = species.flatten()
        atom_index12 = inputs["pair_indices"]
        species12 = species[atom_index12]

        # get index in right order
        even_closer_indices = (d_ij <= Rca).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        r_ij = r_ij.index_select(0, even_closer_indices)
        central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
        species12_small = species12[:, pair_index12]
        vec12 = r_ij.index_select(0, pair_index12.view(-1)).view(
            2, -1, 3
        ) * sign12.unsqueeze(-1)
        species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])

        return inputs

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        atomic_embedding : torch.Tensor
            Atomic numbers embedding; shape (nr_of_atoms_in_systems, 1, nr_atom_basis).
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

        # Compute the representation for each atom
        representation = self.schnet_representation_module(inputs["d_ij"])
        x = inputs["atomic_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                x,
                inputs["pair_indices"],
                representation["f_ij"],
                representation["rcut_ij"],
            )
            x = x + v  # Update atomic features

        return {
            "scalar_representation": x,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }

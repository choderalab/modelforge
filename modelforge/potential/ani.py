import torch
from torch import nn
from loguru import logger as log
from modelforge.potential.models import BaseNNP
from modelforge.potential.postprocessing import PostprocessingPipeline, NoPostprocess
from typing import Dict


class ANIRepresentation(nn.Module):
    # calculate the atomic environment vectors
    # used for the ANI architecture of NNPs

    def __init__(self):
        # radial symmetry functions
        from .utils import RadialSymmetryFunction, AngularSymmetryFunction

        self.radial_symmetry_functions = RadialSymmetryFunction()
        self.angular_symmetry_functions = AngularSymmetryFunction()


class ANIInteraction(nn.Module):

    def __init__(self):
        pass


class ANI2x(BaseNNP):

    def __init__(
        self,
        postprocessing: PostprocessingPipeline = PostprocessingPipeline(
            [NoPostprocess({})]
        ),
    ) -> None:
        """
        Initialize the ANi NNP architeture.

        Parameters
        ----------
        nr_filters : int, optional
            Number of filters; defines the dimensionality of the intermediate features (default is 2).
        """

        log.debug("Initializing SchNet model.")
        super().__init__(
            cutoff=float(cutoff_module.radial_cutoff), postprocessing=postprocessing
        )

        # initialize the energy readout
        from .utils import EnergyReadout

        # Initialize representation block
        self.ani_representation_module = ANIRepresentation(self.radial_basis_module)
        # Intialize interaction blocks
        self.interaction_modules = ANIRepresentation()

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        inputs = self._prepare_inputs(inputs)
        inputs = self._model_specific_input_preparation(inputs)
        return inputs

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        from modelforge.potential.utils import embed_atom_features

        atomic_embedding = embed_atom_features(
            inputs["atomic_numbers"], self.embedding_module
        )
        inputs["atomic_embedding"] = atomic_embedding
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

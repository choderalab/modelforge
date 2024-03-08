from .models import BaseNNP
from loguru import logger as log
from openff.units import unit
import torch
from typing import Dict


class PhysNET(BaseNNP):
    def __init__(
        self,
        max_Z: int = 100,
        embedding_dimensions: int = 64,
        cutoff: unit.Quantity = 5 * unit.angstrom,
    ) -> None:
        """
        Initialize the PhysNET class.

        Physnet is a neural network potential first described here: https://arxiv.org/pdf/1902.08408.pdf

        The potential takes as input the nuclear charges (Z_i) and pairwise distances (r_i)
        and outputs the atomic energies E_i and partial charges q_i.
        The architecture has the following components:

        - embedding layer: transforming nuclear charges into a vector representation
        - radial basis function layer: transforming the pairwise distances into a vector representation
        - module layer (see Fig 2 for graphical representation):
            embedding and radial basis output are provided as input,
            consists of (1) interaction, (2) residula and (3) output module

        Parameters
        ----------
        max_Z : nn.int
        embedding_dimensions : int
        """

        log.debug("Initializing PhysNET model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.nr_atom_basis = embedding_dimensions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, embedding_dimensions)

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff, self.device)

        # initialize the energy readout
        from .utils import EnergyReadout

        self.readout_module = EnergyReadout(embedding_dimensions)

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        inputs["atomic_embedding"] = self.embedding_module(inputs["atomic_numbers"])
        return inputs

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
        - atomic_embedding : torch.Tensor
            Atomic numbers embedding; shape (nr_of_atoms_in_systems, 1, nr_atom_basis).
        - pairlist:  shape (n_pairs, 2)
        - r_ij:  shape (n_pairs, 3)
        - d_ij:  shape (n_pairs, 1)
        - positions:  shape (nr_of_atoms_per_molecules, 3)
        - atomic_embedding:  shape (nr_of_atoms_in_systems, nr_atom_basis)


        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """
        pass
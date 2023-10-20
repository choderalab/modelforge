from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW


class PairList(nn.Module):
    """A module to handle pair list calculations.

    Attributes
    ----------
    cutoff : float
        The cutoff distance for neighbor calculations.
    """

    def __init__(self, cutoff: float = 5.0):
        """Initialize PairList.

        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for neighbor calculations, default is 5.0.
        """
        super().__init__()
        from .utils import neighbor_pairs_nopbc

        self.calculate_neighbors = neighbor_pairs_nopbc
        self.cutoff = cutoff

    def compute_r_ij(self, atom_index12: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Compute displacement vector between atom pairs.

        Parameters
        ----------
        atom_index12 : torch.Tensor, shape [n_pairs, 2]
            Atom indices for pairs of atoms
        R : torch.Tensor, shape [batch_size, n_atoms, 3]
            Atom coordinates.

        Returns
        -------
        torch.Tensor, shape [n_pairs, 3]
            Displacement vector between atom pairs.
        """
        coordinates = R.flatten(0, 1)
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(
            2, -1, 3
        )
        return selected_coordinates[0] - selected_coordinates[1]

    def forward(
        self, mask_padding: torch.Tensor, positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for PairList.

        Parameters
        ----------
        mask : torch.Tensor, shape [batch_size, n_atoms]
            Mask tensor.
        positions : torch.Tensor, shape [batch_size, n_atoms, n_dims]
            Position tensor.

        Returns
        -------
        dict : Dictionary containing atom index pairs, distances, and displacement vectors.
            - 'pairlist': Dict[str, torch.Tensor], contains pairlist :int; (n_paris,2),
                r_ij:float; (n_pairs, 1) , d_ij: float; (n_pairs, 3)

        """
        pairlist = self.calculate_neighbors(mask_padding, positions, self.cutoff)
        r_ij = self.compute_r_ij(pairlist, positions)

        return {
            "pairlist": pairlist,
            "d_ij": r_ij.norm(2, -1),
            "r_ij": r_ij,
        }


class LightningModuleMixin(pl.LightningModule):
    """A mixin for PyTorch Lightning training."""

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch data. Expected to include 'E', a tensor with shape [batch_size, 1].
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor, shape [batch_size, 1]
            Loss tensor.
        """

        E_hat = self.forward(batch).flatten()
        loss = self.loss_function(E_hat, batch["E"])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for training.

        Returns
        -------
        AdamW
            The AdamW optimizer.
        """

        return self.optimizer(self.parameters(), lr=self.learning_rate)


class BaseNNP(nn.Module):
    """Abstract base class for neural network potentials."""

    def __init__(self, cutoff: float = 0.5):
        """
        Initialize the NNP class.
        """
        super().__init__()
        self.cutoff = cutoff  # in nanometer

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
            - 'atomic_numbers': int; shape (n_systems, n_atoms), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : int; shape (n_system:int)
            - 'positions': float; shape (n_systems, n_atoms, 3)
            - 'pairlist': Dict[str, torch.Tensor], contains pairlist :int; (n_paris,2),
                r_ij:float; (n_pairs, 1) , d_ij: float; (n_pairs, 3), 'atomic_subsystem_index':int; (n_atoms)

        Returns
        -------
        torch.Tensor
            Calculated energies; float; shape (n_systems).

        Raises
        ------
        NotImplementedError
            This method needs to be implemented by subclasses.
        """

        raise NotImplementedError


class SingleTopologyAlchemicalBaseNNPModel(BaseNNP):
    def forward(inputs: Dict[str, torch.Tensor]):
        """
        Calculate the alchemical energy for a given input batch.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
            - 'atomic_numbers': shape (n_systems:int, n_atoms:int), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : shape (n_system:int)
            - (only for alchemical transformation) 'alchemical_atomic_number': shape (n_atoms:int)
            - (only for alchemical transformation) 'lamb': float
            - 'positions': shape (n_atoms, 3)
            - 'pairlist': Dict[str, torch.Tensor], contains pairlist (n_paris,2),
                r_ij (n_pairs, 1) , d_ij (n_pairs, 3), 'atomic_subsystem_index' (n_atoms)

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (n_systems,).
        """

        # emb = nn.Embedding(1,200)
        # lamb_emb = (1 - lamb) * emb(input['Z1']) + lamb * emb(input['Z2'])	def __init__():
        raise NotImplementedError

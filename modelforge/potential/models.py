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

    def forward(self, R: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for PairList.

        Parameters
        ----------
        mask : torch.Tensor, shape [batch_size, n_atoms]
            Mask tensor.
        R : torch.Tensor, shape [batch_size, n_atoms, n_dims]
            Position tensor.

        Returns
        -------
        dict
            Dictionary containing atom index pairs, distances, and displacement vectors.
        """
        atom_index12 = self.calculate_neighbors(R, self.cutoff)
        r_ij = self.compute_r_ij(atom_index12, R)
        return {"atom_index12": atom_index12, "d_ij": r_ij.norm(2, -1), "r_ij": r_ij}


class LighningModuleMixin(pl.LightningModule):
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

    def __init__(self):
        """
        Initialize the NNP class.
        """
        super().__init__()
        # NOTE: let's add a debug mode to the NNP class

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        inputs : dict
            Dictionary of input tensors, shapes are context-dependent.

        Returns
        -------
        torch.Tensor, shape [...]
            Output tensor.

        Raises
        ------
        NotImplementedError
            This method needs to be implemented by subclasses.
        """

        raise NotImplementedError

from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW


class PairList(nn.Module):
    def __init__(self, cutoff: float = 5.0):
        """
        Initialize the PairList class.
        """
        super().__init__()
        from .utils import neighbor_pairs_nopbc

        self.calculate_neighbors = neighbor_pairs_nopbc
        self.cutoff = cutoff

    def compute_distance(
        self, atom_index12: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances based on atom indices and coordinates.

        Parameters
        ----------
        atom_index12 : torch.Tensor, shape [n_pairs, 2]
            Atom indices for pairs of atoms
        R : torch.Tensor, shape [batch_size, n_atoms, n_dims]
            Atom coordinates.

        Returns
        -------
        torch.Tensor, shape [n_pairs]
            Computed distances.
        """

        coordinates = R.flatten(0, 1)
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(
            2, -1, 3
        )
        vec = selected_coordinates[0] - selected_coordinates[1]
        return vec.norm(2, -1)

    def forward(self, mask, R) -> Dict[str, torch.Tensor]:
        atom_index12 = self.calculate_neighbors(mask, R, self.cutoff)
        d_ij = self.compute_distance(atom_index12, R)
        return {"atom_index12": atom_index12, "d_ij": d_ij}


class LighningModuleMixin(pl.LightningModule):
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the training loop.

        Parameters
        ----------
        batch : dict
            Batch data.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            The loss tensor.
        """

        E_hat = self.forward(batch)
        loss = self.loss_function(E_hat.energies, batch["E"])
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # NOTE: let's pass a callable

        return loss

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for training.

        Returns
        -------
        AdamW
            The AdamW optimizer.
        """

        optimizer = self.optimizer()
        return optimizer


class BaseNNP(nn.Module):
    """
    Abstract base class for neural network potentials.
    This class defines the overall structure and ensures that subclasses
    implement the `calculate_energies_and_forces` method.

    Methods
    -------
    forward(inputs: dict) -> SpeciesEnergies:
        Forward pass for the neural network potential.
    calculate_energy(inputs: dict) -> torch.Tensor:
        Placeholder for the method that should calculate energies and forces.
    training_step(batch, batch_idx) -> torch.Tensor:
        Defines the train loop.
    configure_optimizers() -> AdamW:
        Configures the optimizer.
    """

    def __init__(self):
        """
        Initialize the NNP class.
        """
        super().__init__()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the neural network potential.

        Parameters
        ----------
        inputs : dict
            A dictionary containing atomic numbers, positions, etc.

        Returns
        -------
        output: torch.Tensor
            energies.
        """

        raise NotImplementedError

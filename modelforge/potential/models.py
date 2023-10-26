from abc import ABC, abstractmethod
from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW


class PairList(nn.Module):
    """
    A module to handle pair list calculations for neighbor atoms.

    Attributes
    ----------
    cutoff : float
        The cutoff distance for neighbor calculations.

    Methods
    -------
    compute_r_ij(atom_pairs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor
        Compute the displacement vector between atom pairs.
    forward(mask: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]
        Forward pass for PairList.
    """

    def __init__(self, cutoff: float = 5.0):
        """
        Initialize PairList.

        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for neighbor calculations, default is 5.0.
        """
        super().__init__()
        from .utils import neighbor_pairs_nopbc

        self.calculate_neighbors = neighbor_pairs_nopbc
        self.cutoff = cutoff

    def compute_r_ij(
        self, atom_pairs: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vector between atom pairs.

        Parameters
        ----------
        atom_pairs : torch.Tensor, shape [n_pairs, 2]
            Atom indices for pairs of atoms
        positions : torch.Tensor, shape [batch_size, n_atoms, 3]
            Atom coordinates.

        Returns
        -------
        torch.Tensor, shape [n_pairs, 3]
            Displacement vector between atom pairs.
        """
        coordinates = positions.flatten(0, 1)
        selected_coordinates = coordinates.index_select(0, atom_pairs.view(-1)).view(
            2, -1, 3
        )
        return selected_coordinates[0] - selected_coordinates[1]

    def forward(
        self, mask_padding: torch.Tensor, positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PairList.

        Parameters
        ----------
        mask : torch.Tensor, shape [batch_size, n_atoms]
            Mask tensor.
        positions : torch.Tensor, shape [batch_size, n_atoms, n_dims]
            Position tensor.

        Returns
        -------
        dict : Dict[str, torch.Tensor], containing atom index pairs, distances, and displacement vectors.
            - 'pairlist': torch.Tensor, shape (n_paris,2)
            - 'r_ij' : torch.Tensor, shape (n_pairs, 1) ,
            - 'd_ij' : torch.Tenso, shape (n_pairs, 3)

        """
        pairlist = self.calculate_neighbors(mask_padding, positions, self.cutoff)
        r_ij = self.compute_r_ij(pairlist, positions)

        return {
            "pairlist": pairlist,
            "d_ij": r_ij.norm(2, -1),
            "r_ij": r_ij,
        }


class LightningModuleMixin(pl.LightningModule):
    """
    A mixin for PyTorch Lightning training.

    Methods
    -------
    training_step(batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor
        Perform a single training step.
    configure_optimizers() -> AdamW
        Configures the optimizer for training.
    """

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

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
        loss = self.loss_function(E_hat, batch["E_label"])
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


# Abstract Base Class
class AbstractBaseNNP(nn.Module, ABC):
    """
    Abstract base class for neural network potentials.

    Methods
    -------
    forward(inputs: Dict[str, torch.Tensor]) -> torch.Tensor
        Abstract method for forward pass in neural network potentials.
    input_checks(inputs: Dict[str, torch.Tensor])
        Perform input checks to validate the input dictionary.
    """

    def __init__(self):
        """
        Initialize the AbstractBaseNNP class.
        """
        super().__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for neighborlist calculation and forward pass in neural network potentials.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            - 'atomic_numbers', shape (nr_systems, nr_atoms), 0 indicates non-interacting atoms that will be masked
            - 'total_charge', shape (nr_systems, 1)
            - 'positions', shape (n_atoms, 3)
            - 'boxvectors', shape (3, 3)

        Returns
        -------
        torch.Tensor
            Calculated output; shape is implementation-dependent.

        """
        pass

    @abstractmethod
    def _forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Abstract method for forward pass in neural network potentials.
        This method is called by `forward`.

        Parameters
        ----------
        inputs: Dict[str, torch.Tensor]
            - pairlist, shape (n_paris,2)
            - r_ij, shape (n_pairs, 1)
            - d_ij, shape (n_pairs, 3)
            - 'atomic_subsystem_index' (optional), shape n_atoms
            - positions, shape (n_systems, n_atoms, 3)

        """
        pass

    def input_checks(self, inputs: Dict[str, torch.Tensor]):
        """
        Perform input checks to validate the input dictionary.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing necessary data for the forward pass.
            The exact keys and shapes are implementation-dependent.

        Raises
        ------
        ValueError
            If the input dictionary is missing required keys or has invalid shapes.

        """
        required_keys = ["atomic_numbers", "positions"]
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key: {key}")

        if inputs["atomic_numbers"].dim() != 2:
            raise ValueError("Shape mismatch: 'atomic_numbers' should be a 2D tensor.")

        if inputs["positions"].dim() != 3:
            raise ValueError("Shape mismatch: 'positions' should be a 3D tensor.")


class BaseNNP(AbstractBaseNNP):
    """
    Abstract base class for neural network potentials.

    Attributes
    ----------
    nr_of_embeddings : int
        Number of embeddings.
    nr_atom_basis : int
        Number of atom basis.
    cutoff : float
        Cutoff distance for neighbor calculations.

    Methods
    -------
    forward(inputs: Dict[str, torch.Tensor]) -> torch.Tensor
        Abstract method for forward pass in neural network potentials.
    """

    def __init__(
        self,
        nr_of_embeddings: int,
        nr_atom_basis: int,
        cutoff: float = 5.0,
    ):
        """
        Initialize the NNP class.

        Parameters
        ----------
        nr_of_embeddings : int
            Number of embeddings.
        nr_atom_basis : int
            Number of atom basis.
        cutoff : float, optional
            Cutoff distance (in Angstrom) for neighbor calculations, default is 5.0.
        """
        from .models import PairList

        super().__init__()
        self.calculate_distances_and_pairlist = PairList(cutoff)
        self.embedding = nn.Embedding(nr_of_embeddings, nr_atom_basis, padding_idx=0)
        self.nr_of_embeddings = nr_of_embeddings

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
            - 'atomic_numbers': int; shape (n_systems, n_atoms), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : int; shape (n_system)
            - 'positions': float; shape (n_systems, n_atoms, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; float; shape (n_systems).

        """
        self.input_checks(inputs)
        atomic_numbers = inputs["atomic_numbers"]  # shape (n_systems, n_atoms, 3)
        positions = inputs["positions"]  # shape (n_systems, n_atoms, 3)
        mask_padding = atomic_numbers == 0

        pairlist = self.calculate_distances_and_pairlist(mask_padding, positions)
        atomic_numbers_embedding = self.embedding(
            atomic_numbers
        )  # shape (batch_size, n_atoms, n_atom_basis)
        inputs = {
            "pairlist": pairlist["pairlist"],
            "d_ij": pairlist["d_ij"],
            "r_ij": pairlist["r_ij"],
            "atomic_numbers_embedding": atomic_numbers_embedding,
            "positions": positions,
            "atomic_numbers": atomic_numbers,
        }
        return self._forward(inputs)


class SingleTopologyAlchemicalBaseNNPModel(AbstractBaseNNP):
    """
    Subclass for handling alchemical energy calculations.

    Methods
    -------
    forward(inputs: Dict[str, torch.Tensor]) -> torch.Tensor
        Calculate the alchemical energy for a given input batch.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]):
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

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (n_systems,).
        """

        # emb = nn.Embedding(1,200)
        # lamb_emb = (1 - lamb) * emb(input['Z1']) + lamb * emb(input['Z2'])	def __init__():
        self.input_checks(inputs)
        raise NotImplementedError

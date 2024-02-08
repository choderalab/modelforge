from abc import ABC, abstractmethod
from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from loguru import logger as log


class PairList(nn.Module):
    """
    A module to handle pair list calculations for atoms.
    This returns a pair list of atom indices and the displacement vectors between them.


    Methods
    -------
    compute_r_ij(atom_pairs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor
        Compute the displacement vector between atom pairs.
    forward(mask: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]
        Forward pass for PairList.
    """

    def __init__(self, only_unique_pairs: bool = False):
        """
        Initialize PairList.

        Parameters
        ----------
        only_unique_pairs : bool, optional
            If set to True, only unique pairs of atoms are considered, default is False.
        """
        super().__init__()
        from .utils import pair_list

        self.calculate_pairs = pair_list
        self.only_unique_pairs = only_unique_pairs

    def calculate_r_ij(
        self, pair_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vector between atom pairs.

        Parameters
        ----------
        pair_indices : torch.Tensor, shape [2, n_pairs]
            Atom indices for pairs of atoms
        positions : torch.Tensor, shape [nr_systems, nr_atoms, 3]
            Atom positions.

        Returns
        -------
        torch.Tensor, shape [n_pairs, 3]
            Displacement vector between atom pairs.
        """
        # Select the pairs of atom coordinates from the positions
        selected_positions = positions.index_select(0, pair_indices.view(-1)).view(
            2, -1, 3
        )
        return selected_positions[1] - selected_positions[0]

    def calculate_d_ij(self, r_ij):
        # Calculate the euclidian distance between the atoms in the pair
        return r_ij.norm(2, -1).unsqueeze(-1)

    def forward(
        self, positions: torch.Tensor, atomic_subsystem_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PairList.

        Parameters
        ----------
        positions : torch.Tensor, shape [nr_systems, nr_atoms, 3]
            Position tensor.
        atomic_subsystem_indices : torch.Tensor, shape [nr_atoms]
        Returns
        -------
        dict : Dict[str, torch.Tensor], containing atom index pairs, distances, and displacement vectors.
            - 'pair_indices': torch.Tensor, shape (2, n_pairs)
            - 'r_ij' : torch.Tensor, shape (3, n_pairs)
            - 'd_ij' : torch.Tensor, shape (1, n_pairs)

        """
        pair_indices = self.calculate_pairs(
            atomic_subsystem_indices,
            only_unique_pairs=self.only_unique_pairs,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)

        return {
            "pair_indices": pair_indices,
            "d_ij": self.calculate_d_ij(r_ij),
            "r_ij": r_ij,
        }


class NeighborList(PairList):
    def __init__(self, cutoff: float, only_unique_pairs: bool = False):
        """
        Initialize Neighborlist.
        A neighbor list is a list of atom pairs that are within a certain cutoff distance.

        Parameters
        ----------
        cutoff : float
            Cutoff distance for neighbor calculations.
        only_unique_pairs : bool, optional
            If set to True, only unique pairs of atoms are considered, default is False.
        """
        super().__init__(only_unique_pairs=only_unique_pairs)
        from .utils import neighbor_list_with_cutoff

        self.calculate_pairs = neighbor_list_with_cutoff
        self.cutoff = cutoff

    def forward(
        self, positions: torch.Tensor, atomic_subsystem_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for NeighborList.

        Parameters
        ----------
        positions : torch.Tensor, shape [nr_systems, nr_atoms, 3]
            Position tensor.
        atomic_subsystem_indices : torch.Tensor, shape [nr_atoms]
        Returns
        -------
        dict : Dict[str, torch.Tensor], containing atom index pairs, distances, and displacement vectors.
            - 'pair_indices': torch.Tensor, shape (2, n_pairs)
            - 'r_ij' : torch.Tensor, shape (3, n_pairs)
            - 'd_ij' : torch.Tensor, shape (1, n_pairs)

        """
        pair_indices = self.calculate_pairs(
            positions,
            atomic_subsystem_indices,
            cutoff=self.cutoff,
            only_unique_pairs=self.only_unique_pairs,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)

        return {
            "pair_indices": pair_indices,
            "d_ij": self.calculate_d_ij(r_ij),
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


class BaseNNP(nn.Module):
    """
    Base class for neural network potentials.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
    ):
        """
        Initialize the NNP class.

        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance (in Angstrom) for neighbor calculations, default is 5.0.
        """
        from .models import PairList

        super().__init__()
        self._cutoff = cutoff
        self.calculate_distances_and_pairlist = PairList()
        self._dtype = None  # set at runtime

    def preate_input(self, inputs: Dict[str, torch.Tensor]):
        # needs to be implemented by the subclass
        # if subclass needs any additional input preparation (e.g. embedding),
        # it should be done here
        raise NotImplementedError

    def _forward(self, inputs: Dict[str, torch.Tensor]):
        # needs to be implemented by the subclass
        # perform the forward pass implemented in the subclass
        raise NotImplementedError

    def _readout(self, input: Dict[str, torch.Tensor]):
        # needs to be implemented by the subclass
        # perform the readout operation implemented in the subclass
        raise NotImplementedError

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        atomic_numbers = inputs["atomic_numbers"]  # shape (nr_of_atoms_in_batch, 1)
        positions = inputs["positions"]  # shape (nr_of_atoms_in_batch, 3)
        atomic_subsystem_indices = inputs["atomic_subsystem_indices"]
        nr_of_atoms_in_batch = inputs["atomic_numbers"].shape[0]

        r = self.calculate_distances_and_pairlist(positions, atomic_subsystem_indices)

        return {
            "pair_indices": r["pair_indices"],
            "d_ij": r["d_ij"],
            "r_ij": r["r_ij"],
            "nr_of_atoms_in_batch": nr_of_atoms_in_batch,
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "atomic_subsystem_indices": atomic_subsystem_indices,
        }

    def _set_dtype(self):
        dtypes = list({p.dtype for p in self.parameters()})
        assert len(dtypes) == 1
        self._dtype = dtypes[0]
        log.debug(f"Setting dtype to {self._dtype}.")

    def _input_checks(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        required_keys = ["atomic_numbers", "positions", "atomic_subsystem_indices"]
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key: {key}")

        if inputs["atomic_numbers"].dim() != 2:
            raise ValueError("Shape mismatch: 'atomic_numbers' should be a 2D tensor.")

        if inputs["positions"].dim() != 2:
            raise ValueError("Shape mismatch: 'positions' should be a 2D tensor.")

        if inputs["positions"].dtype != self._dtype:
            log.debug(
                f"Precision mismatch: dtype of positions tensor is {inputs['positions'].dtype}, "
                f"but dtype of model parameters is {self._dtype}. "
                f"Setting dtype of positions tensor to {self._dtype}."
            )
            inputs["positions"] = inputs["positions"].to(self._dtype)
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
            - 'atomic_numbers': int; shape (nr_of_atoms_in_batch, 1), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : int; shape (n_system)
            - 'positions': float; shape (nr_of_atoms_in_batch, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; float; shape (n_systems).

        """
        # adjust the dtype of the input tensors to match the model parameters
        self._set_dtype()
        # perform input checks
        inputs = self._input_checks(inputs)
        # prepare the input for the forward pass
        inputs = self.prepare_inputs(inputs)
        # perform the forward pass implemented in the subclass
        output = self._forward(inputs)

        return self._readout(output)


class SingleTopologyAlchemicalBaseNNPModel(BaseNNP):
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
            - 'atomic_numbers': shape (nr_of_atoms_in_batch, *, *), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : shape (nr_of_atoms_in_batch)
            - (only for alchemical transformation) 'alchemical_atomic_number': shape (nr_of_atoms_in_batch)
            - (only for alchemical transformation) 'lamb': float
            - 'positions': shape (nr_of_atoms_in_batch, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (n_systems,).
        """

        # emb = nn.Embedding(1,200)
        # lamb_emb = (1 - lamb) * emb(input['Z1']) + lamb * emb(input['Z2'])	def __init__():
        self._input_checks(inputs)
        raise NotImplementedError

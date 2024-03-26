from typing import Dict, Tuple

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as F

from loguru import logger as log

from abc import ABC, abstractmethod
import torch
from typing import NamedTuple, TYPE_CHECKING
from modelforge.potential.utils import (
    AtomicSelfEnergies,
    NNPInput,
    BatchData,
)
from abc import abstractmethod, ABC
from openff.units import unit
from typing import Dict, Type

if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics


# Define NamedTuple for the outputs of Pairlist and Neighborlist forward method
class PairListOutputs(NamedTuple):
    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class EnergyOutput(NamedTuple):
    E: torch.Tensor
    raw_E: torch.Tensor
    rescaled_E: torch.Tensor
    molecular_ase: torch.Tensor


class Pairlist(nn.Module):
    """Handle pair list calculations for atoms, returning atom indices pairs and displacement vectors.

    Attributes
    ----------
    calculate_pairs : callable
        A function to calculate pairs given specific criteria.

    Methods
    -------
    calculate_r_ij(pair_indices, positions)
        Computes the displacement vector between atom pairs.
    calculate_d_ij(r_ij)
        Computes the Euclidean distance between atoms in a pair.
    forward(positions, atomic_subsystem_indices, only_unique_pairs=False)
        Forward pass to compute pair indices, distances, and displacement vectors.
    """

    def __init__(self):
        """
        Initialize PairList.
        """
        super().__init__()
        from .utils import pair_list

        self.calculate_pairs = pair_list

    def calculate_r_ij(
        self, pair_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vectors between atom pairs.

        Parameters
        ----------
        pair_indices : torch.Tensor
            Atom indices for pairs of atoms. Shape: [2, n_pairs].
        positions : torch.Tensor
            Atom positions. Shape: [atoms, 3].

        Returns
        -------
        torch.Tensor
            Displacement vectors between atom pairs. Shape: [n_pairs, 3].
        """
        # Select the pairs of atom coordinates from the positions
        selected_positions = positions.index_select(0, pair_indices.view(-1)).view(
            2, -1, 3
        )
        return selected_positions[1] - selected_positions[0]

    def calculate_d_ij(self, r_ij: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distances between atoms in each pair.

        Parameters
        ----------
        r_ij : torch.Tensor
            Displacement vectors between atoms in a pair. Shape: [n_pairs, 3].

        Returns
        -------
        torch.Tensor
            Euclidean distances. Shape: [n_pairs, 1].
        """
        return r_ij.norm(dim=1).unsqueeze(1)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        only_unique_pairs: bool = False,
    ) -> PairListOutputs:
        """
        Compute interacting pairs, distances, and displacement vectors.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices to identify atoms in subsystems. Shape: [nr_atoms].
        only_unique_pairs : bool, optional
            If True, considers only unique pairs of atoms. Default is False.

        Returns
        -------
        PairListOutputs
            A NamedTuple containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """
        pair_indices = self.calculate_pairs(
            atomic_subsystem_indices,
            only_unique_pairs=only_unique_pairs,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)

        return PairListOutputs(
            pair_indices=pair_indices,
            d_ij=self.calculate_d_ij(r_ij),
            r_ij=r_ij,
        )


from openff.units import unit


class Neighborlist(Pairlist):
    """Manage neighbor list calculations with a specified cutoff distance.

    This class extends Pairlist to consider a cutoff distance for neighbor calculations.

    Attributes
    ----------
    cutoff : unit.Quantity
        Cutoff distance for neighbor list calculations.
    """

    def __init__(self, cutoff: unit.Quantity):
        """
        Initialize the Neighborlist with a specific cutoff distance.

        Parameters
        ----------
        cutoff : unit.Quantity
            Cutoff distance for neighbor calculations.
        """
        super().__init__()
        from .utils import neighbor_list_with_cutoff

        self.calculate_pairs = neighbor_list_with_cutoff
        self.cutoff = cutoff

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        only_unique_pairs: bool = False,
    ) -> PairListOutputs:
        """
        Forward pass to compute neighbor list considering a cutoff distance.

        Overrides the `forward` method from Pairlist to include cutoff distance in calculations.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_systems, nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices identifying atoms in subsystems. Shape: [nr_atoms].
        only_unique_pairs : bool, optional
            If True, considers only unique pairs of atoms. Default is False.

        Returns
        -------
        PairListOutputs
            A NamedTuple containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """

        pair_indices = self.calculate_pairs(
            positions,
            atomic_subsystem_indices,
            cutoff=self.cutoff,
            only_unique_pairs=only_unique_pairs,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)

        return PairListOutputs(
            pair_indices=pair_indices,
            d_ij=self.calculate_d_ij(r_ij),
            r_ij=r_ij,
        )


from typing import Callable, Optional


class BaseNeuralNetworkPotential(pl.LightningModule, ABC):
    """Abstract base class for neural network potentials.

    Attributes
    ----------
    cutoff : unit.Quantity
        Cutoff distance for neighbor list calculations.
    loss_function : Type[nn.Module]
        Loss function for training the neural network.
    optimizer : Type[torch.optim.Optimizer]
        Optimizer used for training.
    learning_rate : float
        Learning rate for the optimizer.
    calculate_distances_and_pairlist : Neighborlist
        Module for calculating distances and pairlist with a given cutoff.
    readout_module : FromAtomToMoleculeReduction
        Module for reading out per molecule properties from atomic properties.
    """

    def __init__(
        self,
        cutoff: unit.Quantity,
        loss: Callable = F.mse_loss,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ):
        """Initialize the neural network potential class with specified parameters."""
        from .models import Neighborlist
        from modelforge.dataset.dataset import DatasetStatistics

        super().__init__()
        self.calculate_distances_and_pairlist = Neighborlist(cutoff)
        self._dtype: Optional[bool] = None  # set at runtime
        self._log_message_dtype = False
        self._log_message_units = False
        self._dataset_statistics = DatasetStatistics(0.0, 1.0, AtomicSelfEnergies())
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr
        # initialize the per molecule readout module
        from .utils import FromAtomToMoleculeReduction

        self.readout_module = FromAtomToMoleculeReduction()

    @abstractmethod
    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist: PairListOutputs
    ):
        """
        Prepares model-specific inputs before the forward pass.

        This method should be implemented by subclasses to accommodate any
        model-specific processing of inputs.

        Parameters
        ----------
        data : NNPInput
            The initial inputs to the neural network model.
        pairlist : PairListOutputs

        Returns
        -------
            The processed inputs, ready for the model's forward pass.
        """
        pass

    @abstractmethod
    def _forward(self, inputs):
        # needs to be implemented by the subclass
        # perform the forward pass implemented in the subclass
        pass

    @property
    def dataset_statistics(self):
        return self._dataset_statistics

    @dataset_statistics.setter
    def dataset_statistics(self, value: "DatasetStatistics"):
        from modelforge.dataset.dataset import DatasetStatistics

        if not isinstance(value, DatasetStatistics):
            raise ValueError("Value must be an instance of DatasetStatistics.")

        self._dataset_statistics = value

    def update_dataset_statistics(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.dataset_statistics, key):
                setattr(self.dataset_statistics, key, value)
            else:
                log.warning(f"{key} is not a valid field of DatasetStatistics.")

    def _readout(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # readout the per molecule values
        return self.readout_module(x, index)

    def _log_batch_size(self, y: torch.Tensor):
        batch_size = int(y.numel())
        return batch_size

    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch data. Expected to include 'E_predict', a tensor with shape [batch_size, 1].
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor, shape [batch_size, 1]
            Loss tensor.
        """
        E_true, E_predict = self._get_energies(batch)

        import math

        # time.sleep(1)
        loss = self.loss_function(E_true, E_predict)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def test_step(self, batch: BatchData, batch_idx):
        from torch.nn import functional as F

        E_true, E_predict = self._get_energies(batch)

        test_loss = F.mse_loss(E_true, E_predict)
        self.log(
            "test_loss",
            test_loss,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: BatchData, batch_idx):
        from torch.nn import functional as F

        E_true, E_predict = self._get_energies(batch)

        val_loss = F.l1_loss(E_true, E_predict)
        self.log(
            "val_loss",
            val_loss,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
        )

    def _get_energies(self, batch: BatchData) -> Tuple[torch.Tensor, torch.Tensor]:

        nnp_input = batch.nnp_input
        E_true = batch.metadata.E.to(torch.float32).squeeze(1)
        self.batch_size = self._log_batch_size(E_true)

        E_predict = self.forward(nnp_input).E
        return E_true, E_predict

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for training.

        Returns
        -------
        AdamW
            The AdamW optimizer.
        """

        return self.optimizer(self.parameters(), lr=self.learning_rate)

    def _rescale_energy(self, energies: torch.Tensor) -> torch.Tensor:

        return (
            energies * self.dataset_statistics.scaling_stddev
            + self.dataset_statistics.scaling_mean
        )

    def _calculate_molecular_self_energy(
        self, data: NNPInput, number_of_molecules: int
    ) -> torch.Tensor:

        atomic_numbers = data.atomic_numbers
        atomic_subsystem_indices = data.atomic_subsystem_indices.to(
            dtype=torch.long, device=self.device
        )

        # atomic_number_to_energy
        atomic_self_energies = self.dataset_statistics.atomic_self_energies
        ase_tensor_for_indexing = atomic_self_energies.ase_tensor_for_indexing.to(
            device=self.device
        )

        # first, we need to use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        # then, we use the atomic_subsystem_indices to scatter add the atomic self
        # energies in the ase_tensor to generate the molecular self energies
        ase_tensor_zeros = torch.zeros((number_of_molecules,)).to(device=self.device)
        ase_tensor = ase_tensor_zeros.scatter_add(
            0, atomic_subsystem_indices, ase_tensor
        )

        return ase_tensor

    def _energy_postprocessing(
        self, properties_per_molecule, inputs
    ) -> Dict[str, torch.Tensor]:

        # first, resale the energies
        processed_energy = {}
        processed_energy["raw_E"] = properties_per_molecule.clone().detach()
        properties_per_molecule = self._rescale_energy(properties_per_molecule)
        processed_energy["rescaled_E"] = properties_per_molecule.clone().detach()
        # then, calculate the molecular self energy
        molecular_ase = self._calculate_molecular_self_energy(
            inputs, properties_per_molecule.numel()
        )
        processed_energy["molecular_ase"] = molecular_ase.clone().detach()
        # add the molecular self energy to the rescaled energies
        processed_energy["E"] = properties_per_molecule + molecular_ase
        return processed_energy

    def prepare_inputs(self, data: NNPInput, only_unique_pairs: bool = True):
        """Prepares the input tensors for passing to the model.

        Performs general input manipulation like calculating distances,
        generating the pair list, etc. Also calls the model-specific input
        preparation.

        Parameters
        ----------
        data : NNPInput
            NameTuple containing the data provided by the dataset.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs or not in the pairlist.
        Returns
        -------
        nnp_input
            NamedTuple containg the relevant data for the model.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        pairlist_output = self.calculate_distances_and_pairlist(
            positions, atomic_subsystem_indices, only_unique_pairs
        )

        # ---------------------------
        # perform model specific modifications
        nnp_input = self._model_specific_input_preparation(data, pairlist_output)

        return nnp_input

    def _set_dtype(self):
        dtypes = list({p.dtype for p in self.parameters()})
        assert len(dtypes) == 1, f"Multiple dtypes: {dtypes}"

        if not self._log_message_dtype:
            log.debug(f"Setting dtype to {dtypes[0]}.")
            self._log_message_dtype = True

        if self._dtype is not None and self._dtype != dtypes[0]:
            log.warning(f"Setting dtype to {dtypes[0]}.")
            log.warning(f"This is new, be carful. You are resetting the dtype!")

        self._dtype = dtypes[0]

    def _input_checks(self, data: NNPInput):
        """
        Perform input checks to validate the input dictionary.

        Raises
        ------
        ValueError
            If the input dictionary is missing required keys or has invalid shapes.

        """
        # check that the input is instance of NNPInput
        assert isinstance(data, NNPInput)

        nr_of_atoms = data.atomic_numbers.shape[0]
        assert data.atomic_numbers.shape == torch.Size([nr_of_atoms])
        assert data.atomic_subsystem_indices.shape == torch.Size([nr_of_atoms])
        nr_of_molecules = torch.unique(data.atomic_subsystem_indices).numel()
        assert data.total_charge.shape == torch.Size([nr_of_molecules])
        assert data.positions.shape == torch.Size([nr_of_atoms, 3])

    def forward(self, data: NNPInput) -> EnergyOutput:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        data : Metadata
            NamedTuple containing the following fields.
        - atomic_numbers: torch.Tensor
            Contains the atomic number (nuclear charge) of each atom, shape (nr_of_atoms).
        - atom_index: torch.Tensor
            Contains the index of the atom in the molecule, shape (nr_of_atoms).
        - atomic_subsystem_indices: torch.Tensor
            Contains the index of the subsystem the atom belongs to, shape (nr_of_atoms).
        - 'total_charge' : torch.Tensor
            Contains the total charge per molecule, shape (number_of_molecules).
        - 'positions': torch.Tensor
            Positions of each atom, shape (nr_of_atoms, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; float; shape (n_systems).

        """
        # adjust the dtype of the input tensors to match the model parameters
        self._set_dtype()
        # perform input checks
        self._input_checks(data)
        # prepare the input for the forward pass
        inputs = self.prepare_inputs(data, self.only_unique_pairs)
        # perform the forward pass implemented in the subclass
        outputs = self._forward(inputs)
        # sum over atomic properties to generate per molecule properties
        E = self._readout(outputs["E_i"], outputs["atomic_subsystem_indices"])
        # postprocess energies: add atomic self energies,
        # and other constant factors used to optionally normalize the data range of the training dataset
        processed_energy = self._energy_postprocessing(E, inputs)
        # return energies
        return EnergyOutput(
            E=processed_energy["E"],
            raw_E=processed_energy["raw_E"],
            rescaled_E=processed_energy["rescaled_E"],
            molecular_ase=processed_energy["molecular_ase"],
        )


class SingleTopologyAlchemicalBaseNNPModel(BaseNeuralNetworkPotential):
    """
    Subclass for handling alchemical energy calculations.

    Methods
    -------
    forward(data: NNPInput) -> torch.Tensor
        Calculate the alchemical energy for a given input batch.
    """

    def forward(self, data: NNPInput):
        """
        Calculate the alchemical energy for a given input batch.

        Parameters
        ----------
        data : NNPInput
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
        self._input_checks(data)
        raise NotImplementedError

from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from loguru import logger as log

from abc import ABC, abstractmethod
import torch
from typing import NamedTuple
from modelforge.potential.utils import AtomicSelfEnergies, NeuralNetworkInput
from abc import abstractmethod, ABC
from openff.units import unit
from typing import Dict, Type


# Define NamedTuple for the outputs of Pairlist and Neighborlist forward method
class PairListOutputs(NamedTuple):
    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class EnergyOutput(NamedTuple):
    E_predict: torch.Tensor
    raw_E_predict: torch.Tensor
    rescaled_E_predict: torch.Tensor
    molecular_ase: torch.Tensor
    outputs: Dict[str, torch.Tensor]


class DatasetStatistics(NamedTuple):
    scaling_mean: float
    scaling_stddev: float
    atomic_self_energies: AtomicSelfEnergies


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
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ):
        """Initialize the neural network potential class with specified parameters."""
        from .models import Neighborlist

        super().__init__()
        self.calculate_distances_and_pairlist = Neighborlist(cutoff)
        self._dtype = None  # set at runtime
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
        self, inputs: NeuralNetworkInput
    ) -> NeuralNetworkInput:
        """
        Prepares model-specific inputs before the forward pass.

        This method should be implemented by subclasses to accommodate any
        model-specific processing of inputs.

        Parameters
        ----------
        inputs : NeuralNetworkInput
            The initial inputs to the neural network model.

        Returns
        -------
        NeuralNetworkInput
            The processed inputs, ready for the model's forward pass.
        """
        pass

    @abstractmethod
    def _forward(self, inputs: NeuralNetworkInput):
        # needs to be implemented by the subclass
        # perform the forward pass implemented in the subclass
        pass

    @property
    def dataset_statistics(self):
        return self._dataset_statistics

    @dataset_statistics.setter
    def dataset_statistics(self, key: str, value: float):

        self._dataset_statistics.key = value

    def _readout(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # readout the per molecule values
        return self.readout_module(x, index)

    def _log_batch_size(self, batch: Dict[str, torch.Tensor]):
        batch_size = int(len(batch["E"]))
        return batch_size

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
        batch_size = self._log_batch_size(batch)
        predictions = self.forward(batch)["E_predict"].flatten()
        targets = batch["E"].flatten().to(torch.float32)

        import math

        number_of_atoms = math.sqrt(len(batch["atomic_numbers"]))

        # time.sleep(1)
        loss = self.loss_function(predictions, targets) / number_of_atoms
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        from torch.nn import functional as F

        batch_size = self._log_batch_size(batch)

        predictions = self.forward(batch).flatten()
        targets = batch["E"].flatten()
        test_loss = F.mse_loss(predictions["E_predict"], targets)
        self.log(
            "test_loss", test_loss, batch_size=batch_size, on_epoch=True, prog_bar=True
        )

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        from torch.nn import functional as F

        batch_size = self._log_batch_size(batch)
        predictions = self.forward(batch)["E_predict"]
        targets = batch["E"].squeeze(1)
        val_loss = F.mse_loss(predictions, targets)
        self.log(
            "val_loss", val_loss, batch_size=batch_size, on_epoch=True, prog_bar=True
        )

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
            energies * self.dataset_statistics["scaling_stddev"]
            + self.dataset_statistics["scaling_mean"]
        )

    def _calculate_molecular_self_energy(
        self, inputs: Dict[str, torch.Tensor], number_of_molecules: int
    ) -> torch.Tensor:

        atomic_numbers = inputs["atomic_numbers"]
        atomic_subsystem_indices = inputs["atomic_subsystem_indices"].to(
            dtype=torch.long, device=self.device
        )

        # atomic_number_to_energy
        atomic_self_energies = self.dataset_statistics["atomic_self_energies"]
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

    def _energy_postprocessing(self, properties_per_molecule, inputs):

        # first, resale the energies
        processed_energy = {}
        processed_energy["_raw_E_predict"] = properties_per_molecule.clone().detach()
        properties_per_molecule = self._rescale_energy(properties_per_molecule)
        processed_energy["_rescaled_E_predict"] = (
            properties_per_molecule.clone().detach()
        )
        # then, calculate the molecular self energy
        molecular_ase = self._calculate_molecular_self_energy(
            inputs, properties_per_molecule.numel()
        )
        processed_energy["_molecular_ase"] = molecular_ase.clone().detach()
        # add the molecular self energy to the rescaled energies
        processed_energy["E"] = properties_per_molecule + molecular_ase
        return processed_energy

    def prepare_inputs(
        self, inputs: Dict[str, torch.Tensor], only_unique_pairs: bool = True
    ):  # FIXME
        """Prepares the input tensors for passing to the model.

        Performs general input manipulation like calculating distances,
        generating the pair list, etc. Also calls the model-specific input
        preparation.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Input tensors like atomic numbers, positions etc.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs or not in the pairlist.
        Returns
        -------
        inputs : Dict[str, torch.Tensor]
            Input tensors after preparation.
        """
        # ---------------------------
        # general input manipulation
        atomic_numbers = inputs["atomic_numbers"]
        positions = inputs["positions"]
        atomic_subsystem_indices = inputs["atomic_subsystem_indices"]
        atomic_index = inputs["atomic_index"]
        number_of_atoms_in_batch = atomic_numbers.shape[0]

        r = self.calculate_distances_and_pairlist(
            positions, atomic_subsystem_indices, only_unique_pairs
        )

        inputs = {
            "pair_indices": r["pair_indices"],
            "d_ij": r["d_ij"],
            "r_ij": r["r_ij"],
            "number_of_atoms_in_batch": number_of_atoms_in_batch,
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "atomic_subsystem_indices": atomic_subsystem_indices,
            "atomic_index": atomic_index,
        }

        # ---------------------------
        # perform model specific modifications
        inputs = self._model_specific_input_preparation(inputs)

        return inputs

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

    def _input_checks(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform input checks to validate the input dictionary.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor], with distance units attached
            Inputs containing necessary data for the forward pass.
            The exact keys and shapes are implementation-dependent.

        Raises
        ------
        ValueError
            If the input dictionary is missing required keys or has invalid shapes.

        """
        from openff.units import unit

        required_keys = ["atomic_numbers", "positions", "atomic_subsystem_indices"]
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required key: {key}")

        if inputs["atomic_numbers"].dim() != 1:
            print(inputs["atomic_numbers"])
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

        if isinstance(inputs["positions"], unit.Quantity):
            inputs["positions"] = inputs["positions"].to(unit.nanometer).m

        else:
            if not self._log_message_units:
                log.warning(
                    "Could not convert positions to nanometer. Assuming positions are already in nanometer."
                )
                self._log_message_units = True

        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for forward pass in neural network potentials.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
        - 'atomic_numbers': int
            shape (nr_of_atoms_in_batch, 1), 0 indicates non-interacting atoms that will be masked
        - 'total_charge' : int
            shape (n_system)
        - 'positions': float
            shape (nr_of_atoms_in_batch, 3)

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
        inputs = self.prepare_inputs(inputs, self.only_unique_pairs)
        # perform the forward pass implemented in the subclass
        outputs = self._forward(inputs)
        # sum over atomic properties to generate per molecule properties
        E = self._readout(outputs["E_i"], outputs["atomic_subsystem_indices"])
        # postprocess energies: add atomic self energies,
        # and other constant factors used to optionally normalize the data range of the training dataset
        processed_energy = self._energy_postprocessing(E, inputs)
        # return energies
        return {
            "E_predict": processed_energy["E"],
            "_raw_E_predict": processed_energy["_raw_E_predict"],
            "_rescaled_E_predict": processed_energy["_rescaled_E_predict"],
            "_molecular_ase": processed_energy["_molecular_ase"],
            "outputs": outputs,
        }


class SingleTopologyAlchemicalBaseNNPModel(BaseNeuralNetworkPotential):
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

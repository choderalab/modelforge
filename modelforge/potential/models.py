from typing import Dict, Tuple, Any, NamedTuple, TYPE_CHECKING, Type

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as F

from loguru import logger as log

from abc import ABC, abstractmethod
from modelforge.potential.utils import (
    AtomicSelfEnergies,
    NNPInput,
    BatchData,
)
from openff.units import unit
from torch.optim.lr_scheduler import ReduceLROnPlateau


if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics
    from modelforge.potential.painn import PaiNNNeuralNetworkData, PaiNN
    from modelforge.potential.schnet import SchnetNeuralNetworkData, SchNet
    from modelforge.potential.ani import AniNeuralNetworkData, ANI2x
    from modelforge.potential.physnet import PhysNetNeuralNetworkData, PhysNet


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


from typing import Callable, Optional, Union, Literal


class NeuralNetworkPotentialFactory:
    """
    Factory class for creating instances of neural network potentials (NNP).

    This factory allows for the creation of specific NNP instances configured for either
    training or inference purposes based on the given parameters.

    Methods
    -------
    create_nnp(use, nnp_type, nnp_parameters=None, training_parameters=None)
        Creates an instance of a specified NNP type configured for either training or inference.
    """

    @staticmethod
    def create_nnp(
        use: Literal["training", "inference"],
        nnp_type: Literal["ANI2x", "SchNet", "PaiNN", "SAKE", "PhysNet"],
        nnp_parameters: Optional[Dict[str, Union[int, float]]] = {},
        training_parameters: Dict[str, Any] = {},
    ) -> Union[Union["ANI2x", "SchNet", "PaiNN", "PhysNet"], "TrainingAdapter"]:
        """
        Creates an NNP instance of the specified type, configured either for training or inference.

        Parameters
        ----------
        use : {'training', 'inference'}
            The use case for the NNP instance.
        nnp_type : {'ANI2x', 'SchNet', 'PaiNN', 'SAKE', 'PhysNet'}
            The type of NNP to instantiate.
        nnp_parameters : dict, optional
            Parameters specific to the NNP model, by default {}.
        training_parameters : dict, optional
            Parameters for configuring the training, by default {}.

        Returns
        -------
        Union[Union[ANI2x, SchNet, PaiNN, PhysNet], TrainingAdapter]
            An instantiated NNP model.

        Raises
        ------
        ValueError
            If an unknown use case is requested.
        NotImplementedError
            If the requested NNP type is not implemented.
        """

        from modelforge.potential import _IMPLEMENTED_NNPS

        def _return_specific_version_of_nnp(use: str, nnp_class):
            if use == "training":
                nnp_instance = nnp_class(**nnp_parameters)
                trainer = TrainingAdapter(model=nnp_instance, **training_parameters)
                return trainer
            elif use == "inference":
                return nnp_class(**nnp_parameters)
            else:
                raise ValueError("Unknown NNP type requested.")

        if nnp_type == "ANI2x":
            return _return_specific_version_of_nnp(use, _IMPLEMENTED_NNPS[nnp_type])
        elif nnp_type == "SchNet":
            return _return_specific_version_of_nnp(use, _IMPLEMENTED_NNPS[nnp_type])
        elif nnp_type == "PaiNN":
            return _return_specific_version_of_nnp(use, _IMPLEMENTED_NNPS[nnp_type])
        elif nnp_type == "PhysNet":
            return _return_specific_version_of_nnp(use, _IMPLEMENTED_NNPS[nnp_type])
        else:
            raise NotImplementedError("Unknown NNP type requested.")


from modelforge.potential.utils import NeuralNetworkData


class BaseNeuralNetworkPotential(torch.nn.Module, ABC):
    """Abstract base class for neural network potentials.

    Attributes
    ----------
    cutoff : unit.Quantity
        Cutoff distance for neighbor list calculations.
    calculate_distances_and_pairlist : Neighborlist
        Module for calculating distances and pairlist with a given cutoff.
    readout_module : FromAtomToMoleculeReduction
        Module for reading out per molecule properties from atomic properties.
    """

    def __init__(
        self,
        cutoff: unit.Quantity,
    ):
        """
        Initializes the neural network potential class with specified parameters.

        Parameters
        ----------
        cutoff : openff.units.unit.Quantity
            Cutoff distance for the neighbor list calculations.
        """
        from .models import Neighborlist
        from modelforge.dataset.dataset import DatasetStatistics

        super().__init__()
        self.calculate_distances_and_pairlist = Neighborlist(cutoff)
        self._dtype: Optional[bool] = None  # set at runtime
        self._log_message_dtype = False
        self._log_message_units = False
        self._dataset_statistics = DatasetStatistics(0.0, 1.0, AtomicSelfEnergies())
        # initialize the per molecule readout module
        from .utils import FromAtomToMoleculeReduction

        self.readout_module = FromAtomToMoleculeReduction()

    @abstractmethod
    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist: PairListOutputs
    ) -> Union[
        "PhysNetNeuralNetworkData",
        "PaiNNNeuralNetworkData",
        "SchnetNeuralNetworkData",
        "AniNeuralNetworkData",
    ]:
        """
        Prepares model-specific inputs before the forward pass.

        This abstract method should be implemented by subclasses to accommodate any
        model-specific preprocessing of inputs.

        Parameters
        ----------
        data : NNPInput
            The initial inputs to the neural network model, including atomic numbers,
            positions, and other relevant data.
        pairlist : PairListOutputs
            The outputs of a pair list calculation, including pair indices, distances,
            and displacement vectors.

        Returns
        -------
        NeuralNetworkData: The processed inputs, ready for the model's forward pass.
        """
        pass

    @abstractmethod
    def _forward(
        self,
        data: Union[
            "PhysNetNeuralNetworkData",
            "PaiNNNeuralNetworkData",
            "SchnetNeuralNetworkData",
            "AniNeuralNetworkData",
        ],
    ):
        """
        Defines the forward pass of the model.

        This abstract method should be implemented by subclasses to specify the model's
        computation from inputs to outputs.

        Parameters
        ----------
        inputs : The processed input data, specific to the model's requirements.

        Returns
        -------
        The model's output as computed from the inputs.
        """
        pass

    def load_pretrained_weights(self, path: str):
        """
        Loads pretrained weights into the model from the specified path.

        Parameters
        ----------
        path : str
            The path to the file containing the pretrained weights.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()  # Set the model to evaluation mode

    @property
    def dataset_statistics(self):
        """
        Property for accessing the model's dataset statistics.

        Returns
        -------
        DatasetStatistics
            The dataset statistics associated with the model.
        """

        return self._dataset_statistics

    @dataset_statistics.setter
    def dataset_statistics(self, value: "DatasetStatistics"):
        """
        Sets the dataset statistics for the model.

        Parameters
        ----------
        value : DatasetStatistics
            The new dataset statistics to be set for the model.
        """

        from modelforge.dataset.dataset import DatasetStatistics

        if not isinstance(value, DatasetStatistics):
            raise ValueError("Value must be an instance of DatasetStatistics.")

        self._dataset_statistics = value

    def update_dataset_statistics(self, **kwargs):
        """
        Updates specific fields of the model's dataset statistics.

        Parameters
        ----------
        **kwargs
            Fields and their new values to update in the dataset statistics.
        """

        for key, value in kwargs.items():
            if hasattr(self.dataset_statistics, key):
                setattr(self.dataset_statistics, key, value)
            else:
                log.warning(f"{key} is not a valid field of DatasetStatistics.")

    def _readout(
        self, atom_specific_values: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the readout operation to generate per-molecule properties from atomic properties.

        Parameters
        ----------
        atom_specific_values : torch.Tensor
            The tensor containing atomic properties.
        index : torch.Tensor
            The tensor indicating the molecule to which each atom belongs.

        Returns
        -------
        torch.Tensor
            The tensor containing per-molecule properties.
        """
        return self.readout_module(atom_specific_values, index)

    def _rescale_energy(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Rescales energies using the dataset statistics.

        Parameters
        ----------
        energies : torch.Tensor
            The tensor of energies to be rescaled.

        Returns
        -------
        torch.Tensor
            The rescaled energies.
        """

        return (
            energies * self.dataset_statistics.scaling_stddev
            + self.dataset_statistics.scaling_mean
        )

    def _calculate_molecular_self_energy(
        self, data: NeuralNetworkData, number_of_molecules: int
    ) -> torch.Tensor:
        """
        Calculates the molecular self energy.

        Parameters
        ----------
        data : NNPInput
            The input data for the model, including atomic numbers and subsystem indices.
        number_of_molecules : int
            The number of molecules in the batch.

        Returns
        -------
        torch.Tensor
            The tensor containing the molecular self energy for each molecule.
        """

        atomic_numbers = data.atomic_numbers
        atomic_subsystem_indices = data.atomic_subsystem_indices.to(
            dtype=torch.long, device=atomic_numbers.device
        )

        # atomic_number_to_energy
        atomic_self_energies = self.dataset_statistics.atomic_self_energies
        ase_tensor_for_indexing = atomic_self_energies.ase_tensor_for_indexing.to(
            device=atomic_numbers.device
        )

        # first, we need to use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        # then, we use the atomic_subsystem_indices to scatter add the atomic self
        # energies in the ase_tensor to generate the molecular self energies
        ase_tensor_zeros = torch.zeros((number_of_molecules,)).to(
            device=atomic_numbers.device
        )
        ase_tensor = ase_tensor_zeros.scatter_add(
            0, atomic_subsystem_indices, ase_tensor
        )

        return ase_tensor

    def _energy_postprocessing(
        self, properties_per_molecule: torch.Tensor, inputs: NeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocesses the energies by rescaling and adding molecular self energy.

        Parameters
        ----------
        properties_per_molecule : The properties computed per molecule.
        inputs : The original input data to the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The dictionary containing the postprocessed energy tensors.
        """

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
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating distances
        and generating the pair list. It also calls the model-specific input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers, positions,
            and other necessary information.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by default True.

        Returns
        -------
        The processed input data, ready for the model's forward pass.
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
        """
        Sets the data type for the model based on its parameters.

        Ensures consistency in data types across the model's parameters.
        """

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
        Performs input validation checks.

        Ensures the input data conforms to expected shapes and types.

        Parameters
        ----------
        data : NNPInput
            The input data to be validated.

        Raises
        ------
        ValueError
            If the input data does not meet the expected criteria.
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
        Defines the forward pass of the neural network potential.

        Parameters
        ----------
        data : NNPInput
            The input data for the model, containing atomic numbers, positions, and other relevant fields.

        Returns
        -------
        EnergyOutput
            The calculated energies and other properties from the forward pass.
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
        E = self._readout(
            atom_specific_values=outputs["E_i"],
            index=outputs["atomic_subsystem_indices"],
        )
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


class TrainingAdapter(pl.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.

    This class wraps around the base neural network potential model, facilitating training
    and validation steps, optimization, and logging.

    Attributes
    ----------
    base_model : Union[ANI2x, SchNet, PaiNN, PhysNet]
        The underlying neural network potential model.
    loss_function : torch.nn.modules.loss._Loss
        Loss function used during training.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    learning_rate : float
        Learning rate for the optimizer.
    """

    def __init__(
        self,
        model: Union["ANI2x", "SchNet", "PaiNN", "PhysNet"],
        loss: Callable = F.mse_loss,
        optimizer: Any = torch.optim.Adam,
        lr: float = 1e-3,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        model : Union[ANI2x, SchNet, PaiNN, PhysNet]
            The neural network potential model to be trained.
        optimizer : Type[torch.optim.Optimizer], optional
            The optimizer class to use for training, by default torch.optim.Adam.
        lr : float, optional
            The learning rate for the optimizer, by default 1e-3.
        """
        from typing import List

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.learning_rate = lr
        self.eval_loss: List[torch.Tensor] = []

    def compute_loss(
        self,
        energies: Dict[str, torch.Tensor],
        forces: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Computes the combined loss from energies and forces.

        """
        # Compute MSE of energies
        L_E = F.mse_loss(energies["E_predcit"], energies["E_true"])
        if forces:
            # Assuming forces_true and forces_predict are already negative gradients of the potential energy w.r.t positions
            L_F = F.mse_loss(forces["F_predcit"], forces["F_true"])
        else:
            L_F = torch.tensor(0.0)

        return L_E + L_F

    def _get_forces(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """
        Extracts and computes the forces from a given batch during training or evaluation.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Tuple[str, torch.Tensor]
            The true forces from the dataset and the predicted forces by the model.
        """
        nnp_input = batch.nnp_input
        F_true = batch.metadata.F.to(torch.float32)
        E_predict = self.model.forward(nnp_input).E
        F_predict = -torch.autograd.grad(
            E_predict.sum(), nnp_input.positions, create_graph=False, retain_graph=False
        )[0]

        return {"F_true": F_true, "F_predict": F_predict}

    def _get_energies(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """
        Extracts and computes the energies from a given batch during training or evaluation.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Tuple[str, torch.Tensor]
            The true energies from the dataset and the predicted energies by the model.
        """
        nnp_input = batch.nnp_input
        E_true = batch.metadata.E.to(torch.float32).squeeze(1)
        self.batch_size = self._log_batch_size(E_true)
        E_predict = self.model.forward(nnp_input).E
        return {"E_true": E_true, "E_predict": E_predict}

    def config_prior(self):

        if hasattr(self.model, "_config_prior"):
            return self.model._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _log_batch_size(self, y: torch.Tensor) -> int:
        """
        Logs the size of the batch and returns it. Useful for logging and debugging.

        Parameters
        ----------
        y : torch.Tensor
            The tensor containing the target values of the batch.

        Returns
        -------
        int
            The size of the batch.
        """
        batch_size = int(y.numel())
        return batch_size

    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        """
        Performs a training step using the given batch.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for the training.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss tensor computed for the current training step.
        """
        energies = self._get_energies(batch)
        if hasattr(batch.metadata, "F"):
            forces = self._get_forces(batch)
        else:
            forces = None
        loss = self.compute_loss(energies, forces)

        self.log(
            "ptl/train_loss",
            loss,
            on_step=True,
            batch_size=self.batch_size,
        )

        return loss

    def test_step(self, batch: BatchData, batch_idx: int) -> None:
        """
        Executes a test step using the given batch of data.

        This method is called automatically during the test loop of the training process. It computes
        the loss on a batch of test data and logs the results for analysis.

        Parameters
        ----------
        batch : BatchData
            The batch of data to test the model on.
        batch_idx : int
            The index of the batch within the test dataset.

        Returns
        -------
        None
            The results are logged and not directly returned.
        """

        energies = self._get_energies(batch)

        test_loss = F.l1_loss(energies["E_true"], energies["E_predict"])
        self.log(
            "ptl/test_loss",
            test_loss,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
        """
        Executes a single validation step.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for validation.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss tensor computed for the current validation step.
        """

        from torch.nn import functional as F

        energies = self._get_energies(batch)

        val_loss = F.l1_loss(energies["E_true"], energies["E_predict"])
        self.log(
            "val_loss",
            val_loss,
            batch_size=self.batch_size,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.eval_loss.append(val_loss.detach())
        return val_loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.eval_loss.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the model's optimizers (and optionally schedulers).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the learning rate scheduler
            to be used within the PyTorch Lightning training process.
        """

        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=20, verbose=True
            ),
            "monitor": "val_loss",  # Name of the metric to monitor
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_trainer(self):
        """
        Sets up and returns a PyTorch Lightning Trainer instance with configured logger and callbacks.

        The trainer is configured with TensorBoard logging and an EarlyStopping callback to halt
        the training process when the validation loss stops improving.

        Returns
        -------
        Trainer
            The configured PyTorch Lightning Trainer instance.
        """

        from pytorch_lightning.loggers import TensorBoardLogger
        from lightning import Trainer
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping

        # set up tensor board logger
        logger = TensorBoardLogger("tb_logs", name="training")
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0.05, patience=20, verbose=True
        )

        return Trainer(
            max_epochs=10_000,
            num_nodes=1,
            devices="auto",
            accelerator="auto",
            logger=logger,  # Add the logger here
            callbacks=[early_stopping],
        )

    def train_func(self):
        """
        Defines the training function to be used with Ray for distributed training.

        This function configures a PyTorch Lightning trainer with the Ray Distributed Data Parallel
        (DDP) strategy for efficient distributed training. The training process utilizes a custom
        training loop and environment setup provided by Ray.

        Note: This function should be passed to a Ray Trainer or directly used with Ray tasks.
        """

        from ray.train.lightning import (
            RayDDPStrategy,
            RayLightningEnvironment,
            RayTrainReportCallback,
            prepare_trainer,
        )

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(self, self.train_dataloader, self.val_dataloader)

    def get_ray_trainer(self, number_of_workers: int = 2, use_gpu: bool = False):
        """
        Initializes and returns a Ray Trainer for distributed training.

        Configures a Ray Trainer with a specified number of workers and GPU usage settings. This trainer
        is prepared for distributed training using Ray, with support for checkpointing.

        Parameters
        ----------
        number_of_workers : int, optional
            The number of distributed workers to use, by default 2.
        use_gpu : bool, optional
            Specifies whether to use GPUs for training, by default False.

        Returns
        -------
        Ray Trainer
            The configured Ray Trainer for distributed training.
        """

        from ray.train import RunConfig, ScalingConfig, CheckpointConfig

        scaling_config = ScalingConfig(
            num_workers=number_of_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 1, "GPU": 1} if use_gpu else {"CPU": 1},
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="ptl/val_loss",
                checkpoint_score_order="min",
            ),
        )
        from ray.train.torch import TorchTrainer

        # Define a TorchTrainer without hyper-parameters for Tuner
        ray_trainer = TorchTrainer(
            self.train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        return ray_trainer

    def tune_with_ray(
        self,
        train_dataloader,
        val_dataloader,
        number_of_epochs: int = 5,
        number_of_samples: int = 10,
        number_of_ray_workers: int = 2,
        train_on_gpu: bool = False,
    ):
        """
        Performs hyperparameter tuning using Ray Tune.

        This method sets up and starts a Ray Tune hyperparameter tuning session, utilizing the ASHA scheduler
        for efficient trial scheduling and early stopping.

        Parameters
        ----------
        train_dataloader : DataLoader
            The DataLoader for training data.
        val_dataloader : DataLoader
            The DataLoader for validation data.
        number_of_epochs : int, optional
            The maximum number of epochs for training, by default 5.
        number_of_samples : int, optional
            The number of samples (trial runs) to perform, by default 10.
        number_of_ray_workers : int, optional
            The number of Ray workers to use for distributed training, by default 2.
        use_gpu : bool, optional
            Whether to use GPUs for training, by default False.

        Returns
        -------
        Tune experiment analysis object
            The result of the hyperparameter tuning session, containing performance metrics and the best hyperparameters found.
        """

        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        ray_trainer = self.get_ray_trainer(
            number_of_workers=number_of_ray_workers, use_gpu=train_on_gpu
        )
        scheduler = ASHAScheduler(
            max_t=number_of_epochs, grace_period=1, reduction_factor=2
        )

        tune_config = tune.TuneConfig(
            metric="ptl/val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=number_of_samples,
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": self.config_prior()},
            tune_config=tune_config,
        )
        return tuner.fit()


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

"""
This module contains classes and functions for training neural network potentials using PyTorch Lightning.
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, Union, Dict, Type, Optional, List
import torch
from loguru import logger as log
from modelforge.dataset.dataset import BatchData, NNPInput
import torchmetrics
from torch import nn
from abc import ABC, abstractmethod
from modelforge.dataset.dataset import DatasetParameters
from modelforge.potential.parameters import (
    ANI2xParameters,
    PhysNetParameters,
    SchNetParameters,
    PaiNNParameters,
    SAKEParameters,
    TensorNetParameters,
)
from lightning import Trainer
import lightning.pytorch as pL
from modelforge.dataset.dataset import DataModule

__all__ = [
    "Error",
    "FromPerAtomToPerMoleculeMeanSquaredError",
    "Loss",
    "LossFactory",
    "PerMoleculeMeanSquaredError",
    "ModelTrainer",
    "create_error_metrics",
    "ModelTrainer",
]


class Error(nn.Module, ABC):
    """
    Class representing the error calculation for predicted and true values.

    Methods:
        calculate_error(predicted: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
            Calculates the error between the predicted and true values.

        scale_by_number_of_atoms(error, atomic_subsystem_counts) -> torch.Tensor:
            Scales the error by the number of atoms in the atomic subsystems.
    """

    @abstractmethod
    def calculate_error(
        self, predicted: torch.Tensor, true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the error between the predicted and true values
        """
        raise NotImplementedError

    @staticmethod
    def calculate_squared_error(
        predicted_tensor: torch.Tensor, reference_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the squared error between the predicted and true values.

        Parameters
        ----------
        predicted_tensor : torch.Tensor
            The predicted values.
        reference_tensor : torch.Tensor
            The reference values provided by the dataset.

        Returns
        -------
        torch.Tensor
            The calculated error.
        """
        return (predicted_tensor - reference_tensor).pow(2).sum(dim=1, keepdim=True)

    @staticmethod
    def scale_by_number_of_atoms(error, atomic_subsystem_counts) -> torch.Tensor:
        """
        Scales the error by the number of atoms in the atomic subsystems.

        Parameters
        ----------
        error : torch.Tensor
            The error to be scaled.
        atomic_subsystem_counts : torch.Tensor
            The number of atoms in the atomic subsystems.

        Returns
        -------
        torch.Tensor
            The scaled error.
        """
        # divide by number of atoms
        scaled_by_number_of_atoms = error / atomic_subsystem_counts.unsqueeze(
            1
        )  # FIXME: ensure that all per-atom properties have dimension (N, 1)
        return scaled_by_number_of_atoms


class FromPerAtomToPerMoleculeMeanSquaredError(Error):
    """
    Calculates the per-atom error and aggregates it to per-molecule mean squared error.
    """

    def __init__(self):
        """
        Initializes the PerAtomToPerMoleculeError class.
        """
        super().__init__()

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the per-atom error.
        """
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)

    def forward(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
        batch: "NNPInput",
    ) -> torch.Tensor:
        """
        Computes the per-atom error and aggregates it to per-molecule mean squared error.

        Parameters
        ----------
        per_atom_prediction : torch.Tensor
            The predicted values.
        per_atom_reference : torch.Tensor
            The reference values provided by the dataset.
        batch : NNPInput
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The aggregated per-molecule error.
        """

        # squared error
        per_atom_squared_error = self.calculate_error(
            per_atom_prediction, per_atom_reference
        )

        per_molecule_squared_error = torch.zeros_like(
            batch.metadata.E, dtype=per_atom_squared_error.dtype
        )
        # Aggregate error per molecule

        per_molecule_squared_error.scatter_add_(
            0,
            batch.nnp_input.atomic_subsystem_indices.long().unsqueeze(1),
            per_atom_squared_error,
        )
        # divide by number of atoms
        per_molecule_square_error_scaled = self.scale_by_number_of_atoms(
            per_molecule_squared_error, batch.metadata.atomic_subsystem_counts
        )
        # return the average
        return torch.mean(per_molecule_square_error_scaled)


class PerMoleculeMeanSquaredError(Error):
    """
    Calculates the per-molecule mean squared error.

    """

    def __init__(self):
        """
        Initializes the PerMoleculeMeanSquaredError class.
        """

        super().__init__()

    def forward(
        self,
        per_molecule_prediction: torch.Tensor,
        per_molecule_reference: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """
        Computes the per-molecule mean squared error.

        Parameters
        ----------
        per_molecule_prediction : torch.Tensor
            The predicted values.
        per_molecule_reference : torch.Tensor
            The true values.
        batch : Any
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The mean per-molecule error.
        """

        per_molecule_squared_error = self.calculate_error(
            per_molecule_prediction, per_molecule_reference
        )
        per_molecule_square_error_scaled = self.scale_by_number_of_atoms(
            per_molecule_squared_error, batch.metadata.atomic_subsystem_counts
        )

        # return the average
        return torch.mean(per_molecule_square_error_scaled)

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the per-atom error.
        """
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)


class Loss(nn.Module):
    """
    Calculates the combined loss for energy and force predictions.

    Attributes
    ----------
    loss_property : List[str]
        List of properties to include in the loss calculation.
    weight : Dict[str, float]
        Dictionary containing the weights for each property in the loss calculation.
    loss : nn.ModuleDict
        Module dictionary containing the loss functions for each property.
    """

    _SUPPORTED_PROPERTIES = ["per_molecule_energy", "per_atom_force"]

    def __init__(self, loss_porperty: List[str], weight: Dict[str, float]):
        """
        Initializes the Loss class.

        Parameters
        ----------
        loss_property : List[str]
            List of properties to include in the loss calculation.
        weight : Dict[str, float]
            Dictionary containing the weights for each property in the loss calculation.

        Raises
        ------
        NotImplementedError
            If an unsupported loss type is specified.
        """
        super().__init__()
        from torch.nn import ModuleDict

        self.loss_property = loss_porperty
        self.weight = weight

        self.loss = ModuleDict()

        for prop, w in weight.items():
            if prop in self._SUPPORTED_PROPERTIES:
                if prop == "per_atom_force":
                    self.loss[prop] = FromPerAtomToPerMoleculeMeanSquaredError()
                elif prop == "per_molecule_energy":
                    self.loss[prop] = PerMoleculeMeanSquaredError()
                self.register_buffer(prop, torch.tensor(w))
            else:
                raise NotImplementedError(f"Loss type {prop} not implemented.")

    def forward(self, predict_target: Dict[str, torch.Tensor], batch):
        """
        Calculates the combined loss for the specified properties.

        Parameters
        ----------
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for energy and per_atom_force.
        batch : Any
            The batch data containing metadata and input information.

        Returns
        -------
        Dict{str, torch.Tensor]
            Individual loss terms and the combined, total loss.
        """
        # save the loss as a dictionary
        loss_dict = {}
        # accumulate loss
        loss = torch.tensor(
            [0.0], dtype=batch.metadata.E.dtype, device=batch.metadata.E.device
        )
        # iterate over loss properties
        for prop in self.loss_property:
            # calculate loss per property
            loss_ = self.weight[prop] * self.loss[prop](
                predict_target[f"{prop}_predict"], predict_target[f"{prop}_true"], batch
            )
            # add total loss
            loss = loss + loss_
            # save loss
            loss_dict[f"{prop}/mse"] = loss_

        # add total loss to results dict and return
        loss_dict["total_loss"] = loss

        return loss_dict


class LossFactory(object):
    """
    Factory class to create different types of loss functions.
    """

    @staticmethod
    def create_loss(loss_property: List[str], weight: Dict[str, float]) -> Type[Loss]:
        """
        Creates an instance of the specified loss type.

        Parameters
        ----------
        loss_property : List[str]
            List of properties to include in the loss calculation.
        weight : Dict[str, float]
            Dictionary containing the weights for each property in the loss calculation.
        Returns
        -------
        Loss
            An instance of the specified loss function.
        """

        return Loss(loss_property, weight)


from torch.optim import Optimizer


from torch.nn import ModuleDict


def create_error_metrics(loss_properties: List[str]) -> ModuleDict:
    """
    Creates a ModuleDict of MetricCollections for the given loss properties.

    Parameters
    ----------
    loss_properties : List[str]
        List of loss properties for which to create the metrics.

    Returns
    -------
    ModuleDict
        A dictionary where keys are loss properties and values are MetricCollections.
    """
    from torchmetrics.regression import (
        MeanAbsoluteError,
        MeanSquaredError,
    )
    from torchmetrics import MetricCollection

    return ModuleDict(
        {
            prop: MetricCollection(
                [MeanAbsoluteError(), MeanSquaredError(squared=False)]
            )
            for prop in loss_properties
        }
    )


from modelforge.train.parameters import RuntimeParameters, TrainingParameters


class CalculateProperties(torch.nn.Module):

    def __init__(self, model):
        self.model = model

    def _get_forces(
        self, batch: "BatchData", energies: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        energies : dict
            Dictionary containing predicted energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true forces from the dataset and the predicted forces by the model.
        """
        nnp_input = batch.nnp_input
        per_atom_force_true = batch.metadata.F.to(torch.float32)

        if per_atom_force_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        per_molecule_energy_predict = energies["per_molecule_energy_predict"]

        # Ensure E_predict and nnp_input.positions require gradients and are on the same device
        if not per_molecule_energy_predict.requires_grad:
            per_molecule_energy_predict.requires_grad = True
        if not nnp_input.positions.requires_grad:
            nnp_input.positions.requires_grad = True

        # Compute the gradient (forces) from the predicted energies
        grad = torch.autograd.grad(
            per_molecule_energy_predict.sum(),
            nnp_input.positions,
            create_graph=False,
            retain_graph=True,
        )[0]
        per_atom_force_predict = -1 * grad  # Forces are the negative gradient of energy

        return {
            "per_atom_force_true": per_atom_force_true,
            "per_atom_force_predict": per_atom_force_predict,
        }

    def _get_energies(self, batch: "BatchData") -> Dict[str, torch.Tensor]:
        """
        Computes the energies from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true energies from the dataset and the predicted energies by the model.
        """
        nnp_input = batch.nnp_input
        per_molecule_energy_true = batch.metadata.E.to(torch.float32)
        per_molecule_energy_predict = self.model.forward(nnp_input)[
            "per_molecule_energy"
        ].unsqueeze(
            1
        )  # FIXME: ensure that all per-molecule properties have dimension (N, 1)
        assert per_molecule_energy_true.shape == per_molecule_energy_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{per_molecule_energy_true.shape} != {per_molecule_energy_predict.shape}"
        )
        return {
            "per_molecule_energy_true": per_molecule_energy_true,
            "per_molecule_energy_predict": per_molecule_energy_predict,
        }

    def forward(self, batch: "BatchData") -> Dict[str, torch.Tensor]:
        """
        Computes the energies and forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the model.
        """
        energies = self._get_energies(batch)
        forces = self._get_forces(batch, energies)
        return {**energies, **forces}


class TrainingAdapter(pL.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        potential_parameter: Union[
            ANI2xParameters,
            SAKEParameters,
            SchNetParameters,
            PhysNetParameters,
            PaiNNParameters,
            TensorNetParameters,
        ],
        dataset_statistic: Dict[str, float],
        training_parameter: TrainingParameters,
        model_seed: Optional[int] = None,
        optimizer: Type[Optimizer],
        verbose: bool = False,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        model_parameter : Dict[str, Any]
            The parameters for the neural network potential model.
        lr_scheduler : Dict[str, Union[str, int, float]]
            The configuration for the learning rate scheduler.
        lr : float
            The learning rate for the optimizer.
        loss_module : Loss, optional
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        """

        from modelforge.potential import _Implemented_NNPs

        super().__init__()
        self.save_hyperparameters()
        self.training_parameter = training_parameter

        # Get requested model class
        nnp_class = _Implemented_NNPs.get_neural_network_class(
            potential_parameter.potential_name
        )
        self.model = nnp_class(
            **potential_parameter.core_parameter.model_dump(),
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=potential_parameter.postprocessing_parameter.model_dump(),
            model_seed=model_seed,
        )

        self.calculate_prediction = CalculateProperties(self.model)
        self.optimizer = training_parameter.optimizer
        self.learning_rate = training_parameter.lr
        self.lr_scheduler = training_parameter.lr_scheduler

        # verbose output, only True if requested
        if verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # initialize loss
        self.loss = LossFactory.create_loss(
            **training_parameter.loss_parameter.model_dump()
        )

        # Assign the created error metrics to the respective attributes
        self.test_error = create_error_metrics(loss_parameter["loss_property"])
        self.val_error = create_error_metrics(loss_parameter["loss_property"])
        self.train_error = create_error_metrics(loss_parameter["loss_property"])

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.model, "_config_prior"):
            return self.model._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _update_metrics(
        self,
        error_dict: Dict[str, torchmetrics.MetricCollection],
        predict_target: Dict[str, torch.Tensor],
    ):
        """
        Updates the provided metric collections with the predicted and true targets.

        Parameters
        ----------
        error_dict : dict
            Dictionary containing metric collections for energy and force.
        predict_target : dict
            Dictionary containing predicted and true values for energy and force.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing updated metrics.
        """

        for property, metrics in error_dict.items():
            for _, error_log in metrics.items():
                error_log(
                    predict_target[f"{property}_predict"].detach(),
                    predict_target[f"{property}_true"].detach(),
                )

    def training_step(self, batch: "BatchData", batch_idx: int) -> torch.Tensor:
        """
        Training step to compute the MSE loss for a given batch.

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

        # calculate energy and forces
        predict_target = self.calculate_predictions(batch)

        # calculate the loss
        loss_dict = self.loss(predict_target, batch)

        # Update and log training error
        self._update_metrics(self.train_error, predict_target)

        # log the loss (this includes the individual contributions that the loss contains)
        for key, loss in loss_dict.items():
            self.log(
                f"loss/{key}",
                torch.mean(loss),
                on_step=False,
                prog_bar=True,
                on_epoch=True,
                batch_size=1,
            )  # batch size is 1 because the mean of the batch is logged

        return loss_dict["total_loss"]

    @torch.enable_grad()
    def validation_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Validation step to compute the RMSE/MAE across epochs.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for validation.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """

        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        predict_target = self._get_predictions(batch)
        # calculate the loss
        loss = self.loss(predict_target, batch)
        # log the loss
        self._update_metrics(self.val_error, predict_target)

    @torch.enable_grad()
    def test_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Test step to compute the RMSE loss for a given batch.

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
        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        predict_target = self._get_predictions(batch)
        # Update and log metrics
        self._update_metrics(self.test_error, predict_target)

    def on_test_epoch_end(self):
        """
        Operations to perform at the end of the test set pass.
        """
        self._log_on_epoch(log_mode="test")

    def on_train_epoch_end(self):
        """
        Operations to perform at the end of each training epoch.

        Logs histograms of weights and biases, and learning rate.
        Also, resets validation loss.
        """
        if self.log_histograms == True:
            for name, params in self.named_parameters():
                if params is not None:
                    self.logger.experiment.add_histogram(
                        name, params, self.current_epoch
                    )
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}.grad", params.grad, self.current_epoch
                    )

        sch = self.lr_schedulers()
        try:
            self.log("lr", sch.get_last_lr()[0], on_epoch=True, prog_bar=True)
        except AttributeError:
            pass

        self._log_on_epoch()

    def _log_on_epoch(self, log_mode: str = "train"):
        # convert long names to shorter versions
        conv = {
            "MeanAbsoluteError": "mae",
            "MeanSquaredError": "rmse",
        }  # NOTE: MeanSquaredError(squared=False) is RMSE

        # Log all accumulated metrics for train and val phases
        if log_mode == "train":
            errors = [
                ("train", self.train_error),
                ("val", self.val_error),
            ]
        elif log_mode == "test":
            errors = [
                ("test", self.test_error),
            ]
        else:
            raise RuntimeError(f"Unrecognized mode: {log_mode}")

        for phase, error_dict in errors:
            # skip if log_on_training_step is not requested
            if phase == "train" and not self.log_on_training_step:
                continue

            metrics = {}
            for property, metrics_dict in error_dict.items():
                for name, metric in metrics_dict.items():
                    name = f"{phase}/{property}/{conv[name]}"
                    metrics[name] = metric.compute()
                    metric.reset()
            # log dict, print val metrics to console
            self.log_dict(metrics, on_epoch=True, prog_bar=(phase == "val"))

    def configure_optimizers(self):
        """
        Configures the model's optimizers (and optionally schedulers).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the learning rate scheduler
            to be used within the PyTorch Lightning training process.
        """

        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

        lr_scheduler = self.lr_scheduler.copy()
        interval = lr_scheduler.pop("interval")
        frequency = lr_scheduler.pop("frequency")
        monitor = lr_scheduler.pop("monitor")

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            **lr_scheduler,
        )

        lr_scheduler = {
            "scheduler": lr_scheduler,
            "monitor": monitor,  # Name of the metric to monitor
            "interval": interval,
            "frequency": frequency,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ModelTrainer:
    """
    Class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        dataset_parameter: DatasetParameters,
        potential_parameter: Union[
            ANI2xParameters,
            SAKEParameters,
            SchNetParameters,
            PhysNetParameters,
            PaiNNParameters,
            TensorNetParameters,
        ],
        training_parameter: TrainingParameters,
        runtime_parameter: RuntimeParameters,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        model_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        dataset_config : DatasetParameters
            Parameters for the dataset.
        potential_parameter : Union[ANI2xParameters, SAKEParameters, SchNetParameters, PhysNetParameters, PaiNNParameters, TensorNetParameters]
            Parameters for the potential model.
        training_config : TrainingParameters
            Parameters for the training process.
        runtime_config : RuntimeParameters
            Parameters for runtime configuration.
        lr_scheduler : Dict[str, Union[str, int, float]]
            The configuration for the learning rate scheduler.
        lr : float
            The learning rate for the optimizer.
        loss_parameter : Dict[str, Any]
            Configuration for the loss function.
        datamodule : DataModule
            The DataModule for loading datasets.
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        verbose : bool, optional
            If True, enables verbose logging, by default False.
        """

        super().__init__()
        self.save_hyperparameters()

        self.dataset_parameter = dataset_parameter
        self.potential_parameter = potential_parameter
        self.training_parameter = training_parameter
        self.runtime_parameter = runtime_parameter

        self.datamodule = self.setup_datamodule()
        self.dataset_statistic = self.read_dataset_statistics()
        self.experiment_logger = self.setup_logger()

        self.model = self.setup_model(model_seed)
        self.callbacks = self.setup_callbacks()

        self.trainer = self.setup_trainer()
        self.optimizer = optimizer
        self.learning_rate = self.training_parameter.lr
        self.lr_scheduler = self.training_parameter.lr_scheduler

        # Verbose output
        if verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # Initialize loss
        self.loss = LossFactory.create_loss(
            **self.training_parameter.loss_parameter.model_dump()
        )

        # Assign the created error metrics to the respective attributes
        self.test_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )
        self.val_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )
        self.train_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )

        # Setup callbacks and Trainer
        self.callbacks = self.setup_callbacks()
        self.trainer = self.setup_trainer()

    def read_dataset_statistics(self) -> Dict[str, float]:
        """
        Read and log dataset statistics.

        Returns
        -------
        Dict[str, float]
            The dataset statistics.
        """
        import toml

        dataset_statistic = toml.load(self.datamodule.dataset_statistic_filename)
        log.info(
            f"Setting per_atom_energy_mean and per_atom_energy_stddev for {self.potential_parameter.potential_name}"
        )
        log.info(
            f"per_atom_energy_mean: {dataset_statistic['training_dataset_statistics']['per_atom_energy_mean']}"
        )
        log.info(
            f"per_atom_energy_stddev: {dataset_statistic['training_dataset_statistics']['per_atom_energy_stddev']}"
        )
        return dataset_statistic

    def setup_datamodule(self) -> DataModule:
        """
        Set up the DataModule for the dataset.

        Returns
        -------
        DataModule
            Configured DataModule instance.
        """
        from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES
        from modelforge.dataset.dataset import DataModule

        dm = DataModule(
            name=self.dataset_parameter.dataset_name,
            batch_size=self.training_parameter.batch_size,
            remove_self_energies=self.training_parameter.remove_self_energies,
            version_select=self.dataset_parameter.version_select,
            local_cache_dir=self.runtime_parameter.local_cache_dir,
            splitting_strategy=REGISTERED_SPLITTING_STRATEGIES[
                self.training_parameter.splitting_strategy.name
            ](
                seed=self.training_parameter.splitting_strategy.seed,
                split=self.training_parameter.splitting_strategy.data_split,
            ),
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def setup_model(self, model_seed: Optional[int] = None) -> nn.Module:
        """
        Set up the model for training.
        Parameters:
        -----------
        model_seed : int, optional
            Seed to be used to initialize the model, default is None.

        Returns
        -------
        nn.Module
            Configured model instance.
        """
        # Initialize model
        return TrainingAdapter(
            self.potential_parameter,
            dataset_statistic=self.dataset_statistic,
            postprocessing_parameter=self.potential_parameter.postprocessing_parameter.model_dump(),
            model_seed=model_seed,
        )

    def setup_logger(self) -> pL.loggers.Logger:
        """
        Set up the experiment logger based on the configuration.

        Returns
        -------
        pL.loggers.Logger
            Configured logger instance.
        """
        if self.training_parameter.experiment_logger.logger_name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            logger = TensorBoardLogger(
                save_dir=str(
                    self.training_parameter.experiment_logger.tensorboard_configuration.save_dir
                ),  # FIXME: same variable for all logger, maybe we can use a varable not bound to a logger for this?
                name=self._replace_placeholder_in_experimental_name(
                    self.runtime_parameter.experiment_name
                ),
            )
        elif self.training_parameter.experiment_logger.logger_name == "wandb":
            from modelforge.utils.io import check_import

            check_import("wandb")
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                save_dir=str(
                    self.training_parameter.experiment_logger.wandb_configuration.save_dir
                ),
                log_model=str(
                    self.training_parameter.experiment_logger.wandb_configuration.log_model
                ),
                project=self.training_parameter.experiment_logger.wandb_configuration.project,
                group=self.training_parameter.experiment_logger.wandb_configuration.group,
                job_type=self.training_parameter.experiment_logger.wandb_configuration.job_type,
                tags=self._add_tags(
                    self.training_parameter.experiment_logger.wandb_configuration.tags
                ),
                notes=self.training_parameter.experiment_logger.wandb_configuration.notes,
                name=self._replace_placeholder_in_experimental_name(
                    self.runtime_parameter.experiment_name
                ),
            )
        return logger

    def setup_callbacks(self) -> List[Any]:
        """
        Set up the callbacks for the trainer.

        Returns
        -------
        List[Any]
            List of configured callbacks.
        """
        from lightning.pytorch.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            StochasticWeightAveraging,
        )

        callbacks = []
        if self.training_parameter.stochastic_weight_averaging is not None:
            callbacks.append(
                StochasticWeightAveraging(
                    **self.training_parameter.stochastic_weight_averaging.model_dump()
                )
            )

        if self.training_parameter.early_stopping is not None:
            callbacks.append(
                EarlyStopping(**self.training_parameter.early_stopping.model_dump())
            )

        checkpoint_filename = (
            f"best_{self.potential_parameter.potential_name}-{self.dataset_parameter.dataset_name}"
            + "-{epoch:02d}-{val_loss:.2f}"
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor=self.training_parameter.monitor,
            filename=checkpoint_filename,
        )
        callbacks.append(checkpoint_callback)
        return callbacks

    def setup_trainer(self) -> Trainer:
        """
        Set up the Trainer for training.

        Returns
        -------
        Trainer
            Configured Trainer instance.
        """
        from lightning import Trainer

        trainer = Trainer(
            max_epochs=self.training_parameter.number_of_epochs,
            num_nodes=self.runtime_parameter.number_of_nodes,
            devices=self.runtime_parameter.devices,
            accelerator=self.runtime_parameter.accelerator,
            logger=self.logger,
            callbacks=self.callbacks,
            inference_mode=False,
            num_sanity_val_steps=2,
            log_every_n_steps=self.runtime_parameter.log_every_n_steps,
        )
        return trainer

    def train_dataloader(self):
        """
        Fetch the train dataloader from the DataModule.
        """
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        """
        Fetch the validation dataloader from the DataModule.
        """
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        """
        Fetch the test dataloader from the DataModule.
        """
        return self.datamodule.test_dataloader()

    def train_potential(self) -> Trainer:
        """
        Run the training process.

        Returns
        -------
        Trainer
            The configured trainer instance.
        """
        self.trainer.fit(
            self,
            train_dataloaders=self.datamodule.train_dataloader(
                num_workers=self.dataset_parameter.num_workers,
                pin_memory=self.dataset_parameter.pin_memory,
            ),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=(
                self.runtime_parameter.checkpoint_path
                if self.runtime_parameter.checkpoint_path != "None"
                else None
            ),  # NOTE: automatically resumes training from checkpoint
        )

        self.trainer.validate(
            dataloaders=self.datamodule.val_dataloader(),
            ckpt_path="best",
            verbose=True,
        )
        self.trainer.test(
            dataloaders=self.datamodule.test_dataloader(),
            ckpt_path="best",
            verbose=True,
        )
        return self.trainer

    def _get_forces(
        self, batch: "BatchData", energies: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        energies : dict
            Dictionary containing predicted energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true forces from the dataset and the predicted forces by the model.
        """
        nnp_input = batch.nnp_input
        per_atom_force_true = batch.metadata.F.to(torch.float32)

        if per_atom_force_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        per_molecule_energy_predict = energies["per_molecule_energy_predict"]

        # Ensure E_predict and nnp_input.positions require gradients and are on the same device
        if not per_molecule_energy_predict.requires_grad:
            per_molecule_energy_predict.requires_grad = True
        if not nnp_input.positions.requires_grad:
            nnp_input.positions.requires_grad = True

        # Compute the gradient (forces) from the predicted energies
        grad = torch.autograd.grad(
            per_molecule_energy_predict.sum(),
            nnp_input.positions,
            create_graph=False,
            retain_graph=True,
        )[0]
        per_atom_force_predict = -1 * grad  # Forces are the negative gradient of energy

        return {
            "per_atom_force_true": per_atom_force_true,
            "per_atom_force_predict": per_atom_force_predict,
        }

    def _get_energies(self, batch: "BatchData") -> Dict[str, torch.Tensor]:
        """
        Computes the energies from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true energies from the dataset and the predicted energies by the model.
        """
        nnp_input = batch.nnp_input
        per_molecule_energy_true = batch.metadata.E.to(torch.float32)
        per_molecule_energy_predict = self.model.forward(nnp_input)[
            "per_molecule_energy"
        ].unsqueeze(
            1
        )  # FIXME: ensure that all per-molecule properties have dimension (N, 1)
        assert per_molecule_energy_true.shape == per_molecule_energy_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{per_molecule_energy_true.shape} != {per_molecule_energy_predict.shape}"
        )
        return {
            "per_molecule_energy_true": per_molecule_energy_true,
            "per_molecule_energy_predict": per_molecule_energy_predict,
        }

    def _get_predictions(self, batch: "BatchData") -> Dict[str, torch.Tensor]:
        """
        Computes the energies and forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the model.
        """
        energies = self._get_energies(batch)
        forces = self._get_forces(batch, energies)
        return {**energies, **forces}

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.model, "_config_prior"):
            return self.model._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _update_metrics(
        self,
        error_dict: Dict[str, torchmetrics.MetricCollection],
        predict_target: Dict[str, torch.Tensor],
    ):
        """
        Updates the provided metric collections with the predicted and true targets.

        Parameters
        ----------
        error_dict : dict
            Dictionary containing metric collections for energy and force.
        predict_target : dict
            Dictionary containing predicted and true values for energy and force.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing updated metrics.
        """

        for property, metrics in error_dict.items():
            for _, error_log in metrics.items():
                error_log(
                    predict_target[f"{property}_predict"].detach(),
                    predict_target[f"{property}_true"].detach(),
                )

    def training_step(self, batch: "BatchData", batch_idx: int) -> torch.Tensor:
        """
        Training step to compute the MSE loss for a given batch.

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

        # calculate energy and forces
        predict_target = self._get_predictions(batch)

        # calculate the loss
        loss_dict = self.loss(predict_target, batch)

        # Update and log training error
        self._update_metrics(
            self.train_error, predict_target
        )  # FIXME: use pytorchmetrics to log the loss

        # log the loss (this includes the individual contributions that the loss contains)
        for key, loss in loss_dict.items():
            self.log(
                f"loss/{key}",
                torch.mean(loss),
                on_step=False,
                prog_bar=True,
                on_epoch=True,
                batch_size=1,
            )  # batch size is 1 because the mean of the batch is logged

        return loss_dict["total_loss"]

    @torch.enable_grad()
    def validation_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Validation step to compute the RMSE/MAE across epochs.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for validation.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """

        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        predict_target = self._get_predictions(batch)
        # calculate the loss
        loss = self.loss(predict_target, batch)
        # log the loss
        self._update_metrics(self.val_error, predict_target)

    @torch.enable_grad()
    def test_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Test step to compute the RMSE loss for a given batch.

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
        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        predict_target = self._get_predictions(batch)
        # Update and log metrics
        self._update_metrics(self.test_error, predict_target)

    def on_test_epoch_end(self):
        """
        Operations to perform at the end of the test set pass.
        """
        self._log_on_epoch(log_mode="test")

    def on_train_epoch_end(self):
        """
        Operations to perform at the end of each training epoch.

        Logs histograms of weights and biases, and learning rate.
        Also, resets validation loss.
        """
        if self.log_histograms == True:
            for name, params in self.named_parameters():
                if params is not None:
                    self.logger.experiment.add_histogram(
                        name, params, self.current_epoch
                    )
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}.grad", params.grad, self.current_epoch
                    )

        sch = self.lr_schedulers()
        try:
            self.log("lr", sch.get_last_lr()[0], on_epoch=True, prog_bar=True)
        except AttributeError:
            pass

        self._log_on_epoch()

    def _log_on_epoch(self, log_mode: str = "train"):
        # convert long names to shorter versions
        conv = {
            "MeanAbsoluteError": "mae",
            "MeanSquaredError": "rmse",
        }  # NOTE: MeanSquaredError(squared=False) is RMSE

        # Log all accumulated metrics for train and val phases
        if log_mode == "train":
            errors = [
                ("train", self.train_error),
                ("val", self.val_error),
            ]
        elif log_mode == "test":
            errors = [
                ("test", self.test_error),
            ]
        else:
            raise RuntimeError(f"Unrecognized mode: {log_mode}")

        for phase, error_dict in errors:
            # skip if log_on_training_step is not requested
            if phase == "train" and not self.log_on_training_step:
                continue

            metrics = {}
            for property, metrics_dict in error_dict.items():
                for name, metric in metrics_dict.items():
                    name = f"{phase}/{property}/{conv[name]}"
                    metrics[name] = metric.compute()
                    metric.reset()
            # log dict, print val metrics to console
            self.log_dict(metrics, on_epoch=True, prog_bar=(phase == "val"))

    def configure_optimizers(self):
        """
        Configures the model's optimizers (and optionally schedulers).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the learning rate scheduler
            to be used within the PyTorch Lightning training process.
        """

        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)

        lr_scheduler = self.lr_scheduler.model_dump()
        interval = lr_scheduler.pop("interval")
        frequency = lr_scheduler.pop("frequency")
        monitor = lr_scheduler.pop("monitor")

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            **lr_scheduler,
        )

        lr_scheduler = {
            "scheduler": lr_scheduler,
            "monitor": monitor,  # Name of the metric to monitor
            "interval": interval,
            "frequency": frequency,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _replace_placeholder_in_experimental_name(self, experiment_name: str) -> str:
        """
        Replace the placeholders in the experiment name with the actual values.

        Parameters
        ----------
        experiment_name : str
            The experiment name with placeholders.

        Returns
        -------
        str
            The experiment name with the placeholders replaced.
        """
        # replace placeholders in the experiment name
        experiment_name = experiment_name.replace(
            "{potential_name}", self.potential_parameter.potential_name
        )
        experiment_name = experiment_name.replace(
            "{dataset_name}", self.dataset_parameter.dataset_name
        )
        return experiment_name


from typing import List, Optional, Union


def read_config(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = None,
):
    """
    Reads one or more TOML configuration files and loads them into the pydantic models

    Parameters
    ----------
    condensed_config_path : str, optional
        Path to the TOML configuration that contains all parameters for the dataset, potential, training, and runtime parameters.
        Any other provided configuration files will be ignored.
    training_parameter_path : str, optional
        Path to the TOML file defining the training parameters.
    dataset_parameter_path : str, optional
        Path to the TOML file defining the dataset parameters.
    potential_parameter_path_path : str, optional
        Path to the TOML file defining the potential parameters.
    runtime_parameter_path : str, optional
        Path to the TOML file defining the runtime parameters. If this is not provided, the code will attempt to use
        the runtime parameters provided as arguments.
    accelerator : str, optional
        Accelerator type to use.  If provided, this  overrides the accelerator type in the runtime_defaults configuration.
    devices : int|List[int], optional
        Device index/indices to use.  If provided, this overrides the devices in the runtime_defaults configuration.
    number_of_nodes : int, optional
        Number of nodes to use.  If provided, this overrides the number of nodes in the runtime_defaults configuration.
    experiment_name : str, optional
        Name of the experiment.  If provided, this overrides the experiment name in the runtime_defaults configuration.
    save_dir : str, optional
        Directory to save the model.  If provided, this overrides the save directory in the runtime_defaults configuration.
    local_cache_dir : str, optional
        Local cache directory.  If provided, this overrides the local cache directory in the runtime_defaults configuration.
    checkpoint_path : str, optional
        Path to the checkpoint file.  If provided, this overrides the checkpoint path in the runtime_defaults configuration.
    log_every_n_steps : int, optional
        Number of steps to log.  If provided, this overrides the log_every_n_steps in the runtime_defaults configuration.
    simulation_environment : str, optional
        Simulation environment.  If provided, this overrides the simulation environment in the runtime_defaults configuration.

    Returns
    -------
    Tuple
        Tuple containing the training, dataset, potential, and runtime parameters.

    """
    import toml

    use_runtime_variables_instead_of_toml = False
    if condensed_config_path is not None:
        config = toml.load(condensed_config_path)
        log.info(f"Reading config from : {condensed_config_path}")

        training_config_dict = config["training"]
        dataset_config_dict = config["dataset"]
        potential_config_dict = config["potential"]
        runtime_config_dict = config["runtime"]

    else:
        if training_parameter_path is None:
            raise ValueError("Training configuration not provided.")
        if dataset_parameter_path is None:
            raise ValueError("Dataset configuration not provided.")
        if potential_parameter_path_path is None:
            raise ValueError("Potential configuration not provided.")

        training_config_dict = toml.load(training_parameter_path)["training"]
        dataset_config_dict = toml.load(dataset_parameter_path)["dataset"]
        potential_config_dict = toml.load(potential_parameter_path_path)["potential"]

        # if the runtime_parameter_path is not defined, let us see if runtime variables are passed
        if runtime_parameter_path is None:
            use_runtime_variables_instead_of_toml = True
            log.info(
                "Runtime configuration not provided. The code will try to use runtime arguments."
            )
            # we can just create a dict with the runtime variables; the pydantic model will then validate them
            runtime_config_dict = {
                "save_dir": save_dir,
                "experiment_name": experiment_name,
                "local_cache_dir": local_cache_dir,
                "checkpoint_path": checkpoint_path,
                "log_every_n_steps": log_every_n_steps,
                "simulation_environment": simulation_environment,
                "accelerator": accelerator,
                "devices": devices,
                "number_of_nodes": number_of_nodes,
            }
        else:
            runtime_config_dict = toml.load(runtime_parameter_path)["runtime"]

    from modelforge.potential import _Implemented_NNP_Parameters
    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.train.parameters import TrainingParameters, RuntimeParameters

    potential_name = potential_config_dict["potential_name"]
    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )

    dataset_parameters = DatasetParameters(**dataset_config_dict)
    training_parameters = TrainingParameters(**training_config_dict)
    runtime_parameters = RuntimeParameters(**runtime_config_dict)
    potential_parameter_paths = PotentialParameters(**potential_config_dict)

    # if accelerator, devices, or number_of_nodes are provided, override the runtime_defaults parameters
    # note, since these are being set in the runtime data model, they will be validated by the model
    # if we use the runtime variables instead of the toml file, these have already been set so we can skip this step.

    if use_runtime_variables_instead_of_toml == False:
        if accelerator:
            runtime_parameters.accelerator = accelerator
            log.info(f"Using accelerator: {accelerator}")
        if devices:
            runtime_parameters.device_index = devices
            log.info(f"Using device index: {devices}")
        if number_of_nodes:
            runtime_parameters.number_of_nodes = number_of_nodes
            log.info(f"Using number of nodes: {number_of_nodes}")
        if experiment_name:
            runtime_parameters.experiment_name = experiment_name
            log.info(f"Using experiment name: {experiment_name}")
        if save_dir:
            runtime_parameters.save_dir = save_dir
            log.info(f"Using save directory: {save_dir}")
        if local_cache_dir:
            runtime_parameters.local_cache_dir = local_cache_dir
            log.info(f"Using local cache directory: {local_cache_dir}")
        if checkpoint_path:
            runtime_parameters.checkpoint_path = checkpoint_path
            log.info(f"Using checkpoint path: {checkpoint_path}")
        if log_every_n_steps:
            runtime_parameters.log_every_n_steps = log_every_n_steps
            log.info(f"Logging every {log_every_n_steps} steps.")
        if simulation_environment:
            runtime_parameters.simulation_environment = simulation_environment
            log.info(f"Using simulation environment: {simulation_environment}")

    return (
        training_parameters,
        dataset_parameters,
        potential_parameter_paths,
        runtime_parameters,
    )


def read_config_and_train(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = "PyTorch",
):
    """
    Reads one or more TOML configuration files and performs training based on the parameters.

    Parameters
    ----------
    condensed_config_path : str, optional
        Path to the TOML configuration that contains all parameters for the dataset, potential, training, and runtime parameters.
        Any other provided configuration files will be ignored.
    training_parameter_path : str, optional
        Path to the TOML file defining the training parameters.
    dataset_parameter_path : str, optional
        Path to the TOML file defining the dataset parameters.
    potential_parameter_path_path : str, optional
        Path to the TOML file defining the potential parameters.
    runtime_parameter_path : str, optional
        Path to the TOML file defining the runtime parameters. If this is not provided, the code will attempt to use
        the runtime parameters provided as arguments.
    accelerator : str, optional
        Accelerator type to use.  If provided, this  overrides the accelerator type in the runtime_defaults configuration.
    devices : int|List[int], optional
        Device index/indices to use.  If provided, this overrides the devices in the runtime_defaults configuration.
    number_of_nodes : int, optional
        Number of nodes to use.  If provided, this overrides the number of nodes in the runtime_defaults configuration.
    experiment_name : str, optional
        Name of the experiment.  If provided, this overrides the experiment name in the runtime_defaults configuration.
    save_dir : str, optional
        Directory to save the model.  If provided, this overrides the save directory in the runtime_defaults configuration.
    local_cache_dir : str, optional
        Local cache directory.  If provided, this overrides the local cache directory in the runtime_defaults configuration.
    checkpoint_path : str, optional
        Path to the checkpoint file.  If provided, this overrides the checkpoint path in the runtime_defaults configuration.
    log_every_n_steps : int, optional
        Number of steps to log.  If provided, this overrides the log_every_n_steps in the runtime_defaults configuration.
    simulation_environment : str, optional
        Simulation environment.  If provided, this overrides the simulation environment in the runtime_defaults configuration.

    Returns
    -------
    Trainer
        The configured trainer instance after running the training process.
    """
    (
        training_parameter,
        dataset_parameter,
        potential_parameter,
        runtime_parameter,
    ) = read_config(
        condensed_config_path=condensed_config_path,
        training_parameter_path=training_parameter_path,
        dataset_parameter_path=dataset_parameter_path,
        potential_parameter_path_path=potential_parameter_path_path,
        runtime_parameter_path=runtime_parameter_path,
        accelerator=accelerator,
        devices=devices,
        number_of_nodes=number_of_nodes,
        experiment_name=experiment_name,
        save_dir=save_dir,
        local_cache_dir=local_cache_dir,
        checkpoint_path=checkpoint_path,
        log_every_n_steps=log_every_n_steps,
        simulation_environment=simulation_environment,
    )
    from modelforge.potential.models import (
        NeuralNetworkPotentialFactory,
    )

    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        simulation_environment=simulation_environment,
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )

    return model.train()

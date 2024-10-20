"""
This module contains classes and functions for training neural network potentials using PyTorch Lightning.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar

import lightning.pytorch as pL
import torch
from lightning import Trainer
from loguru import logger as log
from openff.units import unit
from torch.nn import ModuleDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modelforge.dataset.dataset import DataModule, DatasetParameters
from modelforge.potential.parameters import (
    AimNet2Parameters,
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
)
from modelforge.utils.prop import BatchData

# Define a TypeVar that can be one of the parameter models
T_NNP_Parameters = TypeVar(
    "T_NNP_Parameters",
    ANI2xParameters,
    SAKEParameters,
    SchNetParameters,
    PhysNetParameters,
    PaiNNParameters,
    TensorNetParameters,
    AimNet2Parameters,
)

from modelforge.train.losses import LossFactory, create_error_metrics
from modelforge.train.parameters import RuntimeParameters, TrainingParameters

__all__ = [
    "PotentialTrainer",
]


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def compute_grad_norm(loss, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss.sum(),
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    total_norm = 0.0
    for grad in grads:
        if grad is not None:
            param_norm = grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def _exchange_per_atom_energy_for_per_system_energy(prop: str) -> str:
    """
    Utility function to rename per-atom energy to per-system energy if applicable.

    Parameters
    ----------
    prop : str
        The property name (e.g., "per_atom_energy").

    Returns
    -------
    str
        The updated property name (e.g., "per_system_energy").
    """
    return "per_system_energy" if prop == "per_atom_energy" else prop


class CalculateProperties(torch.nn.Module):
    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_atom_force",
        "per_system_energy",
        "per_system_total_charge",
        "per_system_dipole_moment",
    ]

    def __init__(self, requested_properties: List[str]):
        """
        A utility class for calculating properties such as energies and forces from batches using a neural network model.

        Parameters
        requested_properties : List[str]
            A list of properties to calculate (e.g., per_atom_energy, per_atom_force, per_system_dipole_moment).
        """
        super().__init__()
        self.requested_properties = requested_properties
        self.include_force = "per_atom_force" in self.requested_properties
        self.include_charges = "per_system_total_charge" in self.requested_properties

        assert all(
            prop in self._SUPPORTED_PROPERTIES for prop in self.requested_properties
        ), f"Unsupported property requested: {self.requested_properties}"

    @staticmethod
    def _get_forces(
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
        train_mode: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        model_prediction : Dict[str, torch.Tensor]
            A dictionary containing the predicted energies from the model.
        train_mode : bool
            Whether to retain the graph for gradient computation (True for training).

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted forces.
        """
        nnp_input = batch.nnp_input
        # Ensure gradients are enabled
        nnp_input.positions.requires_grad_(True)
        # Cast to float32 and extract true forces
        per_atom_force_true = batch.metadata.per_atom_force.to(torch.float32)

        if per_atom_force_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        # Sum the energies before computing the gradient
        total_energy = model_prediction["per_system_energy"].sum()
        # Calculate forces as the negative gradient of energy w.r.t. positions
        grad = torch.autograd.grad(
            total_energy,
            nnp_input.positions,
            create_graph=train_mode,
            retain_graph=train_mode,
            allow_unused=False,
        )[0]

        if grad is None:
            raise RuntimeWarning("Force calculation did not return a gradient")

        per_atom_force_predict = -1 * grad  # Forces are the negative gradient of energy

        return {
            "per_atom_force_true": per_atom_force_true,
            "per_atom_force_predict": per_atom_force_predict.contiguous(),
        }

    @staticmethod
    def _get_energies(
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the energies from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        model_prediction : Dict[str, torch.Tensor]
            The neural network model used to compute the energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted energies.
        """
        per_system_energy_true = batch.metadata.per_system_energy.to(torch.float32)
        per_system_energy_predict = model_prediction["per_system_energy"].unsqueeze(1)

        assert per_system_energy_true.shape == per_system_energy_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{per_system_energy_true.shape} != {per_system_energy_predict.shape}"
        )
        return {
            "per_system_energy_true": per_system_energy_true,
            "per_system_energy_predict": per_system_energy_predict,
        }

    def _get_charges(
        self,
        batch: BatchData,
        model_prediction: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total molecular charges and dipole moments from the predicted atomic charges.

        Parameters
        ----------
        batch : BatchData
            A batch of data containing input features and target charges.
        model_prediction : Dict[str, torch.Tensor]
            A dictionary containing the predicted charges from the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the true and predicted charges and dipole moments.
        """
        nnp_input = batch.nnp_input
        per_atom_charges_predict = model_prediction[
            "per_atom_charge"
        ]  # Shape: [num_atoms]

        # Calculate predicted total charge by summing per-atom charges for each system
        per_system_total_charge_predict = (
            torch.zeros_like(model_prediction["per_system_energy"])
            .scatter_add_(
                dim=0,
                index=nnp_input.atomic_subsystem_indices.long(),
                src=per_atom_charges_predict,
            )
            .unsqueeze(-1)
        )  # Shape: [nr_of_systems, 1]

        # Predict the dipole moment
        per_system_dipole_moment = self._predict_dipole_moment(model_prediction, batch)

        return {
            "per_system_total_charge_predict": per_system_total_charge_predict,
            "per_system_total_charge_true": batch.nnp_input.per_system_total_charge,
            "per_system_dipole_moment_predict": per_system_dipole_moment,
            "per_system_dipole_moment_true": batch.metadata.per_system_dipole_moment,
        }

    @staticmethod
    def _predict_dipole_moment(
        model_predictions: Dict[str, torch.Tensor], batch: BatchData
    ) -> torch.Tensor:
        """
        Compute the predicted dipole moment for each system based on the
        predicted partial atomic charges and positions, i.e., the dipole moment
        is calculated as the weighted sum of the partial charges (which requires
        that the coordinates are centered).

        The dipole moment ensures that the predicted charges not only sum up to
        the correct total charge but also reproduce the reference dipole moment.

        Parameters
        ----------
        model_predictions : Dict[str, torch.Tensor]
            A dictionary containing the predicted atomic charges from the model.
        batch : BatchData
            A batch of data containing the atomic positions and indices.

        Returns
        -------
        torch.Tensor
            The predicted dipole moment for each system.
        """
        per_atom_charge = model_predictions["per_atom_charge"]  # Shape: [num_atoms]
        positions = batch.nnp_input.positions  # Shape: [num_atoms, 3]
        per_atom_charge = per_atom_charge.unsqueeze(-1)  # Shape: [num_atoms, 1]
        per_atom_dipole_contrib = per_atom_charge * positions  # Shape: [num_atoms, 3]

        indices = batch.nnp_input.atomic_subsystem_indices.long()  # Shape: [num_atoms]
        indices = indices.unsqueeze(-1).expand(-1, 3)  # Shape: [num_atoms, 3]

        # Calculate dipole moment as the sum of dipole contributions for each
        # system
        dipole_predict = torch.zeros(
            (model_predictions["per_system_energy"].shape[0], 3),
            device=positions.device,
            dtype=positions.dtype,
        ).scatter_add_(
            dim=0,
            index=indices,
            src=per_atom_dipole_contrib,
        )  # Shape: [nr_of_systems, 3]

        return dipole_predict

    def forward(
        self,
        batch: BatchData,
        model: torch.nn.Module,
        train_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes energies, forces, and charges from a given batch using the
        model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target
            energies.
        model : Type[torch.nn.Module]
            The neural network model used to compute the properties.
        train_mode : bool, optional
            Whether to calculate gradients for forces (default is False).

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the
            model.
        """
        predict_target = {}
        nnp_input = batch.nnp_input
        model_prediction = model.forward(nnp_input)

        # Get predicted energies
        energies = self._get_energies(batch, model_prediction)
        predict_target.update(energies)

        # Get forces if they are included in the requested properties
        if self.include_force:
            forces = self._get_forces(batch, model_prediction, train_mode)
            predict_target.update(forces)

        # Get charges if they are included in the requested properties
        if self.include_charges:
            charges = self._get_charges(batch, model_prediction)
            predict_target.update(charges)

        return predict_target


class TrainingAdapter(pL.LightningModule):
    """
    A Lightning module that encapsulates the training process for neural network potentials.
    """

    def __init__(
        self,
        *,
        potential_parameter: T_NNP_Parameters,
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        training_parameter: TrainingParameters,
        optimizer_class: Type[Optimizer],
        potential_seed: Optional[int] = None,
    ):
        """
        Initialize the TrainingAdapter with model and training configuration.

        Parameters
        ----------
        potential_parameter : T_NNP_Parameters
            Parameters for the potential model.
        dataset_statistic : Dict[str, Dict[str, unit.Quantity]]
            Dataset statistics such as mean and standard deviation.
        training_parameter : TrainingParameters
            Training configuration, including optimizer, learning rate, and loss functions.
        potential_seed : Optional[int], optional
            Seed for initializing the model (default is None).
        """
        from modelforge.potential.potential import setup_potential

        super().__init__()
        self.save_hyperparameters()
        self.training_parameter = training_parameter

        self.potential = setup_potential(
            potential_parameter=potential_parameter,
            dataset_statistic=dataset_statistic,
            potential_seed=potential_seed,
            jit=False,
            use_training_mode_neighborlist=True,
        )

        self.include_force = (
            "per_atom_force" in training_parameter.loss_parameter.loss_property
        )

        self.calculate_predictions = CalculateProperties(
            training_parameter.loss_parameter.loss_property
        )
        self.optimizer_class = optimizer_class
        self.learning_rate = training_parameter.lr
        self.lr_scheduler = training_parameter.lr_scheduler

        # verbose output, only True if requested
        if training_parameter.verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # Initialize the loss function
        self.loss = LossFactory.create_loss(
            **training_parameter.loss_parameter.model_dump()
        )

        # Initialize performance metrics
        self.test_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )
        self.val_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )
        self.train_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )

        self.loss_metrics = create_error_metrics(
            training_parameter.loss_parameter.loss_property, is_loss=True
        )

    def forward(self, batch: BatchData) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute energies, forces, and other properties from a
        batch.

        Parameters
        ----------
        batch : BatchData
            A batch of data including input features and target properties.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of predicted properties (energies, forces, etc.).
        """
        return self.potential(batch)

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.potential, "_config_prior"):
            return self.potential._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    @staticmethod
    def _update_metrics(
        metrics: ModuleDict,
        predict_target: Dict[str, torch.Tensor],
    ):
        """
        Updates the provided metric collections with the predicted and true
        targets.

        Parameters
        ----------
        metrics : ModuleDict
            Metric collections for energy and force evaluation.
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for properties.
        """

        for prop, metric_collection in metrics.items():
            prop = _exchange_per_atom_energy_for_per_system_energy(
                prop
            )  # only exchange per_atom_energy for per_system_energy
            preds = predict_target[f"{prop}_predict"].detach()
            targets = predict_target[f"{prop}_true"].detach()
            metric_collection.update(preds, targets)

    def training_step(
        self,
        batch: BatchData,
        batch_idx: int,
    ) -> torch.Tensor:
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

        predict_target = self.calculate_predictions(
            batch, self.potential, self.training
        )

        loss_dict = self.loss(predict_target, batch)  # Contains per-sample losses

        # Update loss metrics with per-sample losses
        batch_size = batch.batch_size()
        for key, metric in loss_dict.items():
            self.loss_metrics[key].update(metric.detach(), batch_size=batch_size)

            # Compute and log gradient norms for each loss component
            if self.training_parameter.log_norm:
                if key == "total_loss":
                    continue

                grad_norm = compute_grad_norm(metric.mean(), self)
                self.log(f"grad_norm/{key}", grad_norm)

        # Compute the mean loss for optimization
        total_loss = loss_dict["total_loss"].mean()

        return total_loss

    def on_after_backward(self):
        # After backward pass
        for name, param in self.potential.named_parameters():
            if param.grad is not None and False:
                log.debug(
                    f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}"
                )

    def validation_step(self, batch: BatchData, batch_idx: int) -> None:
        """
        Validation step to compute validation loss and metrics.
        """

        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        with torch.set_grad_enabled(True):
            # calculate energy and forces
            predict_target = self.calculate_predictions(
                batch, self.potential, self.potential.training
            )

        self._update_metrics(self.val_metrics, predict_target)

    def test_step(self, batch: BatchData, batch_idx: int) -> None:
        """
        Test step to compute the test loss and metrics.
        """
        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        with torch.set_grad_enabled(True):
            predict_target = self.calculate_predictions(
                batch, self.potential, self.training
            )
        # Update and log metrics
        self._update_metrics(self.test_metrics, predict_target)

    def on_validation_epoch_end(self):
        """Logs metrics at the end of the validation epoch."""
        self._log_metrics(self.val_metrics, "val")

    def on_test_epoch_end(self):
        """Logs metrics at the end of the test epoch."""
        self._log_metrics(self.test_metrics, "test")

    def on_train_epoch_end(self):
        """Logs metrics at the end of the training epoch."""
        self._log_metrics(self.loss_metrics, "loss")
        self._log_learning_rate()
        self._log_histograms()

    def _log_learning_rate(self):
        """Logs the current learning rate."""
        sch = self.lr_schedulers()
        try:
            self.log(
                "lr", sch.get_last_lr()[0], on_epoch=True, prog_bar=True, sync_dist=True
            )
        except AttributeError:
            pass

    def _log_metrics(self, metrics: ModuleDict, phase: str):
        """Logs all accumulated metrics for a given phase."""
        # abbreviate long names to shorter versions
        abbreviate = {
            "MeanAbsoluteError": "mae",
            "MeanSquaredError": "rmse",
            "MeanMetric": "mse",  # NOTE: MeanMetric is the MSE since we accumulate the squared error
        }  # NOTE: MeanSquaredError(squared=False) is RMSE

        for prop, metric_collection in metrics.items():
            for metric_name, metric in metric_collection.items():
                metric_value = metric.compute()
                metric.reset()
                self.log(
                    f"{phase}/{prop}/{abbreviate[metric_name]}",
                    metric_value,
                    prog_bar=True,
                    sync_dist=True,
                )

    def _log_histograms(self):
        if self.log_histograms:
            for name, params in self.named_parameters():
                if params is not None:
                    self.logger.experiment.add_histogram(
                        name, params, self.current_epoch
                    )
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}.grad", params.grad, self.current_epoch
                    )

    def configure_optimizers(self):
        """Configures the optimizers and learning rate schedulers."""

        optimizer = self.optimizer_class(
            self.potential.parameters(), lr=self.learning_rate
        )

        lr_scheduler = self.lr_scheduler.model_dump()
        interval = lr_scheduler.pop("interval")
        frequency = lr_scheduler.pop("frequency")
        monitor = lr_scheduler.pop("monitor")

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            **lr_scheduler,
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "monitor": monitor,  # Name of the metric to monitor
            "interval": interval,
            "frequency": frequency,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


from openff.units import unit


class PotentialTrainer:
    """
    Class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        dataset_parameter: DatasetParameters,
        potential_parameter: T_NNP_Parameters,
        training_parameter: TrainingParameters,
        runtime_parameter: RuntimeParameters,
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        use_default_dataset_statistic: bool,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        potential_seed: Optional[int] = None,
        verbose: bool = False,
        log_norm: bool = False,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        dataset_parameter : DatasetParameters
            Parameters for the dataset.
        potential_parameter : Union[ANI2xParameters, SAKEParameters, SchNetParameters, PhysNetParameters, PaiNNParameters, TensorNetParameters]
            Parameters for the potential model.
        training_parameter : TrainingParameters
            Parameters for the training process.
        runtime_parameter : RuntimeParameters
            Parameters for runtime configuration.
        dataset_statistic : Dict[str, Dict[str, unit.Quantity]]
            Dataset statistics such as mean and standard deviation.
        use_default_dataset_statistic: bool
            Whether to use default dataset statistic
        optimizer_class : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        potential_seed: Optional[int], optional
            Seed to initialize the potential training adapter, default is None.
        verbose : bool, optional
            If True, enables verbose logging, by default False.
        log_norm : bool, optional
            If True, logs the norm of the gradients, by default False.
        """

        super().__init__()

        self.dataset_parameter = dataset_parameter
        self.potential_parameter = potential_parameter
        self.training_parameter = training_parameter
        self.runtime_parameter = runtime_parameter
        self.verbose = verbose
        self.log_norm = log_norm

        self.datamodule = self.setup_datamodule()
        self.dataset_statistic = (
            self.read_dataset_statistics()
            if not use_default_dataset_statistic
            else dataset_statistic
        )
        self.experiment_logger = self.setup_logger()
        self.callbacks = self.setup_callbacks()
        self.trainer = self.setup_trainer()
        self.optimizer_class = optimizer_class
        self.lightning_module = self.setup_lightning_module(potential_seed)
        self.learning_rate = self.training_parameter.lr
        self.lr_scheduler = self.training_parameter.lr_scheduler

    def read_dataset_statistics(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Read and log dataset statistics.

        Returns
        -------
        Dict[str, float]
            The dataset statistics.
        """
        from modelforge.potential.utils import (
            convert_str_to_unit_in_dataset_statistics,
            read_dataset_statistics,
        )

        # read toml file
        dataset_statistic = read_dataset_statistics(
            self.datamodule.dataset_statistic_filename
        )
        # convert dictionary of str:str to str:units
        dataset_statistic = convert_str_to_unit_in_dataset_statistics(dataset_statistic)
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
        from modelforge.dataset.dataset import DataModule
        from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES

        dm = DataModule(
            name=self.dataset_parameter.dataset_name,
            batch_size=self.training_parameter.batch_size,
            remove_self_energies=self.training_parameter.remove_self_energies,
            shift_center_of_mass_to_origin=self.training_parameter.shift_center_of_mass_to_origin,
            version_select=self.dataset_parameter.version_select,
            local_cache_dir=self.runtime_parameter.local_cache_dir,
            splitting_strategy=REGISTERED_SPLITTING_STRATEGIES[
                self.training_parameter.splitting_strategy.name
            ](
                seed=self.training_parameter.splitting_strategy.seed,
                split=self.training_parameter.splitting_strategy.data_split,
            ),
            regenerate_processed_cache=self.dataset_parameter.regenerate_processed_cache,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def setup_lightning_module(
        self, potential_seed: Optional[int] = None
    ) -> pL.LightningModule:
        """
        Set up the model for training.

        Parameters
        ----------
        potential_seed : int, optional
            Seed to be used to initialize the potential, by default None.

        Returns
        -------
        nn.Module
            Configured model instance, wrapped in a TrainingAdapter.
        """

        # Initialize model
        return TrainingAdapter(
            potential_parameter=self.potential_parameter,
            dataset_statistic=self.dataset_statistic,
            training_parameter=self.training_parameter,
            optimizer_class=self.optimizer_class,
            potential_seed=potential_seed,
        )

    def setup_logger(self) -> pL.loggers.Logger:
        """
        Set up the experiment logger based on the configuration.

        Returns
        -------
        pL.loggers.Logger
            Configured logger instance.
        """

        experiment_name = self._format_experiment_name(
            self.runtime_parameter.experiment_name
        )

        if self.training_parameter.experiment_logger.logger_name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            logger = (
                TensorBoardLogger(
                    save_dir=str(
                        self.training_parameter.experiment_logger.tensorboard_configuration.save_dir
                    ),
                    name=experiment_name,
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
                log_model=self.training_parameter.experiment_logger.wandb_configuration.log_model,
                project=self.training_parameter.experiment_logger.wandb_configuration.project,
                group=self.training_parameter.experiment_logger.wandb_configuration.group,
                job_type=self.training_parameter.experiment_logger.wandb_configuration.job_type,
                tags=self._generate_tags(
                    self.training_parameter.experiment_logger.wandb_configuration.tags
                ),
                notes=self.training_parameter.experiment_logger.wandb_configuration.notes,
                name=experiment_name,
            )
        else:
            raise ValueError("Unsupported logger type.")
        return logger

    def setup_callbacks(self) -> List[Any]:
        """
        Set up the callbacks for the trainer.
        The callbacks include early stopping (optional), model checkpointing, and stochastic weight averaging (optional).


        Returns
        -------
        List[Any]
            List of configured callbacks.
        """
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            StochasticWeightAveraging,
            Callback,
        )

        callbacks = []
        if self.training_parameter.stochastic_weight_averaging:
            callbacks.append(
                StochasticWeightAveraging(
                    **self.training_parameter.stochastic_weight_averaging.model_dump()
                )
            )

        if self.training_parameter.early_stopping:
            callbacks.append(
                EarlyStopping(**self.training_parameter.early_stopping.model_dump())
            )

        # Save the best model based on the validation loss
        # NOTE: The filename is formatted as "best_{potential_name}-{dataset_name}-{epoch:02d}"
        checkpoint_filename = (
            f"best_{self.potential_parameter.potential_name}-{self.dataset_parameter.dataset_name}"
            + "-{epoch:02d}"
        )
        callbacks.append(
            ModelCheckpoint(
                save_top_k=2,
                monitor=self.training_parameter.monitor,
                filename=checkpoint_filename,
            )
        )

        # compute gradient norm
        class GradNormCallback(Callback):
            """
            Logs the gradient norm.
            """

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                pl_module.log("grad_norm/model", gradient_norm(pl_module))

        if self.log_norm:
            callbacks.append(GradNormCallback())

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

        # if devices is a list
        if isinstance(self.runtime_parameter.devices, list) or (
            isinstance(self.runtime_parameter.devices, int)
            and self.runtime_parameter.devices > 1
        ):
            from lightning.pytorch.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = "auto"

        trainer = Trainer(
            strategy=strategy,
            max_epochs=self.training_parameter.number_of_epochs,
            min_epochs=self.training_parameter.min_number_of_epochs,
            num_nodes=self.runtime_parameter.number_of_nodes,
            devices=self.runtime_parameter.devices,
            accelerator=self.runtime_parameter.accelerator,
            logger=self.experiment_logger,
            callbacks=self.callbacks,
            benchmark=True,
            inference_mode=False,
            num_sanity_val_steps=2,
            gradient_clip_val=10.0,  # FIXME: hardcoded for now
            log_every_n_steps=self.runtime_parameter.log_every_n_steps,
            enable_model_summary=True,
            enable_progress_bar=self.runtime_parameter.verbose,  # if true will show progress bar
        )
        return trainer

    def train_potential(self) -> Trainer:
        """
        Run the training process.

        Returns
        -------
        Trainer
            The configured trainer instance after running the training process.
        """
        self.trainer.fit(
            self.lightning_module,
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
            model=self.lightning_module,
            dataloaders=self.datamodule.val_dataloader(),
            ckpt_path="best",
            verbose=True,
        )

        self.trainer.test(
            model=self.lightning_module,
            dataloaders=self.datamodule.test_dataloader(),
            ckpt_path="best",
            verbose=True,
        )
        return self.trainer

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.lightning_module, "_config_prior"):
            return self.lightning_module._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _format_experiment_name(self, experiment_name: str) -> str:
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

    def _generate_tags(self, tags: List[str]) -> List[str]:
        """Generates tags for the experiment."""
        import modelforge

        tags.extend(
            [
                str(modelforge.__version__),
                self.dataset_parameter.dataset_name,
                self.potential_parameter.potential_name,
                f"loss-{'-'.join(self.training_parameter.loss_parameter.loss_property)}",
            ]
        )
        return tags


from typing import List, Optional, Union


def read_config(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
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
    Reads one or more TOML configuration files and loads them into the pydantic models.

    Parameters
    ----------
    (Parameters as described earlier...)

    Returns
    -------
    Tuple
        Tuple containing the training, dataset, potential, and runtime parameters.
    """
    import toml

    # Initialize the config dictionaries
    training_config_dict = {}
    dataset_config_dict = {}
    potential_config_dict = {}
    runtime_config_dict = {}

    if condensed_config_path is not None:
        config = toml.load(condensed_config_path)
        log.info(f"Reading config from : {condensed_config_path}")

        training_config_dict = config.get("training", {})
        dataset_config_dict = config.get("dataset", {})
        potential_config_dict = config.get("potential", {})
        runtime_config_dict = config.get("runtime", {})

    else:
        if training_parameter_path:
            training_config_dict = toml.load(training_parameter_path).get(
                "training", {}
            )
        if dataset_parameter_path:
            dataset_config_dict = toml.load(dataset_parameter_path).get("dataset", {})
        if potential_parameter_path:
            potential_config_dict = toml.load(potential_parameter_path).get(
                "potential", {}
            )
        if runtime_parameter_path:
            runtime_config_dict = toml.load(runtime_parameter_path).get("runtime", {})

    # Override runtime configuration with command-line arguments if provided
    runtime_overrides = {
        "accelerator": accelerator,
        "devices": devices,
        "number_of_nodes": number_of_nodes,
        "experiment_name": experiment_name,
        "save_dir": save_dir,
        "local_cache_dir": local_cache_dir,
        "checkpoint_path": checkpoint_path,
        "log_every_n_steps": log_every_n_steps,
        "simulation_environment": simulation_environment,
    }

    for key, value in runtime_overrides.items():
        if value is not None:
            runtime_config_dict[key] = value

    # Load and instantiate the data classes with the merged configuration
    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.potential import _Implemented_NNP_Parameters
    from modelforge.train.parameters import RuntimeParameters, TrainingParameters

    potential_name = potential_config_dict["potential_name"]
    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )

    dataset_parameters = DatasetParameters(**dataset_config_dict)
    training_parameters = TrainingParameters(**training_config_dict)
    runtime_parameters = RuntimeParameters(**runtime_config_dict)
    potential_parameter = PotentialParameters(**potential_config_dict)

    return (
        training_parameters,
        dataset_parameters,
        potential_parameter,
        runtime_parameters,
    )


def read_config_and_train(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
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
    potential_parameter_path : str, optional
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
        potential_parameter_path=potential_parameter_path,
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
    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    trainer = NeuralNetworkPotentialFactory.generate_potential(
        use="training",
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )

    return trainer.train_potential()

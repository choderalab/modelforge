from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
from typing import TYPE_CHECKING, Any, Union, Dict, Type
import torch
from loguru import logger as log
from modelforge.dataset.dataset import BatchData

if TYPE_CHECKING:
    from modelforge.potential.utils import BatchData

import torchmetrics
from torchmetrics.utilities import dim_zero_cat
from typing import Optional


class MSELossMetric(torchmetrics.Metric):
    """
    Custom metric to calculate the Mean Squared Error (MSE) loss per batch and average it over an epoch.

    Attributes
    ----------
    mse_loss_per_batch : List[torch.Tensor]
        List to store the MSE loss for each batch.
    """

    def __init__(self) -> None:
        """
        Initializes the MSELossMetric class, setting up the state for the metric.
        """
        super().__init__()
        self.add_state("mse_loss_per_batch", default=[], dist_reduce_fx="cat")

    def update(self, loss: torch.Tensor) -> None:
        """
        Updates the metric state with the MSE loss for a batch.

        Parameters
        ----------
        loss : torch.Tensor
            The MSE loss for a batch.
        """
        self.mse_loss_per_batch.append(loss.detach())

    def compute(self) -> torch.Tensor:
        """
        Computes the average MSE loss over all batches in an epoch.

        Returns
        -------
        torch.Tensor
            The average MSE loss for the epoch.
        """
        mse_loss_per_epoch = dim_zero_cat(self.mse_loss_per_batch)
        return torch.mean(mse_loss_per_epoch)


class RMSELossMetric(MSELossMetric):
    """
    Custom metric to calculate the Root Mean Squared Error (RMSE) loss per batch and average it over an epoch.
    Inherits from MSELossMetric.

    Methods
    -------
    compute()
        Computes the average RMSE loss over all batches in an epoch.
    """

    def compute(self) -> torch.Tensor:
        """
        Computes the average RMSE loss over all batches in an epoch.

        Returns
        -------
        torch.Tensor
            The average RMSE loss for the epoch.
        """
        mse_loss_per_epoch = dim_zero_cat(self.mse_loss_per_batch)
        return torch.sqrt(torch.mean(mse_loss_per_epoch))


from torch import nn

from abc import abstractmethod


class Loss(nn.Module):

    @abstractmethod
    def calculate_loss(
        self, predict_target: Dict[str, torch.Tensor], batch: BatchData
    ) -> Dict[str, torch.Tensor]:
        pass


class LossFactory(object):

    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> Type[Loss]:

        if loss_type == "NaiveEnergyAndForceLoss":
            return NaiveEnergyAndForceLoss(**kwargs)
        elif loss_type == "EnergyLoss":
            return EnergyLoss()
        else:
            raise ValueError(f"Loss type {loss_type} not implemented.")


class EnergyLoss(Loss):
    def __init__(
        self,
    ):
        super().__init__()
        from torch.nn import MSELoss

        self.mse_loss = MSELoss()

    def calculate_loss(
        self, predict_target: Dict[str, torch.Tensor], batch: Optional[BatchData] = None
    ):
        E_loss = self.mse_loss(predict_target["E_predict"], predict_target["E_true"])

        return {
            "combined_loss": E_loss,
            "energy_loss": E_loss,
            "force_loss": torch.zeros_like(E_loss),
        }


class NaiveEnergyAndForceLoss(Loss):
    """
    Class to calculate the combined loss for both energy and force predictions.

    Attributes
    ----------
    include_force : bool
        Whether to include force in the loss calculation.
    energy_weight : torch.Tensor
        Weight for the energy loss component.
    force_weight : torch.Tensor
        Weight for the force loss component.
    """

    def __init__(
        self,
        include_force: bool = False,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
    ):
        """
        Initializes the NaiveEnergyAndForceLoss class.

        Parameters
        ----------
        include_force : bool, optional
            Whether to include force in the loss calculation, by default False.
        energy_weight : float, optional
            Weight for the energy loss component, by default 1.0.
        force_weight : float, optional
            Weight for the force loss component, by default 1.0.
        """
        super().__init__()
        self.include_force = include_force
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("force_weight", torch.tensor(force_weight))

    def calculate_loss(self, predict_target: Dict[str, torch.Tensor], batch: BatchData):
        """
        Calculates the combined loss for both energy and force predictions.

        Parameters
        ----------
        predict_target : dict
            Dictionary containing predicted and true values for energy and force.
            Expected keys are 'E_predict', 'E_true', 'F_predict', 'F_true'.
        batch : BatchData
            Batch of data, including input features and target values.

        Returns
        -------
        dict
            Dictionary containing combined loss, energy loss, and force loss.
        """
        from torch_scatter import scatter_sum

        # Calculate per-atom force error
        F_error_per_atom = (
            torch.norm(predict_target["F_predict"] - predict_target["F_true"], dim=1)
            ** 2
        )
        # Aggregate force error per molecule
        F_error_per_molecule = scatter_sum(
            F_error_per_atom, batch.nnp_input.atomic_subsystem_indices.long(), 0
        )

        # Scale factor for force loss
        scale = self.force_weight / (3 * batch.metadata.atomic_subsystem_counts)
        # Calculate energy loss
        E_loss = (
            self.energy_weight
            * (predict_target["E_predict"] - predict_target["E_true"]) ** 2
        )
        # Calculate force loss
        F_loss = scale * F_error_per_molecule
        # Combine energy and force losses
        combined_loss = torch.mean(E_loss + F_loss)
        return {
            "combined_loss": combined_loss,
            "energy_loss": E_loss,
            "force_loss": F_loss,
        }


from torch.optim import Optimizer


class TrainingAdapter(pl.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        model_parameters: Dict[str, Any],
        lr_scheduler_config: Dict[str, Union[str, int, float]],
        lr: float,
        loss_parameter: Dict[str, Any],
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        verbose_level: int = 5,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        nnp_parameters : Dict[str, Any]
            The parameters for the neural network potential model.
        lr_scheduler_config : Dict[str, Union[str, int, float]]
            The configuration for the learning rate scheduler.
        lr : float
            The learning rate for the optimizer.
        loss_module : Loss, optional
            Whether to include force in the loss function, by default False.
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        """

        from typing import List
        from modelforge.potential import _Implemented_NNPs

        super().__init__()
        self.save_hyperparameters(ignore=["loss_module"])
        # Extracting and instantiating the model from parameters
        nnp_parameters_ = model_parameters.copy()
        nnp_name = nnp_parameters_.pop("nnp_name", None)
        if nnp_name is None:
            raise ValueError(
                "NNP name must be specified in nnp_parameters with key 'nnp_name'."
            )
        nnp_class: Type = _Implemented_NNPs.get_neural_network_class(nnp_name)
        if nnp_class is None:
            raise ValueError(f"Specified NNP name '{nnp_name}' is not implemented.")

        self.model = nnp_class(**nnp_parameters_)
        self.optimizer = optimizer
        self.learning_rate = lr
        self.lr_scheduler_config = lr_scheduler_config
        self.loss_module = LossFactory.create_loss(**loss_parameter)

        self.unused_parameters = set()
        self.are_unused_parameters_present = False
        self.verbose_level = verbose_level

        self.val_energy_loss_metric = RMSELossMetric()
        self.val_force_loss_metric = RMSELossMetric()
        self.val_combined_loss_metric = RMSELossMetric()

        self.train_energy_loss_metric = MSELossMetric()
        self.train_force_loss_metric = MSELossMetric()
        self.train_combined_loss_metric = MSELossMetric()

        self.test_energy_loss_metric = RMSELossMetric()
        self.test_force_loss_metric = RMSELossMetric()
        self.test_combined_loss_metric = RMSELossMetric()

        self.val_loss = []

    def _log_on_epoch_metrics(
        self, loss: Dict[str, torch.Tensor], prefix: str, progress_bar: bool = True
    ):
        """
        Logs metrics for the specified prefix (train, val, test) during the epoch.

        Parameters
        ----------
        loss : dict
            Dictionary containing the loss error for each batch.
            Expected keys are 'combined_loss', 'energy_loss', and 'force_loss'.
        prefix : str
            Prefix indicating the current phase of the training process.
            Should be one of 'train', 'val', or 'test'.
        progress_bar : bool, optional
            Whether to display the metrics in the progress bar, by default True.

        Raises
        ------
        RuntimeError
            If the prefix is not recognized.
        """
        # Validate prefix
        if prefix not in ["train", "val", "test"]:
            raise RuntimeError(f"Unknown prefix: {prefix}")

        # Dictionary to hold loss metrics based on the prefix
        loss_metrics = {
            "train": {
                "combined_loss": self.train_combined_loss_metric,
                "energy_loss": self.train_energy_loss_metric,
                "force_loss": self.train_force_loss_metric,
            },
            "val": {
                "combined_loss": self.val_combined_loss_metric,
                "energy_loss": self.val_energy_loss_metric,
                "force_loss": self.val_force_loss_metric,
            },
            "test": {
                "combined_loss": self.test_combined_loss_metric,
                "energy_loss": self.test_energy_loss_metric,
                "force_loss": self.test_force_loss_metric,
            },
        }[prefix]

        # Log metrics
        for key, metric in loss_metrics.items():
            if key == "energy_loss" or self.verbose_level > 0:
                metric(loss[key])
                self.log(
                    f"{prefix}/{key}",
                    metric,
                    on_epoch=True,
                    prog_bar=progress_bar,
                )

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
        F_true = batch.metadata.F.to(torch.float32)

        if F_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        E_predict = energies["E_predict"]

        # Ensure E_predict and nnp_input.positions require gradients and are on the same device
        if not E_predict.requires_grad:
            E_predict.requires_grad = True
        if not nnp_input.positions.requires_grad:
            nnp_input.positions.requires_grad = True

        # Compute the gradient (forces) from the predicted energies
        grad = torch.autograd.grad(
            E_predict.sum(),
            nnp_input.positions,
            create_graph=False,
            retain_graph=True,
        )[0]
        F_predict = -1 * grad  # Forces are the negative gradient of energy
        return {"F_true": F_true, "F_predict": F_predict}

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
        E_true = batch.metadata.E.to(torch.float32).squeeze(1)
        E_predict = self.model.forward(nnp_input).E
        assert E_true.shape == E_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{E_true.shape} != {E_predict.shape}"
        )
        return {"E_true": E_true, "E_predict": E_predict}

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
        loss = self.loss_module.calculate_loss(predict_target, batch)
        # Update and log metrics
        self._log_on_epoch_metrics(loss, "train", progress_bar=False)
        return loss["combined_loss"]

    @torch.enable_grad()
    def validation_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Validation step to compute the RMSE loss and accumulate L1 loss across epochs.

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
        loss = self.loss_module.calculate_loss(predict_target, batch)
        # log the loss
        self.val_loss.append(loss["combined_loss"].detach().item())
        self._log_on_epoch_metrics(loss, "val")

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
        # calculate the loss
        loss = self.loss_module.calculate_loss(predict_target, batch)
        # Update and log metrics
        self._log_on_epoch_metrics(loss, "test")

    def on_train_epoch_end(self):
        """
        Operations to perform at the end of each training epoch.

        Logs histograms of weights and biases, and learning rate. 
        Also, resets validation loss.
        """
        for name, params in self.named_parameters():
            if params is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
            if params.grad is not None:
                self.logger.experiment.add_histogram(
                    f"{name}.grad", params.grad, self.current_epoch
                )

        sch = self.lr_schedulers()
        try:
            self.log("lr", sch.get_last_lr()[0], on_epoch=True, prog_bar=True)
        except AttributeError:
            pass

        # Log and reset the validation loss
        avg_val_loss = torch.sqrt(torch.mean(torch.tensor(self.val_loss)))
        log.debug(avg_val_loss.item())
        self.val_loss = []

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

        lr_scheduler_config = self.lr_scheduler_config
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=lr_scheduler_config["mode"],
            factor=lr_scheduler_config["factor"],
            patience=lr_scheduler_config["patience"],
            cooldown=lr_scheduler_config["cooldown"],
            min_lr=lr_scheduler_config["min_lr"],
            threshold=lr_scheduler_config["threshold"],
            threshold_mode="abs",
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "monitor": "val/combined_loss",  # Name of the metric to monitor
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def return_toml_config(
    config_path: Optional[str] = None,
    potential_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    training_path: Optional[str] = None,
):
    """
    Read one or more TOML configuration files and return the parsed configuration.

    Parameters
    ----------
    config_path : str, optional
        The path to the TOML configuration file.
    potential_path : str, optional
        The path to the TOML file defining the potential configuration.
    dataset_path : str, optional
        The path to the TOML file defining the dataset configuration.
    training_path : str, optional
        The path to the TOML file defining the training configuration.

    Returns
    -------
    dict
        The merged parsed configuration from the TOML files.
    """
    import toml

    config = {}

    if config_path:
        config = toml.load(config_path)
        log.info(f"Reading config from : {config_path}")
    else:
        if potential_path:
            config["potential"] = toml.load(potential_path)["potential"]
            log.info(f"Reading potential config from : {potential_path}")
        if dataset_path:
            config["dataset"] = toml.load(dataset_path)["dataset"]
            log.info(f"Reading dataset config from : {dataset_path}")
        if training_path:
            config["training"] = toml.load(training_path)["training"]
            log.info(f"Reading training config from : {training_path}")

    return config


from typing import List, Optional, Union


def read_config_and_train(
    config_path: Optional[str] = None,
    potential_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    training_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    device: Optional[Union[int, List[int]]] = None,
):
    """
    Reads one or more TOML configuration files and performs training based on the parameters.

    Parameters
    ----------
    config_path : str, optional
        Path to the TOML configuration file.
    potential_path : str, optional
        Path to the TOML file defining the potential configuration.
    dataset_path : str, optional
        Path to the TOML file defining the dataset configuration.
    training_path : str, optional
        Path to the TOML file defining the training configuration.
    accelerator : str, optional
        Accelerator type to use for training.
    device : int|List[int], optional
        Device index to use for training.
    """
    # Read the TOML file
    config = return_toml_config(
        config_path, potential_path, dataset_path, training_path
    )

    # Extract parameters
    potential_config = config["potential"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    # Override config parameters with command-line arguments if provided
    if accelerator:
        training_config["accelerator"] = accelerator
    if device is not None:
        training_config["devices"] = device

    log.debug(f"Potential config: {potential_config}")
    log.debug(f"Dataset config: {dataset_config}")
    log.debug(f"Training config: {training_config}")
    # Call the perform_training function with extracted parameters
    perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
    )


from lightning import Trainer


def log_training_arguments(
    potential_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
):
    """
    Log arguments that are passed to the training routine.

    Arguments
    ----
        potential_config: Dict[str, Any]
            config for the potential model
        training_config: Dict[str, Any]
            config for the training process
        dataset_config: Dict[str, Any]
            config for the dataset
    """
    save_dir = training_config.get("save_dir", "lightning_logs")
    if save_dir == "lightning_logs":
        log.info(f"Saving logs to default location: {save_dir}")
    else:
        log.info(f"Saving logs to custom location: {save_dir}")

    experiment_name = training_config.get("experiment_name", "exp")
    if experiment_name == "experiment_name":
        log.info(f"Saving logs in default dir: {experiment_name}")
    else:
        log.info(f"Saving logs in custom dir: {experiment_name}")

    version_select = dataset_config.get("version_select", "latest")
    if version_select == "latest":
        log.info(f"Using default dataset version: {version_select}")

    else:
        log.info(f"Using dataset version: {version_select}")
    accelerator = training_config.get("accelerator", "cpu")
    if accelerator == "cpu":
        log.info(f"Using default accelerator: {accelerator}")
    else:
        log.info(f"Using accelerator: {accelerator}")
    nr_of_epochs = training_config.get("nr_of_epochs", 10)
    if nr_of_epochs == 10:
        log.info(f"Using default number of epochs: {nr_of_epochs}")
    else:
        log.info(f"Training for {nr_of_epochs} epochs")
    num_nodes = training_config.get("num_nodes", 1)
    if num_nodes == 1:
        log.info(f"Using default number of nodes: {num_nodes}")
    else:
        log.info(f"Training on {num_nodes} nodes")
    devices = training_config.get("devices", 1)
    if devices == 1:
        log.info(f"Using default device index/number: {devices}")
    else:
        log.info(f"Using device index/number: {devices}")
    batch_size = training_config.get("batch_size", 128)
    if batch_size == 128:
        log.info(f"Using default batch size: {batch_size}")
    else:
        log.info(f"Using batch size: {batch_size}")

    remove_self_energies = training_config.get("remove_self_energies", False)
    if remove_self_energies is False:
        log.warning(
            f"Using default for removing self energies: Self energies are not removed"
        )
    else:
        log.info(f"Removing self energies: {remove_self_energies}")
    early_stopping_config = training_config.get("early_stopping", None)
    if early_stopping_config is None:
        log.info(f"Using default: No early stopping performed")

    stochastic_weight_averaging_config = training_config.get(
        "stochastic_weight_averaging_config", None
    )

    num_workers = dataset_config.get("number_of_worker", 4)
    if num_workers == 4:
        log.info(
            f"Using default number of workers for training data loader: {num_workers}"
        )
    else:
        log.info(f"Using {num_workers} workers for training data loader")
    pin_memory = dataset_config.get("pin_memory", False)
    if pin_memory is False:
        log.info(f"Using default value for pinned_memory: {pin_memory}")
    else:
        log.info(f"Using pinned_memory: {pin_memory}")
    model_name = potential_config["model_name"]
    dataset_name = dataset_config["dataset_name"]
    log.info(training_config["loss_parameter"])
    log.debug(
        f"""
Training {model_name} on {dataset_name}-{version_select} dataset with {accelerator}
accelerator on {num_nodes} nodes for {nr_of_epochs} epochs.
Experiments are saved to: {save_dir}/{experiment_name}.
"""
    )


def perform_training(
    potential_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> Trainer:
    """
    Performs the training process for a neural network potential model.

    Parameters
    ----------
    potential_config : Dict[str, Any], optional
        Additional parameters for the potential model.
    training_config : Dict[str, Any], optional
        Additional parameters for the training process.
    dataset_config : Dict[str, Any], optional
        Additional parameters for the dataset.

    Returns
    -------
    Trainer
    """

    from pytorch_lightning.loggers import TensorBoardLogger
    from modelforge.dataset.utils import RandomRecordSplittingStrategy
    from lightning import Trainer
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.dataset.dataset import DataModule
    from lightning.pytorch.callbacks import ModelSummary

    save_dir = training_config.get("save_dir", "lightning_logs")
    if save_dir == "lightning_logs":
        log.info(f"Saving logs to default location: {save_dir}")

    experiment_name = training_config.get("experiment_name", "exp")
    model_name = potential_config["model_name"]
    dataset_name = dataset_config["dataset_name"]
    log_training_arguments(potential_config, training_config, dataset_config)

    version_select = dataset_config.get("version_select", "latest")
    accelerator = training_config.get("accelerator", "cpu")
    nr_of_epochs = training_config.get("nr_of_epochs", 10)
    num_nodes = training_config.get("num_nodes", 1)
    devices = training_config.get("devices", 1)
    batch_size = training_config.get("batch_size", 128)
    remove_self_energies = training_config.get("remove_self_energies", False)
    early_stopping_config = training_config.get("early_stopping", None)
    stochastic_weight_averaging_config = training_config.get(
        "stochastic_weight_averaging_config", None
    )
    num_workers = dataset_config.get("number_of_worker", 4)
    pin_memory = dataset_config.get("pin_memory", False)

    # set up tensor board logger
    logger = TensorBoardLogger(save_dir, name=experiment_name)

    log.debug(
        f"""
Training {model_name} on {dataset_name}-{version_select} dataset with {accelerator}
accelerator on {num_nodes} nodes for {nr_of_epochs} epochs.
Experiments are saved to: {save_dir}/{experiment_name}.
"""
    )

    log.debug(f"Using {potential_config} potential config")
    log.debug(f"Using {training_config} training config")

    # Set up dataset
    dm = DataModule(
        name=dataset_name,
        batch_size=batch_size,
        splitting_strategy=RandomRecordSplittingStrategy(),
        remove_self_energies=remove_self_energies,
        version_select=version_select,
    )
    # Set up model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training",
        model_type=model_name,
        loss_parameter=training_config["loss_parameter"],
        model_parameters=potential_config["potential_parameter"],
        training_parameters=training_config["training_parameter"],
    )

    # set up traininer
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks.stochastic_weight_avg import (
        StochasticWeightAveraging,
    )

    # set up callbacks
    callbacks = []
    if stochastic_weight_averaging_config:
        callbacks.append(
            StochasticWeightAveraging(**stochastic_weight_averaging_config)
        )
    if early_stopping_config:
        callbacks.append(EarlyStopping(**early_stopping_config))

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val/combined_loss",
        filename="best_{potential_name}-{dataset_name}-{epoch:02d}-{val_loss:.2f}",
    )

    callbacks.append(checkpoint_callback)

    # set up trainer
    trainer = Trainer(
        max_epochs=nr_of_epochs,
        num_nodes=num_nodes,
        devices=devices,
        accelerator=accelerator,
        logger=logger,  # Add the logger here
        callbacks=callbacks,
        inference_mode=False,
        num_sanity_val_steps=2,
    )

    dm.prepare_data()
    dm.setup()

    log.info(f"Setting E_i_mean and E_i_stddev for {model_name}")
    log.info(f"E_i_mean: {dm.dataset_statistics.E_i_mean}")
    log.info(f"E_i_stddev: {dm.dataset_statistics.E_i_stddev}")
    model.model.core_module.readout_module.E_i_mean = torch.tensor(
        [dm.dataset_statistics.E_i_mean], dtype=torch.float32
    )
    model.model.core_module.readout_module.E_i_stddev = torch.tensor(
        [dm.dataset_statistics.E_i_stddev], dtype=torch.float32
    )

    # Run training loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(
            num_workers=num_workers, pin_memory=pin_memory
        ),
        val_dataloaders=dm.val_dataloader(),
    )
    trainer.validate(model=model, dataloaders=dm.val_dataloader(), ckpt_path="best")
    trainer.test(dataloaders=dm.test_dataloader(), ckpt_path="best")
    return trainer

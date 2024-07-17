from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
from typing import Any, Union, Dict, Type, Optional, List
import torch
from loguru import logger as log
from modelforge.dataset.dataset import BatchData, NNPInput
import torchmetrics
from torch import nn
from abc import ABC, abstractmethod


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

        Parameters:
            predicted_tensor (torch.Tensor): The predicted values.
            reference_tensor (torch.Tensor): The values provided by the dataset.

        Returns:
            torch.Tensor: The calculated error.
        """
        return (predicted_tensor - reference_tensor).pow(2).sum(dim=1, keepdim=True)

    @staticmethod
    def scale_by_number_of_atoms(error, atomic_subsystem_counts) -> torch.Tensor:
        """
        Scales the error by the number of atoms in the atomic subsystems.

        Parameters:
            error: The error to be scaled.
            atomic_subsystem_counts: The number of atoms in the atomic subsystems.

        Returns:
            torch.Tensor: The scaled error.
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


class TrainingAdapter(pl.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        lr_scheduler_config: Dict[str, Union[str, int, float]],
        model_parameter: Dict[str, Any],
        lr: float,
        loss_parameter: Dict[str, Any],
        dataset_statistic: Optional[Dict[str, float]] = None,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        verbose: bool = False,
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
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        """

        from modelforge.potential import _Implemented_NNPs

        super().__init__()
        self.save_hyperparameters()

        # Get requested model class
        model_name = model_parameter["model_name"]
        nnp_class: Type = _Implemented_NNPs.get_neural_network_class(model_name)

        # initialize model
        self.model = nnp_class(
            **model_parameter["core_parameter"],
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=model_parameter["postprocessing_parameter"],
        )

        self.optimizer = optimizer
        self.learning_rate = lr
        self.lr_scheduler_config = lr_scheduler_config

        # verbose output, only True if requested
        if verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # initialize loss
        self.loss = LossFactory.create_loss(**loss_parameter)

        # Assign the created error metrics to the respective attributes
        self.test_error = create_error_metrics(loss_parameter["loss_property"])
        self.val_error = create_error_metrics(loss_parameter["loss_property"])
        self.train_error = create_error_metrics(loss_parameter["loss_property"])

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
            "monitor": lr_scheduler_config["monitor"],  # Name of the metric to monitor
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def return_toml_config(
    config_path: Optional[str] = None,
    potential_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    training_path: Optional[str] = None,
    runtime_path: Optional[str] = None,
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
    runtime_path : str, optional
        The path to the TOML file defining the runtime configuration.

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
        if runtime_path:
            config["runtime"] = toml.load(runtime_path)["runtime"]
            log.info(f"Reading runtime config from : {runtime_path}")
    return config


from typing import List, Optional, Union


def read_config_and_train(
    config_path: Optional[str] = None,
    potential_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    training_path: Optional[str] = None,
    runtime_path: Optional[str] = None,
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
    runtime_path : str, optional
        Path to the TOML file defining the runtime configuration.
    accelerator : str, optional
        Accelerator type to use for training.
    device : int|List[int], optional
        Device index to use for training.
    """
    # Read the TOML file
    config = return_toml_config(
        config_path, potential_path, dataset_path, training_path, runtime_path
    )

    # Extract parameters
    potential_config = config["potential"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    runtime_config = config["runtime"]

    # Override config parameters with command-line arguments if provided
    if accelerator:
        runtime_config["accelerator"] = accelerator
    if device is not None:
        runtime_config["devices"] = device

    log.debug(f"Potential config: {potential_config}")
    log.debug(f"Dataset config: {dataset_config}")
    log.debug(f"Training config: {training_config}")
    log.debug(f"Runtime config: {runtime_config}")

    # Call the perform_training function with extracted parameters
    perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
        runtime_config=runtime_config,
    )


from lightning import Trainer


def log_training_arguments(
    potential_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
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
        runtime_config: Dict[str, Any]
            config for the runtime
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

    local_cache_dir = runtime_config.get("local_cache_dir", "./")
    if local_cache_dir is None:
        log.info(f"Using default cache directory: {local_cache_dir}")
    else:
        log.info(f"Using cache directory: {local_cache_dir}")

    accelerator = training_config.get("accelerator", "cpu")
    if accelerator == "cpu":
        log.info(f"Using default accelerator: {accelerator}")
    else:
        log.info(f"Using accelerator: {accelerator}")
    nr_of_epochs = runtime_config.get("nr_of_epochs", 10)
    if nr_of_epochs == 10:
        log.info(f"Using default number of epochs: {nr_of_epochs}")
    else:
        log.info(f"Training for {nr_of_epochs} epochs")
    num_nodes = runtime_config.get("num_nodes", 1)
    if num_nodes == 1:
        log.info(f"Using default number of nodes: {num_nodes}")
    else:
        log.info(f"Training on {num_nodes} nodes")
    devices = runtime_config.get("devices", 1)
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

    splitting_strategy = training_config["splitting_strategy"]["name"]
    data_split = training_config["splitting_strategy"]["data_split"]
    log.info(f"Using splitting strategy: {splitting_strategy} with split: {data_split}")

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
    log.info(training_config["training_parameter"]["loss_parameter"])
    log.debug(
        f"""
Training {model_name} on {dataset_name}-{version_select} dataset with {accelerator}
accelerator on {num_nodes} nodes for {nr_of_epochs} epochs.
Experiments are saved to: {save_dir}/{experiment_name}.
Local cache directory: {local_cache_dir}
"""
    )


def perform_training(
    potential_config: Dict[str, Any],
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
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

    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

    from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES
    from lightning import Trainer
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.dataset.dataset import DataModule

    # NOTE --------------------------------------- NOTE #
    # FIXME TODO: move this to a dataclass and control default
    # behavior from there this current approach is hacky and error prone
    save_dir = runtime_config.get("save_dir", "lightning_logs")
    if save_dir == "lightning_logs":
        log.info(f"Saving logs to default location: {save_dir}")

    experiment_name = training_config.get("experiment_name", "exp")
    if experiment_name == "{model_name}_{dataset_name}":
        experiment_name = (
            f"{potential_config['model_name']}_{dataset_config['dataset_name']}"
        )
        training_config["experiment_name"] = (
            experiment_name  # update the save_dir in training_config
        )
    experiment_name = runtime_config.get("experiment_name", "exp")
    model_name = potential_config["model_name"]
    dataset_name = dataset_config["dataset_name"]

    log_training_arguments(
        potential_config, training_config, dataset_config, runtime_config
    )

    version_select = dataset_config.get("version_select", "latest")
    accelerator = runtime_config.get("accelerator", "cpu")
    splitting_strategy = training_config["splitting_strategy"]
    nr_of_epochs = runtime_config.get("nr_of_epochs", 10)
    num_nodes = runtime_config.get("num_nodes", 1)
    devices = runtime_config.get("devices", 1)
    batch_size = training_config.get("batch_size", 128)
    remove_self_energies = training_config.get("remove_self_energies", False)
    early_stopping_config = training_config.get("early_stopping", None)
    stochastic_weight_averaging_config = training_config.get(
        "stochastic_weight_averaging_config", None
    )
    num_workers = dataset_config.get("number_of_worker", 4)
    pin_memory = dataset_config.get("pin_memory", False)
    local_cache_dir = runtime_config.get("local_cache_dir", "./")
    # NOTE --------------------------------------- NOTE #
    # FIXME TODO: move this to a dataclass and control default
    # behavior from there this current approach is hacky and error prone

    # set up tensor board logger
    if training_config["experiment_logger"]["logger_name"].lower() == "tensorboard":
        logger = TensorBoardLogger(save_dir, name=experiment_name)
    elif training_config["experiment_logger"]["logger_name"].lower() == "wandb":
        logger = WandbLogger(save_dir=save_dir, log_model="all", name=experiment_name)

    else:
        raise ValueError(f"Unknown logger name: {training_config['logger_name']}")
    # Set up dataset
    dm = DataModule(
        name=dataset_name,
        batch_size=batch_size,
        remove_self_energies=remove_self_energies,
        version_select=version_select,
        local_cache_dir=local_cache_dir,
        splitting_strategy=REGISTERED_SPLITTING_STRATEGIES[splitting_strategy["name"]](
            seed=splitting_strategy.get("splitting_seed", 42),
            split=splitting_strategy["data_split"],
        ),
    )
    dm.prepare_data()
    dm.setup()

    # read dataset statistics
    import toml

    dataset_statistic = toml.load(dm.dataset_statistic_filename)
    log.info(
        f"Setting per_atom_energy_mean and per_atom_energy_stddev for {model_name}"
    )
    log.info(
        f"per_atom_energy_mean: {dataset_statistic['training_dataset_statistics']['per_atom_energy_mean']}"
    )
    log.info(
        f"per_atom_energy_stddev: {dataset_statistic['training_dataset_statistics']['per_atom_energy_stddev']}"
    )

    # Set up model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        dataset_statistic=dataset_statistic,
        model_parameter=potential_config,
        training_parameter=training_config["training_parameter"],
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
        monitor="val/per_molecule_energy/rmse",
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
        log_every_n_steps=50,
    )

    # Run training loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(
            num_workers=num_workers, pin_memory=pin_memory
        ),
        val_dataloaders=dm.val_dataloader(),
        ckpt_path=checkpoint_path,
    )

    trainer.validate(
        model=model, dataloaders=dm.val_dataloader(), ckpt_path="best", verbose=True
    )
    trainer.test(dataloaders=dm.test_dataloader(), ckpt_path="best", verbose=True)
    return trainer

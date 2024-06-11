from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
from typing import TYPE_CHECKING, Any, Union, Dict, NamedTuple, Tuple, Type, Mapping
import torch
from loguru import logger as log
from modelforge.dataset.dataset import BatchData
from torch.nn import functional as F

if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics
    from modelforge.potential.ani import ANI2x, AniNeuralNetworkData
    from modelforge.potential.painn import PaiNN, PaiNNNeuralNetworkData
    from modelforge.potential.physnet import PhysNet, PhysNetNeuralNetworkData
    from modelforge.potential.schnet import SchNet, SchnetNeuralNetworkData
    from modelforge.potential.sake import SAKE
    from modelforge.potential.utils import BatchData


class Loss:
    """
    Base class for loss computations, designed to be overridden by subclasses for specific types of losses.
    Initializes with a model to compute predictions for energies and forces.
    """

    def __init__(
        self, model: Union["ANI2x", "SchNet", "PaiNN", "PhysNet", "SAKE"]
    ) -> None:

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
        F_predict = -torch.autograd.grad(
            E_predict.sum(), nnp_input.positions, create_graph=False, retain_graph=True
        )[0]

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


from typing import Optional


class EnergyAndForceLoss(Loss):
    """
    Computes combined loss from energies and forces, with adjustable weighting.
    """

    def __init__(
        self,
        model: Union["ANI2x", "SchNet", "PaiNN", "PhysNet", "SAKE"],
        include_force: bool = False,
        force_weight: float = 1.0,
        energy_weight: float = 1.0,
    ) -> None:
        super().__init__(model)
        log.info("Initializing EnergyAndForceLoss")
        self.force_weight = force_weight
        self.energy_weight = energy_weight
        self.include_force = include_force
        self.have_raised_warning = False

    def compute_loss(
        self,
        batch: "BatchData",
        loss_fn: Dict[str, torch.nn.Module] = {
            "energy_loss": F.l1_loss,
            "force_loss": F.l1_loss,
        },
    ) -> torch.Tensor:
        """
        Computes the weighted combined loss for energies and optionally forces.

        Parameters
        ----------
        batch : BatchData
            The batch of data to compute the loss for.
        loss_fn : function, optional
            The PyTorch loss function to apply, by default F.l1_loss.

        Returns
        -------
        torch.Tensor
            The computed loss as a PyTorch tensor.
        """
        with torch.set_grad_enabled(True):
            energies = self._get_energies(batch)
            if self.include_force:
                forces = self._get_forces(batch, energies)
            else:
                forces = None
            loss = self._compute_loss(energies, forces, loss_fn)

        return loss

    def _compute_loss(
        self,
        energies: Dict[str, torch.Tensor],
        forces: Optional[Dict[str, torch.Tensor]],
        loss_fn: Dict[str, torch.nn.Module],
    ) -> Dict[str, torch.Tensor]:

        E_loss = self.energy_weight * loss_fn["energy_loss"](
            energies["E_predict"], energies["E_true"]
        )
        if forces is None:
            return {"combined_loss": E_loss, "energy_loss": E_loss, "force_loss": 0}

        F_loss = self.force_weight * loss_fn["force_loss"](
            forces["F_predict"], forces["F_true"]
        )

        # combined loss with weights
        combined_loss = self.energy_weight * E_loss + self.force_weight * F_loss

        return {
            "combined_loss": combined_loss,
            "energy_loss": E_loss,
            "force_loss": F_loss,
        }


from torch.optim import Optimizer

if TYPE_CHECKING:
    from modelforge.potential import _Implemented_NNPs


class TrainingAdapter(pl.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        nnp_parameters: Dict[str, Any],
        lr_scheduler_config: Dict[str, Union[str, int, float]],
        lr: float,
        include_force: bool = False,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
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
        include_force : bool, optional
            Whether to include force in the loss function, by default False.
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        """

        from typing import List
        from modelforge.potential import _Implemented_NNPs

        super().__init__()
        self.save_hyperparameters()
        # Extracting and instantiating the model from parameters
        nnp_parameters_ = nnp_parameters.copy()
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
        self.loss = EnergyAndForceLoss(model=self.model, include_force=include_force)
        self.lr_scheduler_config = lr_scheduler_config
        self.test_mse: List[float] = []
        self.val_mse: List[float] = []
        self.training_and_validation_loss_fn = {
            "energy_loss": F.mse_loss,
            "force_loss": F.mse_loss,
        }
        self.test_loss_fn = {"energy_loss": F.mse_loss, "force_loss": F.mse_loss}
        log.info(
            f"Training & validation loss fn: {[(k.__name__, v) for v, k in self.training_and_validation_loss_fn.items()]}"
        )
        log.info(
            f"Test loss fn: {[(k.__name__, v) for v, k in self.test_loss_fn.items()]}"
        )
        self.unused_parameters: bool = False

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
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
        batch_size = y.size(0)
        return batch_size

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

        loss = self.loss.compute_loss(batch, self.training_and_validation_loss_fn)
        self.log("L(E) [kJ/mol]", loss["energy_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E))
        self.log("L(F) [kJ/mol]", loss["force_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E))
        self.log("L(E+F)", loss["combined_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E))
        return loss["combined_loss"]

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
        loss = self.loss.compute_loss(batch, loss_fn=self.test_loss_fn)

        self.test_mse.append(float(loss["combined_loss"].detach()))
        self.log("L(E)_test [kJ/mol]", loss["energy_loss"], on_step=True, prog_bar=True)
        self.log("L(F)_test [kJ/mol]", loss["force_loss"], on_step=True, prog_bar=True)
        self.log(
            "L(F+E)_test [kJ/mol]", loss["combined_loss"], on_step=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """
        Calculates the root mean squared error (RMSE) of the test set and logs it to the progress bar.

        This method is called at the end of each test epoch during training. It calculates the RMSE of the test set by taking the square root of the mean of the test mean squared error (MSE) values. The RMSE is then logged to the progress bar with the key "rmse_test_loss".
        """

        import numpy as np
        if self.unused_parameters:
            log.warning(f"Unused parameters during training!\nPlease investigate if you aren't expecting this.")

        rmse_loss = np.sqrt(np.mean(np.array(self.test_mse)))
        self.log(
            "rmse_test_loss",
            rmse_loss,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: "BatchData", batch_idx: int) -> torch.Tensor:
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
        torch.Tensor
            The loss tensor computed for the current validation step.
        """
        loss = self.loss.compute_loss(
            batch, loss_fn=self.training_and_validation_loss_fn
        )
        self.log("L(E)_v [kJ/mol]", loss["energy_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E))
        self.log("L(F)_v [kJ/mol]", loss["force_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E))
        self.log(
            "L(E+F)_v [kJ/mol]", loss["combined_loss"], on_step=True, prog_bar=True, batch_size=self._log_batch_size(batch.metadata.E)
        )

        self.val_mse.append(float(loss["combined_loss"].item()))

    def on_after_backward(self):
        # Log histograms of weights and biases after each backward pass
        for name, params in self.named_parameters():
            if params is not None:
                self.logger.experiment.add_histogram(name, params, self.current_epoch)
            if params.grad is not None:
                self.logger.experiment.add_histogram(
                    f"{name}.grad", params.grad, self.current_epoch
                )

    def on_validation_epoch_end(self):
        """
        Handles end-of-validation-epoch events to compute and log the average RMSE validation loss.
        """
        import numpy as np

        rmse_loss = np.sqrt(np.mean(np.array(self.val_mse)))
        sch = self.lr_schedulers()
        try:
            log.info(f"Current learning rate: {sch.get_last_lr()}")
        except AttributeError:
            pass

        self.log(
            "rmse_val_loss", rmse_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.val_mse.clear()
        return rmse_loss

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
            "monitor": "rmse_val_loss",  # Name of the metric to monitor
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_after_backward(self) -> None:
        for name, p in self.named_parameters():
            if p.grad is None:
                unused_parameters = True
                # print(name, p)

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

        from ray.train import CheckpointConfig, RunConfig, ScalingConfig

        scaling_config = ScalingConfig(
            num_workers=number_of_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 1, "GPU": 1} if use_gpu else {"CPU": 1},
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="rmse_val_loss",
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
            metric="rmse_val_loss",
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


def return_toml_config(config_path: str):
    """
    Read a TOML configuration file and return the parsed configuration.

    Parameters
    ----------
    config_path : str
        The path to the TOML configuration file.

    Returns
    -------
    dict
        The parsed configuration from the TOML file.
    """
    import toml

    # Read the TOML file
    config = toml.load(config_path)
    log.info(f"Reading config from : {config_path}")
    return config


def read_config_and_train(config_path: str):
    """
    Reads a TOML configuration file and performs training based on the parameters.

    Parameters
    ----------
    config_path : str
        Path to the TOML configuration file.
    """
    # Read the TOML file
    config = return_toml_config(config_path)

    # Extract parameters
    # potential
    potential_config = config["potential"]

    # dataset
    dataset_config = config["dataset"]

    # training
    training_config = config["training"]

    # Call the perform_training function with extracted parameters
    perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
    )


from lightning import Trainer


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
    if experiment_name == "experiment_name":
        log.info(f"Saving logs in default dir: {experiment_name}")

    model_name = potential_config["model_name"]
    dataset_name = dataset_config["dataset_name"]

    version_select = dataset_config.get("version_select", "latest")
    if version_select == "latest":
        log.info(f"Using default dataset version: {version_select}")

    accelerator = training_config.get("accelerator", "cpu")
    if accelerator == "cpu":
        log.info(f"Using default accelerator: {accelerator}")

    nr_of_epochs = training_config.get("nr_of_epochs", 10)
    if nr_of_epochs == 10:
        log.info(f"Using default number of epochs: {nr_of_epochs}")

    num_nodes = training_config.get("num_nodes", 1)
    if num_nodes == 1:
        log.info(f"Using default number of nodes: {num_nodes}")

    devices = training_config.get("devices", 1)
    if devices == 1:
        log.info(f"Using default device index/number: {devices}")

    batch_size = training_config.get("batch_size", 128)
    if batch_size == 128:
        log.info(f"Using default batch size: {batch_size}")

    remove_self_energies = dataset_config.get("remove_self_energies", False)
    if remove_self_energies is False:
        log.info(
            f"Using default for removing self energies: Self energies are not removed"
        )
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
    pin_memory = dataset_config.get("pin_memory", False)
    if pin_memory is False:
        log.info(f"Using default value for pinned_memory: {pin_memory}")

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
        model_parameters=potential_config["potential_parameters"],
        training_parameters=training_config["training_parameters"],
    )

    # set up traininer
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks.stochastic_weight_avg import (
        StochasticWeightAveraging,
    )

    # set up trainer
    callbacks = [ModelSummary(max_depth=-1)]
    if stochastic_weight_averaging_config:
        callbacks.append(
            StochasticWeightAveraging(**stochastic_weight_averaging_config)
        )
    if early_stopping_config:
        callbacks.append(EarlyStopping(**early_stopping_config))

    trainer = Trainer(
        max_epochs=nr_of_epochs,
        num_nodes=num_nodes,
        devices=devices,
        accelerator=accelerator,
        logger=logger,  # Add the logger here
        callbacks=callbacks,
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

    # from modelforge.utils.misc import visualize_model

    # visualize_model(dm, model_name)

    # Run training loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(
            num_workers=num_workers, pin_memory=pin_memory
        ),
        val_dataloaders=dm.val_dataloader(),
    )
    trainer.validate(model, dataloaders=dm.val_dataloader())
    trainer.test(dataloaders=dm.test_dataloader())
    return trainer

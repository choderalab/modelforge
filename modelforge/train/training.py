from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as pl
from typing import TYPE_CHECKING, Any, Union, Dict, NamedTuple, Tuple, Type, Mapping
import torch
from loguru import logger as log
from modelforge.potential.utils import BatchData, NNPInput
from torch.nn import functional as F

if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics
    from modelforge.potential.ani import ANI2x, AniNeuralNetworkData
    from modelforge.potential.painn import PaiNN, PaiNNNeuralNetworkData
    from modelforge.potential.physnet import PhysNet, PhysNetNeuralNetworkData
    from modelforge.potential.schnet import SchNet, SchnetNeuralNetworkData
    from modelforge.potential.sake import SAKE


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
        self, batch: BatchData, energies: Dict[str, torch.Tensor]
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
        E_predict = energies["E_predict"]
        F_predict = -torch.autograd.grad(
            E_predict.sum(), nnp_input.positions, create_graph=False, retain_graph=False
        )[0]

        return {"F_true": F_true, "F_predict": F_predict}

    def _get_energies(self, batch: BatchData) -> Dict[str, torch.Tensor]:
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
        return {"E_true": E_true, "E_predict": E_predict}


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


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

    def compute_loss(self, batch: BatchData, loss_fn=F.l1_loss) -> torch.Tensor:
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
        energies = self._get_energies(batch)
        loss = self.energy_weight * loss_fn(energies["E_predict"], energies["E_true"])
        if self.include_force:
            forces = self._get_forces(batch, energies)
            loss += self.force_weight * loss_fn(forces["F_predict"], forces["F_true"])
        return loss


from torch.optim import Optimizer

if TYPE_CHECKING:
    from modelforge.potential import _IMPLEMENTED_NNPS


class TrainingAdapter(pl.LightningModule):
    """
    Adapter class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        nnp_parameters: Dict[str, Any],
        include_force: bool = False,
        optimizer: Type[Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        model : Union[ANI2x, SchNet, PaiNN, PhysNet, SAKE]
            The neural network potential model to be trained.
        optimizer : Type[torch.optim.Optimizer], optional
            The optimizer class to use for training, by default torch.optim.Adam.
        lr : float, optional
            The learning rate for the optimizer, by default 1e-3.
        """
        from typing import List
        from modelforge.potential import _IMPLEMENTED_NNPS

        super().__init__()
        self.save_hyperparameters()
        # Extracting and instantiating the model from parameters
        nnp_name = nnp_parameters.pop("nnp_name", None)
        if nnp_name is None:
            raise ValueError(
                "NNP name must be specified in nnp_parameters with key 'nnp_name'."
            )
        nnp_class: Type = _IMPLEMENTED_NNPS.get(nnp_name)
        if nnp_class is None:
            raise ValueError(f"Specified NNP name '{nnp_name}' is not implemented.")

        self.model = nnp_class(**nnp_parameters)
        self.optimizer = optimizer
        self.learning_rate = lr
        self.loss = EnergyAndForceLoss(model=self.model, include_force=include_force)
        self.eval_loss: List[torch.Tensor] = []

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

    def training_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
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

        loss = self.loss.compute_loss(batch, F.mse_loss)
        self.log("mse_train_loss", loss, on_step=True, batch_size=int(loss.numel()))
        return loss

    def test_step(self, batch: BatchData, batch_idx: int) -> None:
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
        loss = self.loss.compute_loss(batch, F.mse_loss) ** 0.5
        self.log(
            "rmse_test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(loss.numel()),
        )

    def validation_step(self, batch: BatchData, batch_idx: int) -> torch.Tensor:
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

        mse_loss = self.loss.compute_loss(batch, F.mse_loss) ** 0.5
        rmse_loss = mse_loss**0.5
        self.log(
            "rmse_val_loss",
            rmse_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(rmse_loss.numel()),
        )

        self.eval_loss.append(mse_loss.detach())
        return rmse_loss

    def on_validation_epoch_end(self):
        """
        Handles end-of-validation-epoch events to compute and log the average RMSE validation loss.
        """
        if self.eval_loss:
            avg_loss = torch.stack(self.eval_loss).mean()
            self.log("avg_rmse_val_loss", avg_loss**0.5, sync_dist=True)
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
            "monitor": "acu_rmse_val_loss",  # Name of the metric to monitor
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

        from lightning import Trainer
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger

        # set up tensor board logger
        logger = TensorBoardLogger("tb_logs", name="training")
        early_stopping = EarlyStopping(
            monitor="acu_rmse_val_loss", min_delta=0.05, patience=20, verbose=True
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

        from ray.train import CheckpointConfig, RunConfig, ScalingConfig

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

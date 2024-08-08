import torch

from ray import air, tune


from typing import Type
from ray.tune.schedulers import ASHAScheduler


def tune_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_samples: int = 100,
    name: str = "tune",
):
    """A function to tune a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to tune.

    dataset : torch.utils.data.Dataset
        The dataset to use for tuning.

    num_samples : int, optional
        The number of samples to use for tuning. Default is 100.

    """
    # access the model's configuration prior
    # TODO: not finalized yet
    config_prior = model._config_prior()

    def objective():
        raise NotImplementedError

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="rmse_vl",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
    )

    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        num_samples=1000,
    )

    run_config = air.RunConfig(
        name=name,
        verbose=1,
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
        param_space=config_prior,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()


class RayTuner:

    def __init__(self, model: Type[torch.nn.Module]) -> None:
        """
        Initializes the RayTuner with the given model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be tuned and trained using Ray.
        """
        self.model = model

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
        import lightning as pl

        # Configure PyTorch Lightning trainer with Ray DDP strategy
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        # Fit the model using the trainer
        trainer.fit(self.model, self.train_dataloader, self.val_dataloader)

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
        TorchTrainer
            The configured Ray Trainer for distributed training.
        """

        from ray.train import CheckpointConfig, RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        # Configure scaling for Ray Trainer
        scaling_config = ScalingConfig(
            num_workers=number_of_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 1, "GPU": 1} if use_gpu else {"CPU": 1},
        )

        # Configure run settings for Ray Trainer
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val/energy/rmse",
                checkpoint_score_order="min",
            ),
        )
        # Define and return the TorchTrainer
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
        metric: str = "val/per_molecule_energy/rmse",
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
        train_on_gpu : bool, optional
            Whether to use GPUs for training, by default False.
        metric : str, optional
            The metric to use for evaluation and early stopping, by default "val/per_molecule_energy/rmse

        Returns
        -------
        ExperimentAnalysis
            The result of the hyperparameter tuning session, containing performance metrics and the best hyperparameters found.
        """


        from ray import tune

        from ray.tune.schedulers import ASHAScheduler

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Initialize Ray Trainer
        ray_trainer = self.get_ray_trainer(
            number_of_workers=number_of_ray_workers, use_gpu=train_on_gpu
        )
        # Configure ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=number_of_epochs, grace_period=1, reduction_factor=2
        )

        # Define tuning configuration
        tune_config = tune.TuneConfig(
            metric=metric,
            mode="min",
            scheduler=scheduler,
            num_samples=number_of_samples,
        )

        # Initialize and run the tuner
        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": self.model.config_prior()},
            tune_config=tune_config,
        )
        return tuner.fit()

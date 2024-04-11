import torch
from ray import air, tune
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

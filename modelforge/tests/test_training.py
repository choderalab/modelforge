import os
import pytest

import platform

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import _Implemented_NNPs
from modelforge.dataset import _ImplementedDatasets
from modelforge.potential import NeuralNetworkPotentialFactory


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["ANI2x"])
@pytest.mark.parametrize("include_force", [False, True])
def test_train_with_lightning(model_name, dataset_name, include_force):
    """
    Test the forward pass for a given model and dataset.
    """

    from modelforge.tests.test_models import load_configs
    from modelforge.train.training import perform_training

    # read default parameters
    config = load_configs(f"{model_name}_without_ase", "qm9")


    from pytorch_lightning.loggers import TensorBoardLogger

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]

    training_config["include_force"] = include_force

    trainer = perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
    )
    # save checkpoint
    trainer.save_checkpoint("test.chp")

    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training",
        model_type=model_name,
        model_parameters=potential_config["potential_parameters"],
        training_parameters=training_config["training_parameters"],
    )
    from modelforge.train.training import TrainingAdapter

    model = TrainingAdapter.load_from_checkpoint("test.chp")
    assert type(model) is not None


@pytest.mark.parametrize("model_name", ["SchNet"])
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_loss(model_name, dataset_name, datamodule_factory):
    from loguru import logger as log

    # read default parameters
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import training_defaults

    file_path = (
        resources.files(training_defaults)
        / f"{model_name.lower()}_{dataset_name.lower()}.toml"
    )
    config = return_toml_config(file_path)
    # Extract parameters
    potential_parameters = config["potential"].get("potential_parameters", {})

    # inference model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="inference",
        model_type=model_name,
        model_parameters=potential_parameters,
    )

    dm = datamodule_factory(dataset_name=dataset_name)

    from modelforge.train.training import EnergyAndForceLoss
    import torch

    loss = EnergyAndForceLoss(model)

    r = loss.compute_loss(next(iter(dm.train_dataloader())))

    if dataset_name != "QM9":
        loss = EnergyAndForceLoss(model, include_force=True)
        r = loss.compute_loss(next(iter(dm.train_dataloader())))


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_hypterparameter_tuning_with_ray(model_name, dataset_name, datamodule_factory):
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import training_defaults

    file_path = resources.files(training_defaults) / f"{model_name.lower()}_qm9.toml"

    dm = datamodule_factory(dataset_name=dataset_name)

    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameters = config["potential"].get("potential_parameters", {})
    training_parameters = config["training"].get("training_parameters", {})
    # training model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training",
        model_type=model_name,
        model_parameters=potential_parameters,
        training_parameters=training_parameters,
    )

    model.tune_with_ray(
        train_dataloader=dm.train_dataloader(),
        val_dataloader=dm.val_dataloader(),
        number_of_ray_workers=1,
        number_of_epochs=1,
        number_of_samples=1,
    )

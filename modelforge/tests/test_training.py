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

    from modelforge.train.training import return_toml_config, perform_training
    from importlib import resources
    from modelforge.tests.data import training_defaults

    file_path = (
        resources.files(training_defaults)
        / f"{model_name.lower()}_{dataset_name.lower()}.toml"
    )
    config = return_toml_config(file_path)

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


import torch
from torch.nn import functional as F


def test_energy_loss_only():
    # test the loss
    from modelforge.train.training import MSELoss

    loss_calculator = MSELoss(include_force=False)

    predict_target = {}
    predict_target["E_predict"] = torch.tensor([[1.0], [2.0], [3.0]])
    predict_target["E_true"] = torch.tensor([[1.0], [-2.0], [3.0]])
    predict_target["F_predict"] = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    predict_target["F_true"] = torch.tensor(
        [[1.0, -2.0, -3.0], [1.0, -2.0, -3.0], [1.0, -2.0, -3.0]]
    )

    expected_loss = torch.sum(
        (predict_target["E_predict"] - predict_target["E_true"]) ** 2
    )
    loss_calculator.update(predict_target)
    loss = loss_calculator.compute()
    assert torch.allclose(
        expected_loss.double(), loss
    ), f"Expected {expected_loss.item()} but got {loss.item()}"

    # now with RMSE
    from modelforge.train.training import RMSELoss

    loss_calculator = RMSELoss(include_force=False)

    loss_calculator.update(predict_target)
    loss = loss_calculator.compute()
    assert torch.allclose(
        torch.sqrt(expected_loss.double()), loss
    ), f"Expected {expected_loss.item()} but got {loss.item()}"


def test_energy_and_force_loss():
    from modelforge.train.training import MSELoss

    loss_calculator = MSELoss(include_force=True)

    predict_target = {}
    predict_target["E_predict"] = torch.tensor([[1.0], [2.0], [3.0]])
    predict_target["E_true"] = torch.tensor([[1.0], [-2.0], [3.0]])
    predict_target["F_predict"] = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    predict_target["F_true"] = torch.tensor(
        [[1.0, -2.0, -3.0], [1.0, -2.0, -3.0], [1.0, -2.0, -3.0]]
    )

    expected_loss = (
        torch.sum((predict_target["E_predict"] - predict_target["E_true"]) ** 2)
        + torch.sum((predict_target["F_predict"] - predict_target["F_true"]) ** 2)
    ) / 2
    loss_calculator.update(predict_target)
    loss = loss_calculator.compute()
    assert torch.allclose(
        expected_loss.double(), loss
    ), f"Expected {expected_loss.item()} but got {loss.item()}"


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

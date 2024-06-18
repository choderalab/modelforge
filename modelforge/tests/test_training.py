import os
import pytest

import platform

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import _Implemented_NNPs
from modelforge.potential import NeuralNetworkPotentialFactory


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["ANI2x"])
@pytest.mark.parametrize("include_force", [False, True])
def test_train_with_lightning(model_name, dataset_name, include_force):
    """
    Test the forward pass for a given model and dataset.
    """

    from modelforge.train.training import (
        return_toml_config,
        perform_training,
        LossFactory,
    )
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
        loss_parameter=training_config["loss_parameter"],
        model_parameters=potential_config["potential_parameter"],
        training_parameters=training_config["training_parameter"],
    )
    from modelforge.train.training import TrainingAdapter

    model = TrainingAdapter.load_from_checkpoint("test.chp")
    assert type(model) is not None


import torch
from torch.nn import functional as F


@pytest.fixture
def _initialize_predict_target_dictionary():
    # initalize the test system
    predict_target = {}
    predict_target["E_predict"] = torch.tensor([[1.0], [2.0], [3.0]])
    predict_target["E_true"] = torch.tensor([[1.0], [-2.0], [3.0]])
    predict_target["F_predict"] = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    predict_target["F_true"] = torch.tensor(
        [[1.0, -2.0, -3.0], [1.0, -2.0, -3.0], [1.0, -2.0, -3.0]]
    )
    return predict_target


def test_energy_loss_only(_initialize_predict_target_dictionary):
    # test the different Loss classes
    from modelforge.train.training import NaiveEnergyAndForceLoss

    # start with NaiveEnergyAndForceLoss
    loss_calculator = NaiveEnergyAndForceLoss(include_force=False)
    predict_target = _initialize_predict_target_dictionary
    # this loss calculates validation and training error as MSE and test error as RMSE
    mse_expected_loss = torch.mean(
        (predict_target["E_predict"] - predict_target["E_true"]) ** 2
    )
    rmse_expected_loss = torch.sqrt(mse_expected_loss)

    # test loss class
    # make sure that train loss is MSE as expected
    loss = loss_calculator.calculate_train_loss(predict_target)
    assert torch.isclose(
        mse_expected_loss, loss["combined_loss"]
    ), f"Expected {mse_expected_loss.item()} but got {loss['combined_loss'].item()}"
    # make sure that no force loss is calculated
    assert torch.isclose(
        torch.tensor([0.0]), loss["force_loss"]
    ), f"Expected 0. but got {loss['force_loss'].item()}"
    loss_val = loss_calculator.calculate_val_loss(predict_target)
    # test that combined loss of train and val loss are the same
    assert torch.isclose(
        loss["combined_loss"], loss_val["combined_loss"]
    ), f"Expected equal loss. but got {loss['combined_loss'].item()} and {loss_val['combined_loss'].item()}"

    # test test loss (RMSE)
    loss = loss_calculator.calculate_test_loss(predict_target)
    assert torch.isclose(
        rmse_expected_loss, loss["combined_loss"]
    ), f"Expected {rmse_expected_loss.item()} but got {loss['combined_loss'].item()}"


def test_energy_and_force_loss(_initialize_predict_target_dictionary):
    # test the different Loss classes with differnt prediction targets (i.e. forces)
    from modelforge.train.training import NaiveEnergyAndForceLoss

    loss_calculator = NaiveEnergyAndForceLoss(include_force=True)

    predict_target = _initialize_predict_target_dictionary

    # generate the MSE and RMSE loss
    mse_expected_loss = (
        torch.mean((predict_target["E_predict"] - predict_target["E_true"]) ** 2)
        + torch.mean((predict_target["F_predict"] - predict_target["F_true"]) ** 2)
    ) / 2
    rmse_expected_loss = torch.sqrt(mse_expected_loss)

    # compare to train/val loss
    train_loss = loss_calculator.calculate_train_loss(predict_target)
    assert torch.isclose(
        train_loss["combined_loss"], mse_expected_loss
    ), f"Expected {mse_expected_loss.item()} but got {train_loss['combined_loss'].item()}"
    val_loss = loss_calculator.calculate_val_loss(predict_target)
    assert torch.isclose(
        val_loss["combined_loss"], mse_expected_loss
    ), f"Expected {mse_expected_loss.item()} but got {train_loss['combined_loss'].item()}"
    # compare to test loss
    test_loss = loss_calculator.calculate_test_loss(predict_target)
    assert torch.isclose(
        test_loss["combined_loss"], rmse_expected_loss
    ), f"Expected {rmse_expected_loss.item()} but got {test_loss['combined_loss'].item()}"


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_hypterparameter_tuning_with_ray(
    model_name,
    dataset_name,
    datamodule_factory,
):
    from modelforge.train.training import return_toml_config, LossFactory
    from importlib import resources
    from modelforge.tests.data import training_defaults

    file_path = resources.files(training_defaults) / f"{model_name.lower()}_qm9.toml"

    dm = datamodule_factory(dataset_name=dataset_name)

    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameter = config["potential"]["potential_parameter"]
    training_parameters = config["training"]["training_parameter"]
    loss_module = LossFactory.create_loss(**config["training"]["loss_parameter"])

    # training model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training",
        model_type=model_name,
        loss_module=loss_module,
        model_parameters=potential_parameter,
        training_parameters=training_parameters,
    )

    from modelforge.train.tuning import RayTuner

    ray_tuner = RayTuner(model)
    ray_tuner.tune_with_ray(
        train_dataloader=dm.train_dataloader(),
        val_dataloader=dm.val_dataloader(),
        number_of_ray_workers=1,
        number_of_epochs=1,
        number_of_samples=1,
    )

import os
import pytest

import platform

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import _Implemented_NNPs
from modelforge.potential import NeuralNetworkPotentialFactory


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize("loss_type", ["NaiveEnergyAndForceLoss", "EnergyLoss"])
def test_train_with_lightning(model_name, dataset_name, loss_type):
    """
    Test the forward pass for a given model and dataset.
    """

    from modelforge.train.training import (
        return_toml_config,
        perform_training,
    )
    from importlib import resources
    from modelforge.tests.data import training, potential, dataset

    training_path = resources.files(training) / "default.toml"
    potential_path = resources.files(potential) / f"{model_name.lower()}_defaults.toml"
    dataset_path = resources.files(dataset) / f"{dataset_name.lower()}.toml"

    config = return_toml_config(
        training_path=training_path,
        potential_path=potential_path,
        dataset_path=dataset_path,
    )

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]

    training_config["loss_parameter"]["loss_type"] = loss_type

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


def test_loss_fkt(single_batch_with_batchsize_2_with_force):
    from torch_scatter import scatter_sum

    batch = single_batch_with_batchsize_2_with_force
    E_true = batch.metadata.E
    F_true = batch.metadata.F
    F_predict = torch.randn_like(F_true)
    E_predict = torch.randn_like(E_true)

    F_scaling = torch.tensor([1.0])

    F_error_per_atom = torch.norm(F_true - F_predict, dim=1) ** 2
    F_error_per_molecule = scatter_sum(
        F_error_per_atom, batch.nnp_input.atomic_subsystem_indices.long(), 0
    )

    scale = F_scaling / (3 * batch.metadata.atomic_subsystem_counts)
    F_per_mol_scaled = F_error_per_molecule / scale


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
    from modelforge.train.training import EnergyLoss

    # initialize loss
    loss_calculator = EnergyLoss()
    predict_target = _initialize_predict_target_dictionary
    # this loss calculates validation and training error as MSE and test error as RMSE
    mse_expected_loss = torch.mean(
        (predict_target["E_predict"] - predict_target["E_true"]) ** 2
    )

    # test loss class
    # make sure that train loss is MSE as expected
    loss = loss_calculator.calculate_loss(predict_target, None)
    assert torch.isclose(
        mse_expected_loss, loss["combined_loss"]
    ), f"Expected {mse_expected_loss.item()} but got {loss['combined_loss'].item()}"


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
    loss_config = config["training"]["loss_parameter"]
    # training model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training",
        model_type=model_name,
        loss_parameter=loss_config,
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

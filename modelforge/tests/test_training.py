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

from unittest.mock import MagicMock
import torch
from torch.nn import functional as F

class MockModel:
    def forward(self, nnp_input):
        class MockOutput:
            def __init__(self, E):
                self.E = E
        return MockOutput(torch.tensor([-10.0, 0.0]))  # Mock prediction

class MockBatchData:
    class Metadata:
        def __init__(self):
            self.E = torch.tensor([[-12.0], [2.0]])  # Mock true energies
            self.F = torch.tensor([[[1.0, 2.0, 3.0]], [[-1.0, -2.0, -3.0]]])  # Mock true forces

    def __init__(self):
        self.nnp_input = MagicMock()
        self.metadata = self.Metadata()

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_batch_data():
    return MockBatchData()

def test_energy_loss_only(mock_model, mock_batch_data):
    # test the loss 
    from modelforge.train.training import EnergyAndForceLoss
    loss_calculator = EnergyAndForceLoss(model=mock_model, include_force=False)
    loss = loss_calculator.compute_loss(mock_batch_data, loss_fn={ 'energy_loss': F.l1_loss,'force_loss': F.l1_loss })
    expected_loss = torch.tensor(2.0)  # The expected L1 loss between predicted and true energies
    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss.item()} but got {loss.item()}"
    
    loss_calculator = EnergyAndForceLoss(model=mock_model, include_force=True)
    energies = {
        "E_true": torch.tensor([[-12.0], [12.0]]),
        "E_predict": torch.tensor([[-10.0], [10.0]])
    }
    loss = loss_calculator._compute_loss(energies, None, {'energy_loss': F.l1_loss})
    assert torch.isclose(loss, expected_loss), f"Expected {expected_loss.item()} but got {loss.item()}"

def test_energy_and_force_loss(mock_model):
    from modelforge.train.training import EnergyAndForceLoss
    loss_calculator = EnergyAndForceLoss(model=mock_model, include_force=False)
    energies = {
        "E_true": torch.tensor([[-12.0], [12.0]]),
        "E_predict": torch.tensor([[-10.0], [10.0]])
    }
    forces = {
        "F_true": torch.tensor([[[0.5, 0.5, 0.5]], [[-0.5, -0.5, -0.5]]]),
        "F_predict": torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    }
    loss = loss_calculator._compute_loss(energies, forces, {'energy_loss': F.mse_loss, 'force_loss': F.mse_loss})
    expected_energy_loss = F.mse_loss(energies["E_predict"], energies["E_true"])
    expected_force_loss = F.mse_loss(forces["F_predict"], forces["F_true"])
    expected_total_loss = expected_energy_loss + expected_force_loss
    assert torch.isclose(loss, expected_total_loss), f"Expected {expected_total_loss.item()} but got {loss.item()}"

@pytest.mark.parametrize("model_name", ["SchNet"])
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_loss(model_name, dataset_name, datamodule_factory):

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

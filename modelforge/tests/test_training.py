import os
import pytest

import platform

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import _Implemented_NNPs
from modelforge.potential import NeuralNetworkPotentialFactory


def load_configs(model_name: str, dataset_name: str):
    from modelforge.tests.data import (
        potential_defaults,
        training_defaults,
        dataset_defaults,
        runtime_defaults,
    )
    from importlib import resources
    from modelforge.train.training import return_toml_config

    potential_path = resources.files(potential_defaults) / f"{model_name.lower()}.toml"
    dataset_path = resources.files(dataset_defaults) / f"{dataset_name.lower()}.toml"
    training_path = resources.files(training_defaults) / "default.toml"
    runtime_path = resources.files(runtime_defaults) / "runtime.toml"
    return return_toml_config(
        potential_path=potential_path,
        dataset_path=dataset_path,
        training_path=training_path,
        runtime_path=runtime_path,
    )


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_train_with_lightning(model_name, dataset_name):
    """
    Test the forward pass for a given model and dataset.
    """

    from modelforge.train.training import perform_training

    # read default parameters
    config = load_configs(model_name, dataset_name)

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    runtime_config = config["runtime"]

    # perform training
    trainer = perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
        runtime_config=runtime_config,
    )
    # save checkpoint
    trainer.save_checkpoint("test.chp")
    # continue training
    trainer = perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
        runtime_config=runtime_config,
        checkpoint_path="test.chp",
    )


import torch


def test_error_calculation(single_batch_with_batchsize_16_with_force):
    # test the different Loss classes
    from modelforge.train.training import (
        FromPerAtomToPerMoleculeMeanSquaredError,
        PerMoleculeMeanSquaredError,
    )

    # generate data
    data = single_batch_with_batchsize_16_with_force
    true_E = data.metadata.E
    true_F = data.metadata.F

    # make predictions
    predicted_E = true_E + torch.rand_like(true_E) * 10
    predicted_F = true_F + torch.rand_like(true_F) * 10

    # test error for property with shape (nr_of_molecules, 1)
    error = PerMoleculeMeanSquaredError()
    E_error = error(predicted_E, true_E, data)

    # compare output (mean squared error)
    squared_error = (predicted_E - true_E) ** 2  
    reference_E_error = torch.mean(squared_error)
    assert torch.allclose(E_error, reference_E_error)

    # test error for property with shape (nr_of_atoms, 3)
    error = FromPerAtomToPerMoleculeMeanSquaredError()
    F_error = error(predicted_F, true_F, data)

    # compare error (mean squared error)
    squared_error = (
        torch.linalg.vector_norm(predicted_F - true_F, dim=1, keepdim=True) ** 2
    )

    per_mol_error = torch.zeros_like(data.metadata.E)
    per_mol_error.scatter_add_(
        0,
        data.nnp_input.atomic_subsystem_indices.unsqueeze(-1)
        .expand(-1, squared_error.size(1))
        .to(torch.int64),
        squared_error,
    )

    reference_F_error = torch.mean(per_mol_error)
    assert torch.allclose(F_error, reference_F_error)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping this test on GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_hypterparameter_tuning_with_ray(
    model_name,
    dataset_name,
    datamodule_factory,
):
    from modelforge.train.training import return_toml_config, LossFactory
    from importlib import resources
    from modelforge.tests.data import (
        training,
        potential_defaults,
        dataset_defaults,
        training_defaults,
    )

    config = load_configs(model_name, dataset_name)

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    runtime_config = config["runtime"]

    dm = datamodule_factory(dataset_name=dataset_name)

    # training model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_parameter=potential_config,
        training_parameter=training_config["training_parameter"],
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

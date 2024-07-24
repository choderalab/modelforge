import os
import pytest

import platform

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import _Implemented_NNPs
from modelforge.potential import NeuralNetworkPotentialFactory


def load_configs(potential_name: str, dataset_name: str):
    from modelforge.tests.data import (
        potential_defaults,
        training_defaults,
        dataset_defaults,
        runtime_defaults,
    )
    from importlib import resources
    from modelforge.train.training import return_toml_config

    potential_path = (
        resources.files(potential_defaults) / f"{potential_name.lower()}.toml"
    )
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
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_train_with_lightning(potential_name, dataset_name):
    """
    Test the forward pass for a given model and dataset.
    """

    from modelforge.train.training import perform_training

    # read default parameters
    config = load_configs(potential_name, dataset_name)

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    runtime_config = config["runtime"]

    # perform runtime_defaults
    trainer = perform_training(
        potential_config=potential_config,
        training_config=training_config,
        dataset_config=dataset_config,
        runtime_config=runtime_config,
    )
    # save checkpoint
    trainer.save_checkpoint("test.chp")
    # continue runtime_defaults
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

    # compare output (mean squared error scaled by number of atoms in the molecule)
    scale_squared_error = (
        (predicted_E - true_E) ** 2
    ) / data.metadata.atomic_subsystem_counts.unsqueeze(
        1
    )  # FIXME : fi
    reference_E_error = torch.mean(scale_squared_error)
    assert torch.allclose(E_error, reference_E_error)

    # test error for property with shape (nr_of_atoms, 3)
    error = FromPerAtomToPerMoleculeMeanSquaredError()
    F_error = error(predicted_F, true_F, data)

    # compare error (mean squared error scaled by number of atoms in the molecule)

    scaled_error = (
        torch.linalg.vector_norm(predicted_F - true_F, dim=1, keepdim=True) ** 2
    )

    per_mol_error = torch.zeros_like(data.metadata.E)
    per_mol_error.scatter_add_(
        0,
        data.nnp_input.atomic_subsystem_indices.unsqueeze(-1)
        .expand(-1, scaled_error.size(1))
        .to(torch.int64),
        scaled_error,
    )

    reference_F_error = torch.mean(
        per_mol_error / data.metadata.atomic_subsystem_counts.unsqueeze(1)
    )
    assert torch.allclose(F_error, reference_F_error)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping this test on GitHub Actions")
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_hypterparameter_tuning_with_ray(
    potential_name,
    dataset_name,
    datamodule_factory,
):
    from modelforge.train.training import return_toml_config, LossFactory
    from importlib import resources
    from modelforge.tests.data import (
        runtime_defaults,
        potential_defaults,
        dataset_defaults,
        training_defaults,
    )

    config = load_configs(potential_name, dataset_name)

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    runtime_config = config["runtime"]

    dm = datamodule_factory(dataset_name=dataset_name)

    # runtime_defaults model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_parameter=potential_config,
        training_parameter=training_config,
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

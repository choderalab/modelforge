import os
import platform

import pytest
import torch

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import NeuralNetworkPotentialFactory, _Implemented_NNPs


def load_configs_into_pydantic_models(
    potential_name: str, dataset_name: str, training_toml: str
):
    from importlib import resources

    import toml

    from modelforge.tests.data import (
        dataset_defaults,
        potential_defaults,
        runtime_defaults,
        training_defaults,
    )

    potential_path = (
        resources.files(potential_defaults) / f"{potential_name.lower()}.toml"
    )
    dataset_path = resources.files(dataset_defaults) / f"{dataset_name.lower()}.toml"
    training_path = resources.files(training_defaults) / f"{training_toml}.toml"
    runtime_path = resources.files(runtime_defaults) / "runtime.toml"

    training_config_dict = toml.load(training_path)
    dataset_config_dict = toml.load(dataset_path)
    potential_config_dict = toml.load(potential_path)
    runtime_config_dict = toml.load(runtime_path)

    potential_name = potential_config_dict["potential"]["potential_name"]

    from modelforge.potential import _Implemented_NNP_Parameters

    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )
    potential_parameters = PotentialParameters(**potential_config_dict["potential"])

    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.train.parameters import RuntimeParameters, TrainingParameters

    dataset_parameters = DatasetParameters(**dataset_config_dict["dataset"])
    training_parameters = TrainingParameters(**training_config_dict["training"])
    runtime_parameters = RuntimeParameters(**runtime_config_dict["runtime"])

    return {
        "potential": potential_parameters,
        "dataset": dataset_parameters,
        "training": training_parameters,
        "runtime": runtime_parameters,
    }


def get_trainer(potential_name: str, dataset_name: str, training_toml: str):
    config = load_configs_into_pydantic_models(
        potential_name, dataset_name, training_toml
    )

    # Extract parameters
    potential_parameter = config["potential"]
    training_parameter = config["training"]
    dataset_parameter = config["dataset"]
    runtime_parameter = config["runtime"]

    return NeuralNetworkPotentialFactory.generate_potential(
        use="training",
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )




@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", ["QM9", "SPICE2"])
@pytest.mark.parametrize("training", ["with_force", "without_force"])
def test_train_with_lightning(training, potential_name, dataset_name):
    """
    Test that we can train, save and load checkpoints.
    """
    # get correct training toml
    training_toml = "default_with_force" if training == "with_force" else "default"
    # SKIP if potential is ANI and dataset is SPICE2
    if "ANI" in potential_name and dataset_name == "SPICE2":
        pytest.skip("ANI potential is not compatible with SPICE2 dataset")

    # train potential
    get_trainer(
        potential_name, dataset_name, training_toml
    ).train_potential().save_checkpoint(
        "test.chp"
    )  # save checkpoint

    # continue training from checkpoint
    get_trainer(potential_name, dataset_name, training_toml).train_potential()

    assert False


def test_train_from_single_toml_file():
    from importlib import resources

    from modelforge.tests import data
    from modelforge.train.training import read_config_and_train

    config_path = resources.files(data) / f"config.toml"

    read_config_and_train(config_path)


def test_error_calculation(single_batch_with_batchsize):
    # test the different Loss classes
    from modelforge.train.training import (
        FromPerAtomToPerMoleculeSquaredError,
        PerMoleculeSquaredError,
    )

    # generate data
    batch = single_batch_with_batchsize(batch_size=16, dataset_name="PHALKETHOH")

    data = batch
    true_E = data.metadata.E
    true_F = data.metadata.F

    # make predictions
    predicted_E = true_E + torch.rand_like(true_E) * 10
    predicted_F = true_F + torch.rand_like(true_F) * 10

    # test error for property with shape (nr_of_molecules, 1)
    error = PerMoleculeSquaredError()
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
    error = FromPerAtomToPerMoleculeSquaredError()
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
        per_mol_error / (3 * data.metadata.atomic_subsystem_counts.unsqueeze(1))
    )
    assert torch.allclose(F_error, reference_F_error)


def test_loss(single_batch_with_batchsize):
    from modelforge.train.training import Loss

    batch = single_batch_with_batchsize(batch_size=16, dataset_name="PHALKETHOH")

    loss_porperty = ["per_molecule_energy", "per_atom_force"]
    loss_weights = {"per_molecule_energy": 0.5, "per_atom_force": 0.5}
    loss = Loss(loss_porperty, loss_weights)
    assert loss is not None

    # get trainer
    trainer = get_trainer("schnet", "QM9")
    prediction = trainer.model.calculate_predictions(batch, trainer.model.potential)

    # pass prediction through loss module
    loss_output = loss(prediction, batch)
    # let's recalculate the loss (NOTE: we scale the loss by the number of atoms)
    # --------------------------------------------- #
    # make sure that both have gradients
    assert prediction["per_molecule_energy_predict"].requires_grad
    assert prediction["per_atom_force_predict"].requires_grad

    # --------------------------------------------- #
    # first, calculate E_loss
    E_loss = torch.mean(
        (
            (
                prediction["per_molecule_energy_predict"]
                - prediction["per_molecule_energy_true"]
            ).pow(2)
        )
    )
    assert torch.allclose(loss_output["per_molecule_energy/mse"], E_loss)

    # --------------------------------------------- #
    # now calculate F_loss
    per_atom_force_squared_error = (
        (prediction["per_atom_force_predict"] - prediction["per_atom_force_true"])
        .pow(2)
        .sum(dim=1, keepdim=True)
    ).squeeze(-1)

    # # Aggregate error per molecule
    per_molecule_squared_error = torch.zeros_like(
        batch.metadata.E.squeeze(-1), dtype=per_atom_force_squared_error.dtype
    )
    per_molecule_squared_error.scatter_add_(
        0,
        batch.nnp_input.atomic_subsystem_indices.long(),
        per_atom_force_squared_error,
    )
    # divide by number of atoms
    per_molecule_squared_error = per_molecule_squared_error / (
        3 * batch.metadata.atomic_subsystem_counts
    )

    per_atom_force_mse = torch.mean(per_molecule_squared_error)
    assert torch.allclose(loss_output["per_atom_force/mse"], per_atom_force_mse)

    # --------------------------------------------- #
    # let's double check that the loss is calculated correctly
    # calculate the total loss

    assert torch.allclose(
        loss_weights["per_molecule_energy"] * loss_output["per_molecule_energy/mse"]
        + loss_weights["per_atom_force"] * loss_output["per_atom_force/mse"],
        loss_output["total_loss"].to(torch.float32),
    )


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
    config = load_configs_into_pydantic_models(potential_name, dataset_name)
    # config = load_configs_(potential_name, dataset_name)

    # Extract parameters
    potential_config = config["potential"]
    training_config = config["training"]

    dm = datamodule_factory(dataset_name=dataset_name)

    # training model
    model = NeuralNetworkPotentialFactory.generate_potential(
        use="training",
        potential_parameter=potential_config,
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
        train_on_gpu=False,
    )

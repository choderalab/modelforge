import os
import platform

import pytest
import torch

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from modelforge.potential import NeuralNetworkPotentialFactory, _Implemented_NNPs


def load_configs_into_pydantic_models(potential_name: str, dataset_name: str):
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
    training_path = resources.files(training_defaults) / f"default.toml"
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


def get_trainer(config):

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


def add_force_to_loss_parameter(config):
    """
    [training.loss_parameter]
    loss_property = ['per_molecule_energy', 'per_atom_force']
    # ------------------------------------------------------------ #
    [training.loss_parameter.weight]
    per_molecule_energy = 0.999 #NOTE: reciprocal units
    per_atom_force = 0.001

    """
    t_config = config["training"]
    t_config.loss_parameter.loss_property.append("per_atom_force")
    t_config.loss_parameter.weight["per_atom_force"] = 0.001


def add_dipole_moment_to_loss_parameter(config):
    """
    [training.loss_parameter]
    loss_property = [
        "per_molecule_energy",
        "per_atom_force",
        "dipole_moment",
        "total_charge",
    ]
    [training.loss_parameter.weight]
    per_molecule_energy = 1 #NOTE: reciprocal units
    per_atom_force = 0.1
    dipole_moment = 0.01
    total_charge = 0.01

    """
    t_config = config["training"]
    t_config.loss_parameter.loss_property.append("dipole_moment")
    t_config.loss_parameter.loss_property.append("total_charge")
    t_config.loss_parameter.weight["dipole_moment"] = 0.01
    t_config.loss_parameter.weight["total_charge"] = 0.01


def replace_per_molecule_with_per_atom_loss(config):
    t_config = config["training"]
    t_config.loss_parameter.loss_property.remove("per_molecule_energy")
    t_config.loss_parameter.loss_property.append("per_atom_energy")

    t_config.loss_parameter.weight.pop("per_molecule_energy")
    t_config.loss_parameter.weight["per_atom_energy"] = 0.999


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", ["PHALKETHOH"])
@pytest.mark.parametrize(
    "loss",
    ["energy", "energy_force", "normalized_energy_force", "energy_force_dipole_moment"],
)
def test_train_with_lightning(loss, potential_name, dataset_name):
    """
    Test that we can train, save and load checkpoints.
    """

    # SKIP if potential is ANI and dataset is SPICE2
    if "ANI" in potential_name and dataset_name == "SPICE2":
        pytest.skip("ANI potential is not compatible with SPICE2 dataset")
    if IN_GITHUB_ACTIONS and potential_name == "SAKE" and "force" in loss:
        pytest.skip(
            "Skipping Sake training with forces on GitHub Actions because it allocates too much memory"
        )

    config = load_configs_into_pydantic_models(potential_name, dataset_name)

    if "force" in loss:
        add_force_to_loss_parameter(config)
    if "normalized" in loss:
        replace_per_molecule_with_per_atom_loss(config)
    if "dipole_moment" in loss:
        add_dipole_moment_to_loss_parameter(config)

    # train potential
    get_trainer(config).train_potential().save_checkpoint("test.chp")  # save checkpoint

    # continue training from checkpoint
    get_trainer(config).train_potential()


def test_train_from_single_toml_file():
    from importlib import resources

    from modelforge.tests import data
    from modelforge.train.training import read_config_and_train

    config_path = resources.files(data) / f"config.toml"

    read_config_and_train(config_path)


def test_error_calculation(single_batch_with_batchsize):
    # test the different Loss classes
    from modelforge.train.losses import (
        ForceSquaredError,
        EnergySquaredError,
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
    error = EnergySquaredError()
    E_error = error(predicted_E, true_E, data)

    # compare output (mean squared error scaled by number of atoms in the molecule)
    scale_squared_error = (
        (predicted_E - true_E) ** 2
    ) / data.metadata.atomic_subsystem_counts.unsqueeze(
        1
    )  # FIXME : fi
    reference_E_error = torch.mean(scale_squared_error)
    assert torch.allclose(torch.mean(E_error), reference_E_error)

    # test error for property with shape (nr_of_atoms, 3)
    error = ForceSquaredError()
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
    assert torch.allclose(torch.mean(F_error), reference_F_error)


def test_loss_with_dipole_moment(single_batch_with_batchsize):

    # Generate a batch with the specified batch size and dataset
    batch = single_batch_with_batchsize(batch_size=16, dataset_name="SPICE2")

    # Get the trainer object with the specified model and dataset
    trainer = get_trainer(
        potential_name="schnet",
        dataset_name="SPICE2",
        training_toml="train_with_dipole_moment",
    )
    # Calculate predictions using the trainer's model
    prediction = trainer.model.calculate_predictions(
        batch,
        trainer.model.potential,
        train_mode=True,  # train_mode=True is required for gradients in force prediction
    )

    # Assertions for energy predictions
    assert prediction["per_molecule_energy_predict"].size(0) == batch.metadata.E.size(
        0
    ), "Mismatch in batch size for energy predictions."

    # Assertions for force predictions
    assert prediction["per_atom_force_predict"].size(0) == batch.metadata.F.size(
        0
    ), "Mismatch in number of atoms for force predictions."

    # Assertions for dipole moment predictions
    assert (
        "per_molecule_dipole_moment_predict" in prediction
    ), "Dipole moment prediction missing."
    assert (
        prediction["per_molecule_dipole_moment_predict"].size()
        == batch.metadata.dipole_moment.size()
    ), "Mismatch in shape for dipole moment predictions."

    # Assertions for total charge predictions
    assert (
        "per_molecule_total_charge_predict" in prediction
    ), "Total charge prediction missing."
    assert (
        prediction["per_molecule_total_charge_predict"].size()
        == batch.nnp_input.total_charge.size()
    ), "Mismatch in shape for total charge predictions."

    # Now compute the loss
    loss_dict = trainer.model.loss(predict_target=prediction, batch=batch)

    # Ensure that the loss contains the total_charge and dipole_moment terms
    assert "total_charge" in loss_dict, "Total charge loss not computed."
    assert "dipole_moment" in loss_dict, "Dipole moment loss not computed."

    # Check that the losses are finite numbers
    assert torch.isfinite(
        loss_dict["total_charge"]
    ).all(), "Total charge loss contains non-finite values."
    assert torch.isfinite(
        loss_dict["dipole_moment"]
    ).all(), "Dipole moment loss contains non-finite values."

    # Optionally, print or log the losses for debugging
    print("Total Charge Loss:", loss_dict["total_charge"].mean().item())
    print("Dipole Moment Loss:", loss_dict["dipole_moment"].mean().item())

    # Check that the total loss includes the new loss terms
    assert "total_loss" in loss_dict, "Total loss not computed."
    assert torch.isfinite(
        loss_dict["total_loss"]
    ).all(), "Total loss contains non-finite values."


def test_loss(single_batch_with_batchsize):
    from modelforge.train.losses import Loss

    batch = single_batch_with_batchsize(batch_size=16, dataset_name="PHALKETHOH")

    loss_porperty = ["per_molecule_energy", "per_atom_force", "per_atom_energy"]
    loss_weights = {
        "per_molecule_energy": 0.5,
        "per_atom_force": 0.5,
        "per_atom_energy": 0.1,
    }
    loss = Loss(loss_porperty, loss_weights)
    assert loss is not None

    # get trainer
    trainer = get_trainer("schnet", "QM9", "default_with_force")
    prediction = trainer.model.calculate_predictions(
        batch, trainer.model.potential, train_mode=True
    )  # train_mode=True is required for gradients in force prediction

    assert prediction["per_molecule_energy_predict"].size(
        dim=0
    ) == batch.metadata.E.size(dim=0)
    assert prediction["per_atom_force_predict"].size(dim=0) == batch.metadata.F.size(
        dim=0
    )

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
    # compare to referenc evalue obtained from Loos class
    ref = torch.mean(loss_output["per_molecule_energy"])
    assert torch.allclose(ref, E_loss)
    E_loss = torch.mean(
        (
            (
                prediction["per_molecule_energy_predict"]
                - prediction["per_molecule_energy_true"]
            ).pow(2)
            / batch.metadata.atomic_subsystem_counts.unsqueeze(1)
        )
    )
    # compare to referenc evalue obtained from Loos class
    ref = torch.mean(loss_output["per_atom_energy"])
    assert torch.allclose(ref, E_loss)

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
    assert torch.allclose(torch.mean(loss_output["per_atom_force"]), per_atom_force_mse)

    # --------------------------------------------- #
    # let's double check that the loss is calculated correctly
    # calculate the total loss

    assert torch.allclose(
        loss_weights["per_molecule_energy"] * loss_output["per_molecule_energy"]
        + loss_weights["per_atom_force"] * loss_output["per_atom_force"]
        + +loss_weights["per_atom_energy"] * loss_output["per_atom_energy"],
        loss_output["total_loss"].to(torch.float32),
    )


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping this test on GitHub Actions")
# @pytest.mark.parametrize(
#     "potential_name", _Implemented_NNPs.get_all_neural_network_names()
# )
# @pytest.mark.parametrize("dataset_name", ["QM9"])
# def test_hypterparameter_tuning_with_ray(
#     potential_name,
#     dataset_name,
#     datamodule_factory,
# ):
#     config = load_configs_into_pydantic_models(potential_name, dataset_name)
#     # config = load_configs_(potential_name, dataset_name)

#     # Extract parameters
#     potential_config = config["potential"]
#     training_config = config["training"]

#     dm = datamodule_factory(dataset_name=dataset_name)

#     # training model
#     model = NeuralNetworkPotentialFactory.generate_potential(
#         use="training",
#         potential_parameter=potential_config,
#         training_parameter=training_config,
#     )

#     from modelforge.train.tuning import RayTuner

#     ray_tuner = RayTuner(model)
#     ray_tuner.tune_with_ray(
#         train_dataloader=dm.train_dataloader(),
#         val_dataloader=dm.val_dataloader(),
#         number_of_ray_workers=1,
#         number_of_epochs=1,
#         number_of_samples=1,
#         train_on_gpu=False,
#     )

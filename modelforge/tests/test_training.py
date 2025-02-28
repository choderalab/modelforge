import os
import platform

import pytest
import torch

from modelforge.potential import NeuralNetworkPotentialFactory, _Implemented_NNPs

ON_MACOS = platform.system() == "Darwin"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_training_temp")
    return fn


def load_configs_into_pydantic_models(
    potential_name: str, dataset_name: str, local_cache_dir: str
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

    runtime_parameters.local_cache_dir = local_cache_dir
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

    return NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )


def add_force_to_loss_parameter(config):
    """
    [training.loss_parameter]
    loss_components = ['per_system_energy', 'per_atom_force']
    # ------------------------------------------------------------ #
    [training.loss_parameter.weight]
    per_system_energy = 0.999 #NOTE: reciprocal units
    per_atom_force = 0.001

    """
    t_config = config["training"]
    t_config.loss_parameter.loss_components.append("per_atom_force")
    t_config.loss_parameter.weight["per_atom_force"] = 0.001
    t_config.loss_parameter.target_weight["per_atom_force"] = 0.001
    t_config.loss_parameter.mixing_steps["per_atom_force"] = 1


def add_dipole_moment_to_loss_parameter(config):
    """
    [training.loss_parameter]
    loss_components = [
        "per_system_energy",
        "per_atom_force",
        "per_system_dipole_moment",
        "per_system_total_charge",
    ]
    [training.loss_parameter.weight]
    per_system_energy = 1 #NOTE: reciprocal units
    per_atom_force = 0.1
    per_system_dipole_moment = 0.01
    per_system_total_charge = 0.01

    """
    t_config = config["training"]
    t_config.loss_parameter.loss_components.append("per_system_dipole_moment")
    t_config.loss_parameter.loss_components.append("per_system_total_charge")
    t_config.loss_parameter.weight["per_system_dipole_moment"] = 0.01
    t_config.loss_parameter.weight["per_system_total_charge"] = 0.01

    t_config.loss_parameter.target_weight["per_system_dipole_moment"] = 0.01
    t_config.loss_parameter.target_weight["per_system_total_charge"] = 0.01

    t_config.loss_parameter.mixing_steps["per_system_dipole_moment"] = 1
    t_config.loss_parameter.mixing_steps["per_system_total_charge"] = 1

    # also add per_atom_charge to predicted properties

    p_config = config["potential"]
    p_config.core_parameter.predicted_properties.append("per_atom_charge")
    p_config.core_parameter.predicted_dim.append(1)


def replace_per_system_with_per_atom_loss(config):
    t_config = config["training"]
    t_config.loss_parameter.loss_components.remove("per_system_energy")
    t_config.loss_parameter.loss_components.append("per_atom_energy")

    t_config.loss_parameter.weight.pop("per_system_energy")
    t_config.loss_parameter.weight["per_atom_energy"] = 0.999

    t_config.loss_parameter.target_weight.pop("per_system_energy")
    t_config.loss_parameter.target_weight["per_atom_energy"] = 0.999

    t_config.loss_parameter.mixing_steps.pop("per_system_energy")
    t_config.loss_parameter.mixing_steps["per_atom_energy"] = 1

    # NOTE: the loss is calculate per_atom, but the validation set error is
    # per_system. This is because otherwise it's difficult to compare.
    t_config.early_stopping.monitor = "val/per_system_energy/rmse"
    t_config.monitor = "val/per_system_energy/rmse"
    t_config.lr_scheduler.monitor = "val/per_system_energy/rmse"


from typing import Literal


from typing import Literal
from modelforge.train.parameters import (
    TrainingParameters,
    ReduceLROnPlateauConfig,
    CosineAnnealingLRConfig,
    CosineAnnealingWarmRestartsConfig,
    CyclicLRConfig,
    OneCycleLRConfig,
)

from typing import Literal


def use_different_LRScheduler(
    training_config: TrainingParameters,
    which_one: Literal[
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "CyclicLR",
    ],
) -> TrainingParameters:
    """
    Modifies the training configuration to use a different learning rate scheduler.
    """

    if which_one == "ReduceLROnPlateau":
        lr_scheduler_config = ReduceLROnPlateauConfig(
            scheduler_name="ReduceLROnPlateau",
            frequency=1,
            interval="epoch",
            monitor=training_config.monitor,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=0.1,
            threshold_mode="abs",
            cooldown=5,
            min_lr=1e-8,
            eps=1e-8,
        )
    elif which_one == "CosineAnnealingLR":
        lr_scheduler_config = CosineAnnealingLRConfig(
            scheduler_name="CosineAnnealingLR",
            frequency=1,
            interval="epoch",
            monitor=training_config.monitor,
            T_max=training_config.number_of_epochs,
            eta_min=0.0,
            last_epoch=-1,
        )
    elif which_one == "CosineAnnealingWarmRestarts":
        lr_scheduler_config = CosineAnnealingWarmRestartsConfig(
            scheduler_name="CosineAnnealingWarmRestarts",
            frequency=1,
            interval="epoch",
            monitor=training_config.monitor,
            T_0=10,
            T_mult=2,
            eta_min=0.0,
            last_epoch=-1,
        )
    elif which_one == "OneCycleLR":
        lr_scheduler_config = OneCycleLRConfig(
            scheduler_name="OneCycleLR",
            frequency=1,
            interval="step",
            monitor=None,
            max_lr=training_config.lr,
            epochs=training_config.number_of_epochs,  # Use epochs from training config
            # steps_per_epoch will be calculated at runtime
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4,
            three_phase=False,
            last_epoch=-1,
        )
    elif which_one == "CyclicLR":
        lr_scheduler_config = CyclicLRConfig(
            scheduler_name="CyclicLR",
            frequency=1,
            interval="step",
            monitor=None,
            base_lr=training_config.lr / 10,
            max_lr=training_config.lr,
            epochs_up=1.0,  # For example, increasing phase lasts 1 epoch
            epochs_down=1.0,  # Decreasing phase lasts 1 epoch
            mode="triangular",
            gamma=1.0,
            scale_mode="cycle",
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {which_one}")

    # Update the lr_scheduler in the training configuration
    training_config.lr_scheduler = lr_scheduler_config
    return training_config


import pytest
from modelforge.train.parameters import TrainingParameters


@pytest.mark.parametrize("potential_name", ["ANI2x"])
@pytest.mark.parametrize("dataset_name", ["PHALKETHOH"])
@pytest.mark.parametrize(
    "lr_scheduler",
    [
        "ReduceLROnPlateau",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "CyclicLR",
    ],
)
def test_learning_rate_scheduler(
    potential_name,
    dataset_name,
    lr_scheduler,
    prep_temp_dir,
):
    """
    Test that we can train, save, and load checkpoints with different learning rate schedulers.
    """
    # Load the configuration into Pydantic models
    config = load_configs_into_pydantic_models(
        potential_name, dataset_name, str(prep_temp_dir)
    )

    # Get the training configuration
    training_config = config["training"]
    # Modify the training configuration to use the selected scheduler
    training_config = use_different_LRScheduler(training_config, lr_scheduler)

    config["training"] = training_config
    # Proceed with training
    get_trainer(config).train_potential().save_checkpoint("test.chp")  # save checkpoint


@pytest.mark.xdist_group(name="test_training_with_lightning")
@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", ["PHALKETHOH"])
@pytest.mark.parametrize(
    "loss",
    ["energy", "energy_force", "normalized_energy_force", "energy_force_dipole_moment"],
)
def test_train_with_lightning(loss, potential_name, dataset_name, prep_temp_dir):
    """
    Test that we can train, save and load checkpoints.
    """

    # SKIP if potential is ANI and dataset is SPICE2
    if "ANI" in potential_name and dataset_name == "SPICE2":
        pytest.skip("ANI potential is not compatible with SPICE2 dataset")
    if IN_GITHUB_ACTIONS and potential_name == "SAKE" and "force" in loss:
        pytest.skip(
            "Skipping Sake training with forces because it allocates too much memory"
        )

    config = load_configs_into_pydantic_models(
        potential_name, dataset_name, str(prep_temp_dir)
    )

    if "force" in loss:
        add_force_to_loss_parameter(config)
    if "normalized" in loss:
        replace_per_system_with_per_atom_loss(config)
    if "dipole_moment" in loss:
        add_dipole_moment_to_loss_parameter(config)

    # train potential
    get_trainer(config).train_potential().save_checkpoint("test.chp")  # save checkpoint
    # continue training from checkpoint
    # get_trainer(config).train_potential()


def test_train_from_single_toml_file(prep_temp_dir):
    from importlib import resources

    from modelforge.tests import data
    from modelforge.train.training import read_config_and_train

    config_path = resources.files(data) / f"config.toml"

    read_config_and_train(config_path, local_cache_dir=str(prep_temp_dir))


def test_train_from_single_toml_file_element_filter(prep_temp_dir):
    from importlib import resources

    from modelforge.tests import data
    from modelforge.train.training import read_config_and_train

    config_path = resources.files(data) / f"config.toml"

    from modelforge.potential.potential import NeuralNetworkPotentialFactory
    from modelforge.train.training import read_config

    (
        training_parameter,
        dataset_parameter,
        potential_parameter,
        runtime_parameter,
    ) = read_config(
        condensed_config_path=config_path,
    )

    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )
    import numpy as np

    assert np.all(np.array(trainer.datamodule.element_filter) == np.array([[6, 1]]))


def test_error_calculation(single_batch_with_batchsize, prep_temp_dir):
    # test the different Loss classes
    from modelforge.train.losses import (
        ForceSquaredError,
        EnergySquaredError,
    )

    # generate data
    batch = single_batch_with_batchsize(
        batch_size=16,
        dataset_name="PHALKETHOH",
        local_cache_dir=str(prep_temp_dir),
        version_select="nc_1000_v0",
    )

    data = batch
    true_E = data.metadata.per_system_energy
    true_F = data.metadata.per_atom_force

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

    per_mol_error = torch.zeros_like(data.metadata.per_system_energy)
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


def test_loss_with_dipole_moment(single_batch_with_batchsize, prep_temp_dir):
    # Generate a batch with the specified batch size and dataset
    batch = single_batch_with_batchsize(
        batch_size=16,
        dataset_name="SPICE2",
        local_cache_dir=str(prep_temp_dir),
        version_select="nc_1000_v0",
    )

    # Get the trainer object with the specified model and dataset
    config = load_configs_into_pydantic_models(
        potential_name="schnet",
        dataset_name="SPICE2",
        local_cache_dir=str(prep_temp_dir),
    )
    add_dipole_moment_to_loss_parameter(config)
    add_force_to_loss_parameter(config)

    trainer = get_trainer(
        config,
    )

    # Calculate predictions using the trainer's model
    prediction = trainer.lightning_module.calculate_predictions(
        batch,
        trainer.lightning_module.potential,
        train_mode=True,  # train_mode=True is required for gradients in force prediction
    )

    # Assertions for energy predictions
    assert prediction["per_system_energy_predict"].size(
        0
    ) == batch.metadata.per_system_energy.size(
        0
    ), "Mismatch in batch size for energy predictions."

    # Assertions for force predictions
    assert prediction["per_atom_force_predict"].size(
        0
    ) == batch.metadata.per_atom_force.size(
        0
    ), "Mismatch in number of atoms for force predictions."

    # Assertions for dipole moment predictions
    assert (
        "per_system_dipole_moment_predict" in prediction
    ), "Dipole moment prediction missing."
    assert (
        prediction["per_system_dipole_moment_predict"].size()
        == batch.metadata.per_system_dipole_moment.size()
    ), "Mismatch in shape for dipole moment predictions."

    # Assertions for total charge predictions
    assert (
        "per_system_total_charge_predict" in prediction
    ), "Total charge prediction missing."
    assert (
        prediction["per_system_total_charge_predict"].size()
        == batch.nnp_input.per_system_total_charge.size()
    ), "Mismatch in shape for total charge predictions."

    # Now compute the loss
    loss_dict = trainer.lightning_module.loss(
        predict_target=prediction,
        batch=batch,
        epoch_idx=0,
    )

    # Ensure that the loss contains the total_charge and dipole_moment terms
    assert "per_system_total_charge" in loss_dict, "Total charge loss not computed."
    assert "per_system_dipole_moment" in loss_dict, "Dipole moment loss not computed."

    # Check that the losses are finite numbers
    assert torch.isfinite(
        loss_dict["per_system_total_charge"]
    ).all(), "Total charge loss contains non-finite values."
    assert torch.isfinite(
        loss_dict["per_system_dipole_moment"]
    ).all(), "Dipole moment loss contains non-finite values."

    # Optionally, print or log the losses for debugging
    print("Total Charge Loss:", loss_dict["per_system_total_charge"].mean().item())
    print("Dipole Moment Loss:", loss_dict["per_system_dipole_moment"].mean().item())

    # Check that the total loss includes the new loss terms
    assert "total_loss" in loss_dict, "Total loss not computed."
    assert torch.isfinite(
        loss_dict["total_loss"]
    ).all(), "Total loss contains non-finite values."


def test_loss(single_batch_with_batchsize, prep_temp_dir):
    from modelforge.train.losses import Loss

    batch = single_batch_with_batchsize(
        batch_size=16,
        dataset_name="PHALKETHOH",
        local_cache_dir=str(prep_temp_dir),
        version_select="nc_1000_v0",
    )

    loss_porperty = ["per_system_energy", "per_atom_force", "per_atom_energy"]
    loss_weights = {
        "per_system_energy": torch.tensor([0.5]),
        "per_atom_force": torch.tensor([0.5]),
        "per_atom_energy": torch.tensor([0.1]),
    }
    loss = Loss(loss_porperty, loss_weights)
    assert loss is not None

    # Get the trainer object with the specified model and dataset
    config = load_configs_into_pydantic_models(
        potential_name="schnet", dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )
    add_force_to_loss_parameter(config)

    trainer = get_trainer(
        config,
    )
    prediction = trainer.lightning_module.calculate_predictions(
        batch, trainer.lightning_module.potential, train_mode=True
    )  # train_mode=True is required for gradients in force prediction

    assert prediction["per_system_energy_predict"].size(
        dim=0
    ) == batch.metadata.per_system_energy.size(dim=0)
    assert prediction["per_atom_force_predict"].size(
        dim=0
    ) == batch.metadata.per_atom_force.size(dim=0)

    # pass prediction through loss module
    loss_output = loss(prediction, batch, epoch_idx=0)
    # let's recalculate the loss (NOTE: we scale the loss by the number of atoms)
    # --------------------------------------------- #
    # make sure that both have gradients
    assert prediction["per_system_energy_predict"].requires_grad
    assert prediction["per_atom_force_predict"].requires_grad

    # --------------------------------------------- #
    # first, calculate E_loss
    E_loss = torch.mean(
        (
            (
                prediction["per_system_energy_predict"]
                - prediction["per_system_energy_true"]
            ).pow(2)
        )
    )
    # compare to reference evalue obtained from Loos class
    ref = torch.mean(loss_output["per_system_energy"])
    assert torch.allclose(ref, E_loss)
    E_loss = torch.mean(
        (
            (
                prediction["per_system_energy_predict"]
                - prediction["per_system_energy_true"]
            ).pow(2)
            / batch.metadata.atomic_subsystem_counts.unsqueeze(1)
        )
    )
    # compare to reference evalue obtained from Loos class
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
    per_system_squared_error = torch.zeros_like(
        batch.metadata.per_system_energy.squeeze(-1),
        dtype=per_atom_force_squared_error.dtype,
    )
    per_system_squared_error.scatter_add_(
        0,
        batch.nnp_input.atomic_subsystem_indices.long(),
        per_atom_force_squared_error,
    )
    # divide by number of atoms
    per_system_squared_error = per_system_squared_error / (
        3 * batch.metadata.atomic_subsystem_counts
    )

    per_atom_force_mse = torch.mean(per_system_squared_error)
    assert torch.allclose(torch.mean(loss_output["per_atom_force"]), per_atom_force_mse)

    # --------------------------------------------- #
    # let's double check that the loss is calculated correctly
    # calculate the total loss

    assert torch.allclose(
        loss_weights["per_system_energy"] * loss_output["per_system_energy"]
        + loss_weights["per_atom_force"] * loss_output["per_atom_force"]
        + +loss_weights["per_atom_energy"] * loss_output["per_atom_energy"],
        loss_output["total_loss"].to(torch.float32),
    )

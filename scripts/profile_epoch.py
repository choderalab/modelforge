import torch
from modelforge.train.training import return_toml_config
from typing import Dict, Any


def perform_training(trainer, model, dm):
    # Run runtime_defaults loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )


def setup(potential_name: str):
    from modelforge.dataset.utils import RandomRecordSplittingStrategy
    from lightning import Trainer
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.dataset.dataset import DataModule
    from importlib import resources
    from modelforge import tests as modelforge_tests

    config = return_toml_config(
        f"{resources.files(modelforge_tests)}/data/training_defaults/{potential_name.lower()}_qm9.toml"
    )
    # Extract parameters
    potential_config = config["potential"]
    training_config = config["runtime_defaults"]
    dataset_config = config["dataset"]
    training_config["nr_of_epochs"] = 1

    dataset_config["version_select"] = "nc_1000_v0"

    potential_name = potential_config["potential_name"]
    dataset_name = dataset_config["dataset_name"]
    version_select = dataset_config.get("version_select", "latest")
    accelerator = training_config.get("accelerator", "cpu")
    nr_of_epochs = training_config.get("nr_of_epochs", 1)
    num_nodes = training_config.get("num_nodes", 1)
    devices = training_config.get("devices", 1)
    batch_size = training_config.get("batch_size", 128)
    remove_self_energies = dataset_config.get("remove_self_energies", False)

    # Set up dataset
    dm = DataModule(
        name=dataset_name,
        batch_size=batch_size,
        splitting_strategy=RandomRecordSplittingStrategy(),
        remove_self_energies=remove_self_energies,
        version_select=version_select,
    )

    trainer = Trainer(
        max_epochs=nr_of_epochs,
        num_nodes=num_nodes,
        devices=devices,
        accelerator=accelerator,
        profiler="pytorch",  # "advanced",
    )

    dm.prepare_data()
    dm.setup()

    # Set up model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="runtime_defaults",
        potential_name=potential_name,
        model_parameters=potential_config["potential"],
        training_parameters=training_config["training"],
    )
    return trainer, model, dm


if __name__ == "__main__":
    potential_name = "SchNet"

    trainer, model, dm = setup(
        potential_name=potential_name,
    )

    perform_training(trainer, model, dm)

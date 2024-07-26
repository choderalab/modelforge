import torch
from typing import Dict, Any


def perform_training(trainer, model, dm):

    # Run training loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )


def setup(model_name: str):
    from modelforge.dataset.utils import RandomRecordSplittingStrategy
    from lightning import Trainer
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.dataset.dataset import DataModule
    from modelforge.tests.test_models import load_configs

    config = load_configs(f"{model_name.lower()}", "qm9")
    accelerator = "gpu"
    nr_of_epochs = 4
    num_nodes = 1
    devices = 1
    batch_size = 64

    # Set up dataset
    dm = DataModule(
        name="qm9",
        batch_size=batch_size,
        splitting_strategy=RandomRecordSplittingStrategy(),
        remove_self_energies=True,
    )
    import toml

    dm.prepare_data()
    dm.setup()

    dataset_statistic = toml.load(dm.dataset_statistic_filename)

    from lightning.pytorch.profilers import PyTorchProfiler

    profiler = PyTorchProfiler()
    trainer = Trainer(
        max_epochs=nr_of_epochs,
        num_nodes=num_nodes,
        devices=devices,
        accelerator=accelerator,
        num_sanity_val_steps=0,
        profiler=profiler,
        limit_train_batches=0.25,
    )

    # Set up model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_parameter=config["potential"],
        training_parameter=config["training"]["training_parameter"],
        dataset_statistic=dataset_statistic,
    )
    return trainer, model, dm


if __name__ == "__main__":

    # This profile script works with the
    # potential and training toml files that
    # are stored in the modelforge/tests/test_models

    model_name = "SAKE"

    trainer, model, dm = setup(
        model_name=model_name,
    )

    perform_training(trainer, model, dm)

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
@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize("include_force", [False, True])
def test_train_with_lightning(model_name, dataset_name, include_force):
    """
    Test the forward pass for a given model and dataset.
    """

    from lightning import Trainer
    from modelforge.train.training import TrainingAdapter
    from modelforge.dataset.dataset import DataModule

    dm = DataModule(
        name=dataset_name,
        batch_size=512,
        remove_self_energies=True,
        for_unit_testing=True,
    )
    dm.prepare_data()
    dm.setup()
    # Set up model
    training_parameters = {"include_force": include_force}
    model = NeuralNetworkPotentialFactory.create_nnp(
        "training", model_name, training_parameters=training_parameters
    )

    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=2)

    # Run training loop and validate
    trainer.fit(
        model,
        dm.train_dataloader(),
        dm.val_dataloader(),
    )
    # save checkpoint
    trainer.save_checkpoint("test.chp")
    model = TrainingAdapter.load_from_checkpoint("test.chp")
    assert type(model) is not None


@pytest.mark.parametrize("model_name", ["SchNet"])
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_loss(model_name, dataset_name, datamodule_factory):
    from loguru import logger as log

    dm = datamodule_factory(dataset_name=dataset_name)
    model = NeuralNetworkPotentialFactory.create_nnp("inference", model_name)

    from modelforge.train.training import EnergyAndForceLoss
    import torch

    loss = EnergyAndForceLoss(model)

    try:
        r = loss.compute_loss(next(iter(dm.train_dataloader())))
    except IndexError as excinfo:
        log.warning(f"IndexError raised: {excinfo}")
    loss = EnergyAndForceLoss(model, include_force=True)

    r = loss.compute_loss(next(iter(dm.train_dataloader())))


@pytest.mark.skipif(ON_MACOS, reason="Skipping this test on MacOS GitHub Actions")
@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_hypterparameter_tuning_with_ray(model_name, dataset_name, datamodule_factory):

    dm = datamodule_factory(dataset_name=dataset_name)
    model = NeuralNetworkPotentialFactory.create_nnp("training", model_name)

    model.tune_with_ray(
        train_dataloader=dm.train_dataloader(),
        val_dataloader=dm.val_dataloader(),
        number_of_ray_workers=1,
        number_of_epochs=1,
        number_of_samples=1,
    )

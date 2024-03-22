from typing import Type

import pytest

from modelforge.potential.models import BaseNeuralNetworkPotential

from .helper_functions import (
    MODELS_TO_TEST,
    DATASETS,
    initialize_dataset,
)


@pytest.mark.parametrize("default_model", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_train_with_lightning(
    dataset: Type[BaseNeuralNetworkPotential],
    default_model: Type[BaseNeuralNetworkPotential],
):
    """
    Test the forward pass for a given model and dataset.

    Parameters
    ----------
    dataset : Type[BaseNNP]
        The dataset class to be used in the test.
    model_class : Type[BaseNNP]
        The model class to be used in the test.
    """

    from lightning import Trainer
    import torch

    # Initialize model
    model = default_model()

    if model is None:
        pytest.fail("Failed to set up the model.")

    # Initialize dataset and data loader
    dataset = initialize_dataset(dataset)
    if dataset is None:
        pytest.fail("Failed to initialize the dataset.")
    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=2)

    # Move model to the appropriate dtype and device
    model = model.to(torch.float32)
    # Run training loop and validate
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())


def test_pt_lightning():
    # This is an example script that trains the PaiNN model on the QM9 dataset.
    from lightning import Trainer
    import torch
    from modelforge.potential.schnet import SchNet

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # Set up dataset
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=128, split=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=True)

    # Set up model
    model = SchNet()  # PaiNN() # SchNET()
    model = model.to(torch.float32)

    # set up traininer

    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
    )

    # set scaling and ase values

    dataset.dataset_statistics.scaling_stddev = 1.0
    model.dataset_statistics = dataset.dataset_statistics

    # Run training loop and validate
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

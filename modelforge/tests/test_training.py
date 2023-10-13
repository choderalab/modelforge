from typing import Optional, Type

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import SchNET

from .helper_functinos import (
    MODELS_TO_TEST,
    DATASETS,
    initialize_dataset,
    setup_simple_model,
)


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(dataset: Type[BaseNNP], model_class: Type[BaseNNP]):
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

    model: Optional[BaseNNP] = setup_simple_model(model_class, lightning=True)
    if model is None:
        pytest.fail("Failed to set up the model.")

    # Initialize dataset and data loader
    dataset = initialize_dataset(dataset, mode="fit")
    if dataset is None:
        pytest.fail("Failed to initialize the dataset.")
    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=2)

    # Move model to the appropriate dtype and device
    model = model.to(torch.float32)
    # Run training loop and validate
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

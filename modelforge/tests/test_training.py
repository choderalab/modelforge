from typing import Type

import pytest


def test_train_with_lightning(train_model, initialized_dataset):
    """
    Test the forward pass for a given model and dataset.

    Parameters
    ----------
    dataset : Type[BaseNNP]
        The dataset class to be used in the test.
    model : str
        The model to be used in the test.
    """

    from lightning import Trainer
    import torch

    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=2)

    # Move model to the appropriate dtype and device
    model = train_model.to(torch.float32)
    # Run training loop and validate
    trainer.fit(
        model,
        initialized_dataset.train_dataloader(),
        initialized_dataset.val_dataloader(),
    )


def test_hypterparameter_tuning_with_ray(train_model, initialized_dataset):

    from lightning import Trainer
    import torch

    train_model.train_model_with_ray(
        num_workers=2,
        train_dataloader=initialized_dataset.train_dataloader(),
        val_dataloader=initialized_dataset.val_dataloader(),
    )


from typing import Optional

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import Schnet

from .helper_functinos import (
    MODELS_TO_TEST,
    DATASETS,
    initialize_dataset,
    setup_simple_model,
)


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(dataset, model_class):
    from lightning import Trainer

    model = setup_simple_model(model_class)
    dataset = initialize_dataset(dataset, mode="fit")
    trainer = Trainer()
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

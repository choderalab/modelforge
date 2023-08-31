from typing import Optional

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import Schnet
from modelforge.utils import Inputs

from .helper_functinos import initialize_dataset

MODELS_TO_TEST = [Schnet]


def setup_simple_model(model_class) -> Optional[BaseNNP]:
    if model_class is Schnet:
        return Schnet(n_atom_basis=128, n_interactions=3, n_filters=64)
    else:
        raise NotImplementedError


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_forward_pass(model_class):
    from lightning import Trainer

    model = setup_simple_model(model_class)
    dataset = initialize_dataset()
    trainer = Trainer()
    trainer.fit(model, dataset.train_dataloader, dataset.val_dataloader)

    # for inp in inputs
    # output = initialized_model.forward(inputs)

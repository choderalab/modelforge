from typing import Optional

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import Schnet

from .helper_functinos import (
    DATASETS,
    MODELS_TO_TEST,
    setup_simple_model,
    return_single_batch,
)


def test_BaseNNP():
    nnp = BaseNNP()


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    initialized_model = setup_simple_model(model_class)
    inputs = return_single_batch(dataset, mode="fit")
    output = initialized_model(inputs)
    print(output.energies.shape)
    assert output.species.shape[0] == inputs["Z"].shape[0]
    assert output.energies.shape[0] == inputs["Z"].shape[0]

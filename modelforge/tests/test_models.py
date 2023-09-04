from typing import Optional

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import Schnet

from .helper_functinos import (
    DATASETS,
    MODELS_TO_TEST,
    setup_simple_model,
    single_default_input,
)


def test_BaseNNP():
    nnp = BaseNNP()


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    initialized_model = setup_simple_model(model_class)
    inputs = single_default_input(dataset, mode="fit")
    output = initialized_model(inputs)

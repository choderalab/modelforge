import pytest

from modelforge.potential.models import BaseNNP

from .helper_functinos import (
    DATASETS,
    MODELS_TO_TEST,
    return_single_batch,
    setup_simple_model,
)


def test_BaseNNP():
    nnp = BaseNNP()


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    initialized_model = setup_simple_model(model_class)
    inputs = return_single_batch(dataset, mode="fit")
    output = initialized_model(inputs)
    print(output.shape)
    assert output.shape[0] == 64
    assert output.shape[1] == 1

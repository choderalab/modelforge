import pytest
import torch

from modelforge.potential.aimnet2 import AimNet2
from modelforge.tests.test_models import load_configs


def test_initialize_model():
    """Test initialization of the Schnet model."""

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AimNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet2 model should be initialized."


@pytest.mark.xfail(raises=NotImplementedError)
def test_rbf():
    raise NotImplementedError


@pytest.mark.xfail(raises=AttributeError)
def test_forward(single_batch_with_batchsize_64):
    """Test initialization of the Schnet model."""
    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AimNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet model should be initialized."

    y_hat = aimnet(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

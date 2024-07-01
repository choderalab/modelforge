from modelforge.potential.schnet import SchNet

import pytest


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    from modelforge.potential.schnet import SchNet

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    schnet = SchNet(**potential_parameter)
    assert schnet is not None, "Schnet model should be initialized."

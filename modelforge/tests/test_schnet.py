from modelforge.potential.schnet import SchNet

import pytest


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    from modelforge.potential.schnet import SchNet

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameterss = config["potential"].get("potential_parameterss", {})
    schnet = SchNet(**potential_parameterss)
    assert schnet is not None, "Schnet model should be initialized."

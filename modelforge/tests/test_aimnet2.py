def test_initialize_model():
    """Test initialization of the Schnet model."""
    from modelforge.potential.aimnet2 import AimNet2

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AimNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet2 model should be initialized."


def test_rbf():
    pass


def test_forward(single_batch_with_batchsize_64):
    """Test initialization of the Schnet model."""
    from modelforge.potential.aimnet2 import AimNet2

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AimNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet model should be initialized."
    import torch

    y_hat = aimnet(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


def test_against_original_implementation():
    pass

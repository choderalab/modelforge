import os
import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_physnet_init():

    from modelforge.potential.physnet import PhysNet
    from importlib import resources
    from modelforge.tests.data import potential_defaults

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"physnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    model = PhysNet(**potential_parameter)


def test_physnet_forward(single_batch_with_batchsize_64):
    import torch
    from modelforge.potential.physnet import PhysNet

    # read default parameters
    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"physnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    # Extract parameters
    potential_parameter["number_of_modules"] = 1
    potential_parameter["number_of_interaction_residual"] = 1

    model = PhysNet(**potential_parameter)
    model = model.to(torch.float32)
    print(model)
    yhat = model(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test fails on macOS")
def test_rbf():
    # This test compares the RBF calculation of the original
    # PhysNet implemntation agains the SAKE/PhysNet implementation in modelforge
    # NOTE: input in PhysNet is expected in angstrom, in contrast to modelforge which
    # expects input in nanomter

    import tensorflow as tf
    import numpy as np
    import torch
    from openff.units import unit

    # set up test parameters
    number_of_radial_basis_functions = K = 20
    _max_distance_in_nanometer = 0.5
    cutoff_in_angstrom = 5
    _min_distance_in_nanometer = 0.0

    #############################
    # RBF comparision
    #############################
    # Initialize the rbf class
    from modelforge.potential.utils import PhysNetRadialSymmetryFunction

    mf_rbf = PhysNetRadialSymmetryFunction(
        number_of_radial_basis_functions,
        max_distance=_max_distance_in_nanometer * unit.nanometer,
    )

    # compare calculation of distribution scaling factor
    #################
    # PhysNet implementation
    def softplus_inverse(x):
        return x + np.log(-np.expm1(-x))

    pn_widths = [
        softplus_inverse((0.5 / ((1.0 - np.exp(-cutoff_in_angstrom)) / K)) ** 2)
    ] * K
    pn_widths = tf.nn.softplus(
        tf.Variable(np.asarray(pn_widths), name="widths", dtype=tf.float32)
    )
    pn_widths_np = pn_widths.numpy()

    # Modelforge implementation
    mf_widths_np = mf_rbf.get_buffer("radial_scale_factor").numpy()
    assert np.allclose(pn_widths_np, mf_widths_np)

    # center_position
    #################
    # PhysNet implementation
    centers = softplus_inverse(np.linspace(1.0, np.exp(-cutoff_in_angstrom), K))
    _centers = tf.nn.softplus(
        tf.Variable(np.asarray(centers), name="centers", dtype=tf.float32)
    )
    pn_centers = _centers.numpy()

    start_value = torch.exp(
        torch.scalar_tensor(-_max_distance_in_nanometer + _min_distance_in_nanometer)
    )
    centers = torch.linspace(start_value, 1, number_of_radial_basis_functions)

    mf_centers = mf_rbf.get_buffer("radial_basis_centers").numpy()
    assert np.allclose(
        np.flip(pn_centers), mf_centers
    )  # NOTE: The PhysNet implementation uses the reverse order of the centers

    # compare the full output
    #################
    # PhysNet implementation
    D = tf.random.uniform(shape=(2, 1), minval=0, maxval=5)
    D = tf.expand_dims(D, -1)  # necessary for proper broadcasting behaviour
    pn_rbf_output = tf.exp(-pn_widths * (tf.exp(-D) - pn_centers) ** 2)

    mf_rbf_output = mf_rbf(torch.tensor(D.numpy() / 10).squeeze())
    assert np.allclose(
        np.flip(pn_rbf_output.numpy().squeeze(), axis=1), mf_rbf_output.numpy()
    )

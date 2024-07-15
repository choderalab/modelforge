import pytest

from openff.units import unit
import torch

from modelforge.potential.aimnet2 import AIMNet2, AIMNet2RadialSymmetryFunction
from modelforge.tests.test_models import load_configs


def test_initialize_model():
    """Test initialization of the Schnet model."""

    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AIMNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet2 model should be initialized."


# TODO: Does this need to be tested since we're using the default GaussianRadialBasisFunction?
@pytest.mark.xfail(raises=NotImplementedError)
def test_rbf():
    raise NotImplementedError


def test_radial_symmetry_function_regression():

    # define radial symmetry function bounds and subdivisions
    num_bins = 10
    lower_bound = unit.Quantity(0.5, unit.angstrom)
    upper_bound = unit.Quantity(5.0, unit.angstrom)

    radial_symmetry_function_module = AIMNet2RadialSymmetryFunction(
        num_bins,
        min_distance=lower_bound,
        max_distance=upper_bound,
    )

    # example interatomic distances in Angstroms, peaks should appear every other index and fall
    # off when outside the upper_bound in outputs given lower_bound, upper_bound, and number of
    # bins (every 0.5 in bin center value)
    d_ij = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])

    # regression outputs
    regression_outputs = torch.tensor(
        [
            [
                6.0653e-01,
                1.0000e00,
                6.0653e-01,
                1.3534e-01,
                1.1109e-02,
                3.3546e-04,
                3.7266e-06,
                1.5230e-08,
                2.2897e-11,
                1.2664e-14,
            ],
            [
                1.1109e-02,
                1.3534e-01,
                6.0653e-01,
                1.0000e00,
                6.0653e-01,
                1.3534e-01,
                1.1109e-02,
                3.3546e-04,
                3.7266e-06,
                1.5230e-08,
            ],
            [
                3.7266e-06,
                3.3546e-04,
                1.1109e-02,
                1.3534e-01,
                6.0653e-01,
                1.0000e00,
                6.0653e-01,
                1.3534e-01,
                1.1109e-02,
                3.3546e-04,
            ],
            [
                2.2897e-11,
                1.5230e-08,
                3.7266e-06,
                3.3546e-04,
                1.1109e-02,
                1.3534e-01,
                6.0653e-01,
                1.0000e00,
                6.0653e-01,
                1.3534e-01,
            ],
            [
                2.5767e-18,
                1.2664e-14,
                2.2897e-11,
                1.5230e-08,
                3.7266e-06,
                3.3546e-04,
                1.1109e-02,
                1.3534e-01,
                6.0653e-01,
                1.0000e00,
            ],
            [
                5.3110e-27,
                1.9287e-22,
                2.5767e-18,
                1.2664e-14,
                2.2897e-11,
                1.5230e-08,
                3.7266e-06,
                3.3546e-04,
                1.1109e-02,
                1.3534e-01,
            ],
            [
                2.0050e-37,
                5.3801e-32,
                5.3110e-27,
                1.9287e-22,
                2.5767e-18,
                1.2664e-14,
                2.2897e-11,
                1.5230e-08,
                3.7266e-06,
                3.3546e-04,
            ],
        ]
    )

    # module call expects units in nanometers, divide by 10 to correct scale
    modelforge_aimnet2_outputs = radial_symmetry_function_module(d_ij / 10.0)

    assert torch.allclose(modelforge_aimnet2_outputs, regression_outputs, atol=1e-5)


@pytest.mark.xfail(raises=AttributeError)
def test_forward(single_batch_with_batchsize_64):
    """Test initialization of the Schnet model."""
    # read default parameters
    config = load_configs(f"schnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AIMNet2(**potential_parameter)
    assert aimnet is not None, "Aimnet model should be initialized."

    y_hat = aimnet(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

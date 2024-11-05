import pytest
import torch
from openff.units import unit

from modelforge.tests.helper_functions import setup_potential_for_test


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_aimnet2_temp")
    return fn


def test_initialize_model(prep_temp_dir):
    """Test initialization of the Schnet model."""

    # read default parameters
    model = setup_potential_for_test(
        "aimnet2", "training", local_cache_dir=str(prep_temp_dir)
    )

    assert model is not None, "Aimnet2 model should be initialized."


def test_radial_symmetry_function_regression():
    from modelforge.potential import SchnetRadialBasisFunction

    # define radial symmetry function bounds and subdivisions
    num_bins = 10
    lower_bound = unit.Quantity(0.5, unit.angstrom)
    upper_bound = unit.Quantity(5.0, unit.angstrom)

    radial_symmetry_function_module = SchnetRadialBasisFunction(
        num_bins,
        min_distance=lower_bound.to(unit.nanometer).m,
        max_distance=upper_bound.to(unit.nanometer).m,
    )

    # example interatomic distances in Angstroms, peaks should appear every
    # other index and fall off when outside the upper_bound in outputs given
    # lower_bound, upper_bound, and number of bins (every 0.5 in bin center
    # value)
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

    assert torch.allclose(modelforge_aimnet2_outputs, regression_outputs, atol=1e-4)


def test_forward(single_batch_with_batchsize, prep_temp_dir):
    """Test initialization of the AIMNet2 model."""
    # read default parameters
    aimnet = setup_potential_for_test("aimnet2", "training", potential_seed=42)

    assert aimnet is not None, "Aimnet model should be initialized."
    batch = single_batch_with_batchsize(64, "QM9", str(prep_temp_dir))

    y_hat = aimnet(batch.nnp_input)

    assert y_hat is not None, "Aimnet model should be able to make predictions."

    ref_per_system_energy = torch.tensor(
        [
            [-1.6222e00],
            [-1.7771e-01],
            [1.5974e-01],
            [-1.2089e-02],
            [-1.8864e-01],
            [-2.7185e-01],
            [-4.3214e00],
            [-1.3357e00],
            [-1.1657e00],
            [-1.4146e00],
            [-1.8898e00],
            [-1.1582e00],
            [-9.1212e00],
            [-4.8285e00],
            [-5.0907e00],
            [-5.4467e00],
            [-1.8100e00],
            [-4.9845e00],
            [-3.7676e00],
            [-2.5988e00],
            [-1.5824e01],
            [-1.0948e01],
            [-2.8324e-01],
            [-4.5179e-01],
            [-6.8437e-01],
            [-3.1547e-01],
            [-5.7387e-01],
            [-4.6788e-01],
            [-1.9818e00],
            [-3.8900e00],
            [-4.2745e00],
            [-2.8107e00],
            [-1.2960e00],
            [-1.5892e00],
            [-5.7663e00],
            [-4.2937e00],
            [-3.0977e00],
            [-2.2906e00],
            [-1.4034e01],
            [-9.6701e00],
            [-7.9657e00],
            [-6.4762e00],
            [-9.7999e00],
            [-5.6619e00],
            [-9.1679e00],
            [-6.8304e00],
            [-1.0582e01],
            [-6.0419e00],
            [-7.2018e00],
            [-5.0521e00],
            [-4.0748e00],
            [-3.5285e00],
            [-2.5017e00],
            [-2.5237e01],
            [-1.9461e01],
            [-1.7413e00],
            [-2.1273e00],
            [-2.5887e00],
            [-1.1963e00],
            [-2.4938e00],
            [-3.1271e00],
            [-1.7812e00],
            [-8.0866e00],
            [-8.7542e00],
        ],
    )

    assert torch.allclose(y_hat['per_system_energy'], ref_per_system_energy, atol=1e-3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

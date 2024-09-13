import pytest

from openff.units import unit
import torch

from modelforge.potential.utils import SchnetRadialBasisFunction
from modelforge.tests.helper_functions import setup_potential_for_test


def test_charge_equilibration():
    from modelforge.potential.aimnet2 import _perform_charge_normalization

    # test charge equilibration
    # ------------------------- #
    # test case 1
    partial_point_charges = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    atomic_subsystem_indices = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64)
    total_charge = torch.tensor([0.0, 1.0])
    charges = _perform_charge_normalization(
        partial_point_charges,
        total_charge,
        atomic_subsystem_indices,
    )

    assert torch.allclose(
        torch.zeros_like(total_charge).scatter_add_(
            0, atomic_subsystem_indices, charges
        ),
        total_charge,
    )

    # ------------------------- #
    # test case 2
    partial_point_charges = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    total_charge = torch.tensor([-1.0, 2.0])
    charges = _perform_charge_normalization(
        partial_point_charges,
        total_charge,
        atomic_subsystem_indices,
    )
    assert torch.allclose(
        torch.zeros_like(total_charge).scatter_add_(
            0, atomic_subsystem_indices, charges
        ),
        total_charge,
    )

    # ------------------------- #
    # test case 3
    partial_point_charges = torch.rand_like(
        atomic_subsystem_indices, dtype=torch.float32
    )
    total_charge = torch.tensor([-1.0, 2.0])
    charges = _perform_charge_normalization(
        partial_point_charges,
        total_charge,
        atomic_subsystem_indices,
    )
    assert torch.allclose(
        torch.zeros_like(total_charge).scatter_add_(
            0, atomic_subsystem_indices, charges
        ),
        total_charge,
    )


def test_initialize_model():
    """Test initialization of the Schnet model."""

    # read default parameters
    model = setup_potential_for_test("aimnet2", "training")

    assert model is not None, "Aimnet2 model should be initialized."


def test_radial_symmetry_function_regression():

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


def test_forward(single_batch_with_batchsize):
    """Test initialization of the AIMNet2 model."""
    # read default parameters
    aimnet = setup_potential_for_test("aimnet2", "training")

    assert aimnet is not None, "Aimnet model should be initialized."
    batch = single_batch_with_batchsize(64, "QM9")

    y_hat = aimnet(batch.nnp_input_tuple)


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

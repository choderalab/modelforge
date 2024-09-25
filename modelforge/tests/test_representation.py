import torch
import pytest


def test_radial_symmetry_function_implementation():
    """
    Test the Radial Symmetry function implementation.
    """
    import torch
    from openff.units import unit
    import numpy as np
    from modelforge.potential.representation import (
        CosineAttenuationFunction,
        GaussianRadialBasisFunctionWithScaling,
    )

    cutoff_module = CosineAttenuationFunction(
        cutoff=unit.Quantity(5.0, unit.angstrom).to(unit.nanometer).m
    )

    class RadialSymmetryFunctionTest(GaussianRadialBasisFunctionWithScaling):
        @staticmethod
        def calculate_radial_basis_centers(
            number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            dtype,
        ):
            centers = torch.linspace(
                _min_distance_in_nanometer,
                _max_distance_in_nanometer,
                number_of_radial_basis_functions,
                dtype=dtype,
            )
            return centers

        @staticmethod
        def calculate_radial_scale_factor(
            number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            dtype,
        ):
            scale_factors = torch.full(
                (number_of_radial_basis_functions,),
                (_min_distance_in_nanometer - _max_distance_in_nanometer)
                / number_of_radial_basis_functions,
            )
            scale_factors = (scale_factors * -15_000) ** -0.5
            return scale_factors

    RSF = RadialSymmetryFunctionTest(
        number_of_radial_basis_functions=18,
        max_distance=unit.Quantity(5.0, unit.angstrom).to(unit.nanometer).m,
    )
    # test a single distance
    d_ij = torch.tensor([[0.2]])
    radial_expension = RSF(d_ij)

    expected_output = np.array(
        [
            5.7777413e-08,
            5.4214674e-06,
            2.4740110e-04,
            5.4905377e-03,
            5.9259072e-02,
            3.1104434e-01,
            7.9399312e-01,
            9.8568588e-01,
            5.9509689e-01,
            1.7472850e-01,
            2.4949821e-02,
            1.7326004e-03,
            5.8513560e-05,
            9.6104134e-07,
            7.6763511e-09,
            2.9819147e-11,
            5.6333109e-14,
            5.1755549e-17,
        ],
        dtype=np.float32,
    )

    assert np.allclose(radial_expension.numpy().flatten(), expected_output, rtol=1e-3)

    # test multiple distances with cutoff
    d_ij = torch.tensor([[r] for r in np.linspace(0, 0.5, 10)])
    radial_expension = RSF(d_ij) * cutoff_module(d_ij)

    expected_output = np.array(
        [
            [
                1.00000000e00,
                6.97370611e-01,
                2.36512753e-01,
                3.90097089e-02,
                3.12909145e-03,
                1.22064879e-04,
                2.31574554e-06,
                2.13657562e-08,
                9.58678574e-11,
                2.09196141e-13,
                2.22005077e-16,
                1.14577532e-19,
                2.87583090e-23,
                3.51038337e-27,
                2.08388175e-31,
                6.01615362e-36,
                8.44679753e-41,
                5.76756600e-46,
            ],
            [
                2.68038176e-01,
                7.29490887e-01,
                9.65540222e-01,
                6.21510012e-01,
                1.94559846e-01,
                2.96200218e-02,
                2.19303227e-03,
                7.89645189e-05,
                1.38275834e-06,
                1.17757010e-08,
                4.87703136e-11,
                9.82316969e-14,
                9.62221521e-17,
                4.58380155e-20,
                1.06194951e-23,
                1.19649050e-27,
                6.55604552e-32,
                1.74703654e-36,
            ],
            [
                5.15165267e-03,
                5.47178933e-02,
                2.82643788e-01,
                7.10030194e-01,
                8.67443988e-01,
                5.15386799e-01,
                1.48919812e-01,
                2.09266151e-02,
                1.43012111e-03,
                4.75305832e-05,
                7.68248035e-07,
                6.03888967e-09,
                2.30855409e-11,
                4.29190731e-14,
                3.88050222e-17,
                1.70629005e-20,
                3.64875837e-24,
                3.79458837e-28,
            ],
            [
                7.05512776e-06,
                2.92447055e-04,
                5.89544925e-03,
                5.77981439e-02,
                2.75573882e-01,
                6.38983424e-01,
                7.20556963e-01,
                3.95161266e-01,
                1.05392022e-01,
                1.36699807e-02,
                8.62294776e-04,
                2.64527563e-05,
                3.94651201e-07,
                2.86340809e-09,
                1.01036987e-11,
                1.73382336e-14,
                1.44696036e-17,
                5.87267193e-21,
            ],
            [
                6.79841545e-10,
                1.09978970e-07,
                8.65244557e-06,
                3.31051436e-04,
                6.15997825e-03,
                5.57430086e-02,
                2.45317579e-01,
                5.25042257e-01,
                5.46496226e-01,
                2.76635027e-01,
                6.81011682e-02,
                8.15322217e-03,
                4.74713206e-04,
                1.34419004e-05,
                1.85104660e-07,
                1.23965647e-09,
                4.03750130e-12,
                6.39515861e-15,
            ],
            [
                4.50275565e-15,
                2.84275808e-12,
                8.72828077e-10,
                1.30330158e-07,
                9.46429271e-06,
                3.34240505e-04,
                5.74059467e-03,
                4.79492711e-02,
                1.94775558e-01,
                3.84781601e-01,
                3.69675978e-01,
                1.72725113e-01,
                3.92479574e-02,
                4.33716512e-03,
                2.33089213e-04,
                6.09208166e-06,
                7.74348707e-08,
                4.78668403e-10,
            ],
            [
                1.95755731e-21,
                4.82320349e-18,
                5.77941614e-15,
                3.36790471e-12,
                9.54470642e-10,
                1.31550705e-07,
                8.81760261e-06,
                2.87432047e-04,
                4.55666577e-03,
                3.51307040e-02,
                1.31720486e-01,
                2.40185684e-01,
                2.12994423e-01,
                9.18579329e-02,
                1.92660386e-02,
                1.96514909e-03,
                9.74823310e-05,
                2.35170925e-06,
            ],
            [
                5.02685557e-29,
                4.83367095e-25,
                2.26039866e-21,
                5.14067897e-18,
                5.68568509e-15,
                3.05825078e-12,
                7.99999982e-10,
                1.01773376e-07,
                6.29659424e-06,
                1.89454631e-04,
                2.77224261e-03,
                1.97280685e-02,
                6.82755520e-02,
                1.14914067e-01,
                9.40607618e-02,
                3.74430407e-02,
                7.24871542e-03,
                6.82461743e-04,
            ],
            [
                5.43174696e-38,
                2.03835481e-33,
                3.72003749e-29,
                3.30173621e-25,
                1.42516199e-21,
                2.99167362e-18,
                3.05415190e-15,
                1.51633227e-12,
                3.66121676e-10,
                4.29917177e-08,
                2.45510656e-06,
                6.81841165e-05,
                9.20923191e-04,
                6.04910223e-03,
                1.93234986e-02,
                3.00198084e-02,
                2.26807491e-02,
                8.33362963e-03,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
        ]
    )

    assert np.allclose(radial_expension.numpy(), expected_output, rtol=1e-3)


def test_schnet_rbf():
    """
    Test the SchnetRadialBasisFunction class.
    """
    from modelforge.potential.representation import SchnetRadialBasisFunction

    # Test parameters
    distances = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32) / 10
    number_of_radial_basis_functions = 3
    max_distance = 2.0 / 10
    min_distance = 0.0
    dtype = torch.float32

    # Instantiate the RBF
    rbf = SchnetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=max_distance,
        min_distance=min_distance,
        dtype=dtype,
        trainable_centers_and_scale_factors=False,
    )

    # Compute expected outputs
    centers = rbf.radial_basis_centers  # Shape: [number_of_radial_basis_functions]
    scale_factors = rbf.radial_scale_factor  # Shape: [number_of_radial_basis_functions]

    # Expand dimensions for broadcasting
    distances_expanded = distances  # Shape: [number_of_pairs, 1]
    centers_expanded = centers.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]
    scale_factors_expanded = scale_factors.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]

    # Calculate nondimensionalized distances and expected outputs
    diff = distances_expanded - centers_expanded
    nondim_distances = diff / scale_factors_expanded
    expected_output = torch.exp(-(nondim_distances**2))

    # Get actual outputs
    actual_output = rbf(distances)

    # Assertions
    assert actual_output.shape == expected_output.shape, "Output shape mismatch"
    assert torch.allclose(
        actual_output, expected_output, atol=1e-6
    ), "Outputs do not match expected values for SchnetRadialBasisFunction"


def test_ani_rbf():
    """
    Test the AniRadialBasisFunction class.
    """
    from modelforge.potential.representation import AniRadialBasisFunction

    # Test parameters
    distances = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)
    number_of_radial_basis_functions = 3
    max_distance = 2.0
    min_distance = 0.0
    dtype = torch.float32

    # Instantiate the RBF
    rbf = AniRadialBasisFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=max_distance,
        min_distance=min_distance,
        dtype=dtype,
        trainable_centers_and_scale_factors=False,
    )

    # Compute expected outputs
    centers = rbf.radial_basis_centers  # Shape: [number_of_radial_basis_functions]
    scale_factors = rbf.radial_scale_factor  # Shape: [number_of_radial_basis_functions]

    # Expand dimensions for broadcasting
    distances_expanded = distances  # Shape: [number_of_pairs, 1]
    centers_expanded = centers.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]
    scale_factors_expanded = scale_factors.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]

    # Calculate nondimensionalized distances and expected outputs
    diff = distances_expanded - centers_expanded
    nondim_distances = diff / scale_factors_expanded
    expected_output = 0.25 * torch.exp(-(nondim_distances**2))  # Include prefactor

    # Get actual outputs
    actual_output = rbf(distances)

    # Assertions
    assert actual_output.shape == expected_output.shape, "Output shape mismatch"
    assert torch.allclose(
        actual_output, expected_output, atol=1e-6
    ), "Outputs do not match expected values for AniRadialBasisFunction"


def test_physnet_rbf():
    """
    Test the PhysNetRadialBasisFunction class.
    """
    from modelforge.potential.representation import PhysNetRadialBasisFunction

    # Test parameters
    distances = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32) / 10
    number_of_radial_basis_functions = 3
    max_distance = 2.0 / 10
    min_distance = 0.0
    alpha = 0.1
    dtype = torch.float32

    # Instantiate the RBF
    rbf = PhysNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=max_distance,
        min_distance=min_distance,
        alpha=alpha,
        dtype=dtype,
        trainable_centers_and_scale_factors=False,
    )

    # Compute expected outputs
    centers = rbf.radial_basis_centers  # Unitless centers
    scale_factors = rbf.radial_scale_factor  # Unitless scale factors

    # Expand dimensions for broadcasting
    distances_expanded = distances  # Shape: [number_of_pairs, 1]
    centers_expanded = centers.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]
    scale_factors_expanded = scale_factors.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]

    # Nondimensionalization as per PhysNet
    nondim_distances = (
        torch.exp((-distances + min_distance) / alpha) - centers_expanded
    ) / scale_factors_expanded
    expected_output = torch.exp(-(nondim_distances**2))

    # Get actual outputs
    actual_output = rbf(distances)

    # Assertions
    assert actual_output.shape == expected_output.shape, "Output shape mismatch"
    assert torch.allclose(
        actual_output, expected_output, atol=1e-6
    ), "Outputs do not match expected values for PhysNetRadialBasisFunction"


def test_tensornet_rbf():
    """
    Test the TensorNetRadialBasisFunction class.
    """
    from modelforge.potential.representation import TensorNetRadialBasisFunction

    # Test parameters
    distances = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)
    number_of_radial_basis_functions = 3
    max_distance = 2.0
    min_distance = 0.0
    dtype = torch.float32

    # Instantiate the RBF
    rbf = TensorNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=max_distance,
        min_distance=min_distance,
        dtype=dtype,
        trainable_centers_and_scale_factors=False,
    )

    # Compute expected outputs
    centers = rbf.radial_basis_centers  # Unitless centers
    scale_factors = rbf.radial_scale_factor  # Unitless scale factors
    alpha = 0.1  # As per TensorNet implementation

    # Expand dimensions for broadcasting
    distances_expanded = distances  # Shape: [number_of_pairs, 1]
    centers_expanded = centers.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]
    scale_factors_expanded = scale_factors.unsqueeze(
        0
    )  # Shape: [1, number_of_radial_basis_functions]

    # Nondimensionalization as per TensorNet
    nondim_distances = (
        torch.exp((-distances + min_distance) / alpha) - centers_expanded
    ) / scale_factors_expanded
    expected_output = torch.exp(-(nondim_distances**2))

    # Get actual outputs
    actual_output = rbf(distances)

    # Assertions
    assert actual_output.shape == expected_output.shape, "Output shape mismatch"
    assert torch.allclose(
        actual_output, expected_output, atol=1e-6
    ), "Outputs do not match expected values for TensorNetRadialBasisFunction"

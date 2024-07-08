
import pytest
import torch


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


def test_compare_radial_symmetry_features():
    # compare schnetpack RadialSymmetryFunction with modelforge RadialSymmetryFunction
    from modelforge.potential.utils import SchnetRadialSymmetryFunction
    from openff.units import unit

    # Initialize the RBFs
    number_of_gaussians = 10
    cutoff = unit.Quantity(5.2, unit.angstrom)
    start = unit.Quantity(0.8, unit.angstrom)

    radial_symmetry_function_module = SchnetRadialSymmetryFunction(
        number_of_radial_basis_functions=number_of_gaussians,
        max_distance=cutoff,
        min_distance=start,
    )

    # define pariwise distances
    d_ij = torch.tensor([[1.0077], [4.2496], [2.8202], [3.4342], [9.2465]])

    # this has been calculated with schnetpack2.0
    schnetpack_rbf_output = torch.tensor(
        [
            [
                [
                    9.1371e-01,
                    8.4755e-01,
                    2.8922e-01,
                    3.6308e-02,
                    1.6768e-03,
                    2.8488e-05,
                    1.7805e-07,
                    4.0939e-10,
                    3.4629e-13,
                    1.0776e-16,
                ]
            ],
            [
                [
                    1.5448e-11,
                    1.0867e-08,
                    2.8121e-06,
                    2.6772e-04,
                    9.3763e-03,
                    1.2081e-01,
                    5.7260e-01,
                    9.9843e-01,
                    6.4046e-01,
                    1.5114e-01,
                ]
            ],
            [
                [
                    1.9595e-04,
                    7.4063e-03,
                    1.0298e-01,
                    5.2678e-01,
                    9.9130e-01,
                    6.8625e-01,
                    1.7477e-01,
                    1.6374e-02,
                    5.6435e-04,
                    7.1557e-06,
                ]
            ],
            [
                [
                    4.9634e-07,
                    6.5867e-05,
                    3.2156e-03,
                    5.7752e-02,
                    3.8157e-01,
                    9.2744e-01,
                    8.2929e-01,
                    2.7279e-01,
                    3.3011e-02,
                    1.4696e-03,
                ]
            ],
            [
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    5.6052e-45,
                    5.2717e-39,
                    1.8660e-33,
                    2.4297e-28,
                    1.1639e-23,
                    2.0511e-19,
                    1.3297e-15,
                ]
            ],
        ]
    )

    assert torch.allclose(
        schnetpack_rbf_output,
        radial_symmetry_function_module(d_ij / 10).unsqueeze(1),
        atol=1e-5,
    )  # NOTE: there is a shape mismatch between the two outputs

import pytest

from openff.units import unit
import torch

from modelforge.potential.aimnet2 import AIMNet2
from modelforge.potential.utils import SchnetRadialBasisFunction
from modelforge.tests.test_models import load_configs_into_pydantic_models


def test_initialize_model():
    """Test initialization of the Schnet model."""

    # read default parameters
    config = load_configs_into_pydantic_models(f"aimnet2", "qm9")

    aimnet = AIMNet2(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )

    assert aimnet is not None, "Aimnet2 model should be initialized."


def test_radial_symmetry_function_regression():

    # define radial symmetry function bounds and subdivisions
    num_bins = 10
    lower_bound = unit.Quantity(0.5, unit.angstrom)
    upper_bound = unit.Quantity(5.0, unit.angstrom)

    radial_symmetry_function_module = SchnetRadialBasisFunction(
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


def test_representation_module():

    from modelforge.potential.aimnet2 import AIMNet2Representation

    num_bins = 10
    radial_cutoff = unit.Quantity(5.0, unit.angstrom)
    # distances for 7 atoms, j, in the system with atom i
    d_ij = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])

    representation_module = AIMNet2Representation(
        radial_cutoff=radial_cutoff, number_of_radial_basis_functions=num_bins
    )
    representation = representation_module(d_ij / 10)

    # check radial basis function component shape
    f_ij = representation["f_ij"]
    assert f_ij.shape[0] == 7 and f_ij.shape[1] == 10

    # check cutoff function component shape
    f_cutoff = representation["f_cutoff"]
    assert f_cutoff.shape[0] == 7 and f_cutoff.shape[1] == 1

    # check full radial symmetry function shape
    g_ijs = f_ij * f_cutoff
    assert g_ijs.shape[0] == 7 and g_ijs.shape[1] == 10

    # TODO: need comparison data
    pass


@pytest.mark.xfail(raises=NotImplementedError)
def test_interaction_module():

    from modelforge.potential.aimnet2 import AIMNet2InteractionModule

    number_of_atomic_features = 10
    number_of_radial_basis_functions = 10

    first_pass_interaction_module = AIMNet2InteractionModule(
        number_of_atomic_features, number_of_radial_basis_functions, first_pass=True
    )

    second_pass_interaction_module = AIMNet2InteractionModule(
        number_of_atomic_features, number_of_radial_basis_functions, first_pass=False
    )

    raise NotImplementedError


@pytest.mark.xfail(raises=AttributeError)
def test_forward(single_batch_with_batchsize_64):
    """Test initialization of the AIMNet2 model."""
    # read default parameters
    config = load_configs_into_pydantic_models(f"aimnet2", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    aimnet = AIMNet2(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )

    assert aimnet is not None, "Aimnet model should be initialized."

    y_hat = aimnet(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

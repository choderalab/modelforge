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


def test_forward(single_batch_with_batchsize, prep_temp_dir, dataset_temp_dir):
    """Test initialization of the AIMNet2 model."""
    # read default parameters
    aimnet = setup_potential_for_test("aimnet2", "training", potential_seed=42)

    assert aimnet is not None, "Aimnet model should be initialized."
    local_cache_dir = str(prep_temp_dir) + "/aimnet2_forward"
    batch = single_batch_with_batchsize(
        batch_size=64,
        dataset_name="QM9",
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_temp_dir,
    )

    y_hat = aimnet(batch.nnp_input)

    assert y_hat is not None, "Aimnet model should be able to make predictions."

    ref_per_system_energy = torch.tensor(
        [
            [0.2630],
            [-0.5150],
            [-0.2999],
            [-0.0297],
            [-0.4382],
            [-0.1805],
            [0.5974],
            [0.1770],
            [0.0842],
            [-0.2955],
            [0.1295],
            [-0.4067],
            [0.4135],
            [0.3202],
            [0.2481],
            [0.6696],
            [0.0380],
            [0.0834],
            [-0.2613],
            [-0.8373],
            [0.2033],
            [0.1554],
            [0.0624],
            [-0.3643],
            [-0.7861],
            [-0.0398],
            [-0.4675],
            [-0.1000],
            [0.3265],
            [0.2546],
            [-0.1597],
            [-0.9611],
            [0.0653],
            [-0.4411],
            [0.2587],
            [-0.1082],
            [0.0461],
            [0.0407],
            [0.6725],
            [0.3874],
            [0.3393],
            [0.1747],
            [0.4048],
            [0.1001],
            [0.1496],
            [0.2432],
            [0.3578],
            [0.2792],
            [-0.3365],
            [-0.3329],
            [-0.8465],
            [0.0463],
            [-0.4385],
            [0.1224],
            [-0.0442],
            [0.1029],
            [-0.4559],
            [-1.1701],
            [-0.2714],
            [0.0318],
            [-0.8579],
            [-0.3836],
            [0.2487],
            [-0.2728],
        ],
    )

    print(y_hat["per_system_energy"])
    assert torch.allclose(y_hat["per_system_energy"], ref_per_system_energy, atol=1e-3)


def test_mlp_initialization():
    # this will test the MLP initialization is as expected

    from modelforge.potential.aimnet2 import AIMNet2InteractionModule
    from modelforge.potential.utils import ACTIVATION_FUNCTIONS

    num_per_atom_features = 128
    number_of_vector_features = 8
    hidden_layers = [512, 256]
    interaction = AIMNet2InteractionModule(
        number_of_per_atom_features=num_per_atom_features,
        number_of_radial_basis_functions=64,
        number_of_vector_features=number_of_vector_features,
        hidden_layers=hidden_layers,
        activation_function=ACTIVATION_FUNCTIONS["GeLU"](),
        is_first_module=True,
    )

    assert len(interaction.mlp) == 3, "MLP should have 3 layers."
    assert (
        interaction.mlp[0].in_features
        == num_per_atom_features + number_of_vector_features
    )
    assert interaction.mlp[0].out_features == hidden_layers[0]
    assert interaction.mlp[1].in_features == hidden_layers[0]
    assert interaction.mlp[1].out_features == hidden_layers[1]
    assert interaction.mlp[2].in_features == hidden_layers[1]
    assert interaction.mlp[2].out_features == num_per_atom_features + 2

    num_per_atom_features = 128
    number_of_vector_features = 8
    hidden_layers = [512, 380, 256]
    interaction = AIMNet2InteractionModule(
        number_of_per_atom_features=num_per_atom_features,
        number_of_radial_basis_functions=64,
        number_of_vector_features=number_of_vector_features,
        hidden_layers=hidden_layers,
        activation_function=ACTIVATION_FUNCTIONS["GeLU"](),
        is_first_module=False,
    )

    assert len(interaction.mlp) == 4, "MLP should have 4 layers."
    assert (
        interaction.mlp[0].in_features
        == num_per_atom_features
        + number_of_vector_features
        + num_per_atom_features
        + number_of_vector_features
    )

    assert interaction.mlp[0].out_features == hidden_layers[0]
    assert interaction.mlp[1].in_features == hidden_layers[0]
    assert interaction.mlp[1].out_features == hidden_layers[1]
    assert interaction.mlp[2].in_features == hidden_layers[1]
    assert interaction.mlp[2].out_features == hidden_layers[2]
    assert interaction.mlp[3].in_features == hidden_layers[2]
    assert interaction.mlp[3].out_features == num_per_atom_features + 2


@pytest.mark.xfail(raises=NotImplementedError)
def test_against_original_implementation():
    raise NotImplementedError

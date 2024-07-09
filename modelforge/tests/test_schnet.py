import pytest
import torch
from modelforge.tests.precalculated_values import (
    load_precalculated_schnet_results,
    setup_single_methane_input,
)


def initialize_schnet(
    cutoff: float,
    number_of_atom_features: int,
    number_of_radial_basis_functions: int,
    nr_of_interactions: int,
):
    # ------------------------------------ #
    # set up the modelforge Painn representation model
    # which means that we only want to call the
    # _transform_input() method
    from modelforge.potential.schnet import SchNet

    return SchNet(
        max_Z=101,
        number_of_atom_features=number_of_atom_features,
        number_of_interaction_modules=nr_of_interactions,
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        cutoff=cutoff,
        number_of_filters=number_of_atom_features,
        shared_interactions=False,
        processing_operation=[],
        readout_operation=[
            {
                "step": "from_atom_to_molecule",
                "mode": "sum",
                "in": "E_i",
                "index_key": "atomic_subsystem_indices",
                "out": "E",
            }
        ],
    )


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
    from modelforge.potential.utils import SchnetRadialBasisFunction
    from openff.units import unit

    # Initialize the RBFs
    number_of_gaussians = 10
    cutoff = unit.Quantity(5.2, unit.angstrom)
    start = unit.Quantity(0.8, unit.angstrom)

    radial_symmetry_function_module = SchnetRadialBasisFunction(
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


def test_schnet_forward_pass():
    # ---------------------------------------- #
    # test the implementation of the representation part of the PaiNN model
    # ---------------------------------------- #
    from openff.units import unit

    cutoff = unit.Quantity(5.0, unit.angstrom)
    number_of_atom_features = 12
    n_rbf = 5
    nr_of_interactions = 3
    torch.manual_seed(1234)
    modelforge_schnet = initialize_schnet(
        cutoff, number_of_atom_features, n_rbf, nr_of_interactions
    ).double()

    # ------------------------------------ #
    # reference values
    # generated with schnetpack2.0

    # ------------------------------------ #
    # set up the input for the spk Schnet model
    input = setup_single_methane_input()
    spk_input = input["spk_methane_input"]
    mf_nnp_input = input["modelforge_methane_input"]

    modelforge_schnet.input_preparation._input_checks(mf_nnp_input)

    pairlist_output = modelforge_schnet.input_preparation.prepare_inputs(mf_nnp_input)
    schnet_nn_input_mf = (
        modelforge_schnet.core_module._model_specific_input_preparation(
            mf_nnp_input, pairlist_output
        )
    )

    # ---------------------------------------- #
    # test neighborlist and distance
    # ---------------------------------------- #
    assert torch.allclose(spk_input["_Rij"] / 10, schnet_nn_input_mf.r_ij, atol=1e-4)
    assert torch.allclose(spk_input["_idx_i"], schnet_nn_input_mf.pair_indices[0])
    assert torch.allclose(spk_input["_idx_j"], schnet_nn_input_mf.pair_indices[1])

    # ---------------------------------------- #
    # test radial symmetry function
    # ---------------------------------------- #
    r_ij = spk_input["_Rij"]
    d_ij = torch.norm(r_ij, dim=1, keepdim=True)

    schnetpack_phi_ij = torch.tensor(
        [
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9130, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9130, 0.8484, 0.2900, 0.0365]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.3615, 0.9130, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.6828, 0.9920, 0.5302, 0.1043, 0.0075]],
            [[0.3615, 0.9130, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
            [[0.3615, 0.9131, 0.8484, 0.2900, 0.0365]],
        ],
        dtype=torch.float64,
    )
    modelforge_phi_ij = modelforge_schnet.core_module.schnet_representation_module.radial_symmetry_function_module(
        d_ij / 10
    )  # NOTE: converting to nm

    assert torch.allclose(schnetpack_phi_ij, modelforge_phi_ij.unsqueeze(1), atol=1e-3)
    # ---------------------------------------- #
    # test cutoff
    # ---------------------------------------- #
    fcut_spk = torch.tensor(
        [
            [0.8869],
            [0.8869],
            [0.8869],
            [0.8869],
            [0.8869],
            [0.7177],
            [0.7177],
            [0.7177],
            [0.8869],
            [0.7177],
            [0.7177],
            [0.7177],
            [0.8869],
            [0.7177],
            [0.7177],
            [0.7177],
            [0.8869],
            [0.7177],
            [0.7177],
            [0.7177],
        ],
        dtype=torch.float64,
    )
    fcut_mf = modelforge_schnet.core_module.schnet_representation_module.cutoff_module(
        d_ij / 10
    )  # NOTE: converting to nm
    assert torch.allclose(fcut_spk, fcut_mf, atol=1e-4)

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #

    # Check full pass
    torch.manual_seed(1234)
    for i in range(nr_of_interactions):
        modelforge_schnet.core_module.interaction_modules[
            i
        ].intput_to_feature.reset_parameters()
        for j in range(2):
            modelforge_schnet.core_module.interaction_modules[i].feature_to_output[
                j
            ].reset_parameters()
            modelforge_schnet.core_module.interaction_modules[i].filter_network[
                j
            ].reset_parameters()

    modelforge_results = modelforge_schnet.core_module.forward(
        schnet_nn_input_mf, pairlist_output
    )
    schnetpack_results = load_precalculated_schnet_results()
    assert (
        schnetpack_results["scalar_representation"].shape
        == modelforge_results["scalar_representation"].shape
    )

    scalar_spk = schnetpack_results["scalar_representation"]
    scalar_mf = modelforge_results["scalar_representation"]
    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)

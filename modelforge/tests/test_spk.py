import torch


def setup_single_methane_input():
    import torch

    # ------------------------------------ #
    # set up the input for the spk Painn model
    methan_spk = {
        "_idx": torch.tensor([0]),
        "dipole_moment": torch.tensor([0.0], dtype=torch.float64),
        "energy_U0": torch.tensor([-40.4789], dtype=torch.float64),
        "energy_U": torch.tensor([-40.4761], dtype=torch.float64),
        "_n_atoms": torch.tensor([5]),
        "_atomic_numbers": torch.tensor([6, 1, 1, 1, 1]),
        "_positions": torch.tensor(
            [
                [-1.2698e-02, 1.0858e00, 8.0010e-03],
                [2.1504e-03, -6.0313e-03, 1.9761e-03],
                [1.0117e00, 1.4638e00, 2.7657e-04],
                [-5.4082e-01, 1.4475e00, -8.7664e-01],
                [-5.2381e-01, 1.4379e00, 9.0640e-01],
            ],
            dtype=torch.float64,
        ),
        "_cell": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64
        ),
        "_pbc": torch.tensor([False, False, False]),
        "_offsets": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        "_idx_i": torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        ),
        "_idx_j": torch.tensor(
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
        ),
        "_Rij": torch.tensor(
            [
                [1.4849e-02, -1.0918e00, -6.0249e-03],
                [1.0244e00, 3.7795e-01, -7.7244e-03],
                [-5.2812e-01, 3.6172e-01, -8.8464e-01],
                [-5.1112e-01, 3.5213e-01, 8.9840e-01],
                [-1.4849e-02, 1.0918e00, 6.0249e-03],
                [1.0096e00, 1.4698e00, -1.6995e-03],
                [-5.4297e-01, 1.4536e00, -8.7862e-01],
                [-5.2596e-01, 1.4440e00, 9.0442e-01],
                [-1.0244e00, -3.7795e-01, 7.7244e-03],
                [-1.0096e00, -1.4698e00, 1.6995e-03],
                [-1.5525e00, -1.6225e-02, -8.7692e-01],
                [-1.5355e00, -2.5819e-02, 9.0612e-01],
                [5.2812e-01, -3.6172e-01, 8.8464e-01],
                [5.4297e-01, -1.4536e00, 8.7862e-01],
                [1.5525e00, 1.6225e-02, 8.7692e-01],
                [1.7001e-02, -9.5940e-03, 1.7830e00],
                [5.1112e-01, -3.5213e-01, -8.9840e-01],
                [5.2596e-01, -1.4440e00, -9.0442e-01],
                [1.5355e00, 2.5819e-02, -9.0612e-01],
                [-1.7001e-02, 9.5940e-03, -1.7830e00],
            ],
            dtype=torch.float64,
        ),
    }
    # ------------------------------------ #

    # ------------------------------------ #
    # set up the input for the modelforge Painn model
    atomic_numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.int64)

    positions = (
        torch.tensor(
            [
                [-1.2698e-02, 1.0858e00, 8.0010e-03],
                [2.1504e-03, -6.0313e-03, 1.9761e-03],
                [1.0117e00, 1.4638e00, 2.7657e-04],
                [-5.4082e-01, 1.4475e00, -8.7664e-01],
                [-5.2381e-01, 1.4379e00, 9.0640e-01],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        / 10
    )
    E = torch.tensor([0.0], requires_grad=True)
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    from modelforge.dataset.dataset import NNPInput

    modelforge_methane = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=atomic_subsystem_indices,
        total_charge=torch.tensor([0], dtype=torch.int32),
    )
    # ------------------------------------ #

    return {
        "spk_methane_input": methan_spk,
        "modelforge_methane_input": modelforge_methane,
    }


def setup_spk_painn_representation(
    cutoff, nr_atom_basis, number_of_gaussians, nr_of_interactions
):
    # ------------------------------------ #
    # set up the schnetpack Painn representation model
    from schnetpack.nn import GaussianRBF, CosineCutoff
    from schnetpack.representation import PaiNN as schnetpack_PaiNN
    from openff.units import unit

    radial_basis = GaussianRBF(
        n_rbf=number_of_gaussians, cutoff=cutoff.to(unit.angstrom).m
    )
    return schnetpack_PaiNN(
        n_atom_basis=nr_atom_basis,
        n_interactions=nr_of_interactions,
        radial_basis=radial_basis,
        cutoff_fn=CosineCutoff(cutoff.to(unit.angstrom).m),
    )


def setup_modelforge_painn_representation(
    cutoff, nr_atom_basis, number_of_gaussians, nr_of_interactions
):
    # ------------------------------------ #
    # set up the modelforge Painn representation model
    # which means that we only want to call the
    # _transform_input() method
    from modelforge.potential.painn import PaiNN as mf_PaiNN
    from openff.units import unit

    return mf_PaiNN(
        max_Z=100,
        number_of_atom_features=nr_atom_basis,
        number_of_interaction_modules=nr_of_interactions,
        number_of_radial_basis_functions=number_of_gaussians,
        cutoff=cutoff,
        shared_interactions=False,
        shared_filters=False,
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


def test_painn_representation_implementation():
    # ---------------------------------------- #
    # test the implementation of the representation part of the PaiNN model
    # ---------------------------------------- #
    from openff.units import unit

    cutoff = unit.Quantity(5.0, unit.angstrom)
    nr_atom_basis = 128
    number_of_gaussians = 5
    nr_of_interactions = 3
    torch.manual_seed(1234)
    schnetpack_painn = setup_spk_painn_representation(
        cutoff, nr_atom_basis, number_of_gaussians, nr_of_interactions
    ).double()
    torch.manual_seed(1234)

    modelforge_painn = setup_modelforge_painn_representation(
        cutoff, nr_atom_basis, number_of_gaussians, nr_of_interactions
    ).double()
    # ------------------------------------ #
    # set up the input for the spk Painn model
    input = setup_single_methane_input()
    spk_input = input["spk_methane_input"]
    mf_nnp_input = input["modelforge_methane_input"]

    schnetpack_results = schnetpack_painn(spk_input)
    modelforge_painn.input_preparation._input_checks(mf_nnp_input)
    pairlist_output = modelforge_painn.input_preparation.prepare_inputs(mf_nnp_input)
    pain_nn_input_mf = modelforge_painn.core_module._model_specific_input_preparation(
        mf_nnp_input, pairlist_output
    )

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #

    # reset filter parameters
    torch.manual_seed(1234)
    modelforge_painn.core_module.representation_module.filter_net.reset_parameters()
    torch.manual_seed(1234)
    schnetpack_painn.filter_net.reset_parameters()

    modelforge_results = modelforge_painn.core_module._forward(pain_nn_input_mf)
    schnetpack_results = schnetpack_painn(spk_input)

    assert (
        schnetpack_results["scalar_representation"].shape
        == modelforge_results["q"].shape
    )

    scalar_spk = schnetpack_results["scalar_representation"]
    scalar_mf = modelforge_results["q"]
    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)

    assert torch.allclose(
        schnetpack_results["vector_representation"],
        modelforge_results["mu"],
        atol=1e-4,
    )


def setup_spk_schnet_representation(
    cutoff: float, number_of_atom_features: int, n_rbf: int, nr_of_interactions: int
):
    # ------------------------------------ #
    # set up the schnetpack Painn representation model
    from schnetpack.nn import GaussianRBF, CosineCutoff
    from schnetpack.representation import SchNet as schnetpack_SchNET
    from openff.units import unit

    radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff.to(unit.angstrom).m)
    return schnetpack_SchNET(
        n_atom_basis=number_of_atom_features,
        n_interactions=nr_of_interactions,
        radial_basis=radial_basis,
        cutoff_fn=CosineCutoff(cutoff.to(unit.angstrom).m),
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

    return SchNET(
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


def test_schnet_representation_implementation():
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
    schnetpack_results = {
        "_idx": torch.tensor([0]),
        "dipole_moment": torch.tensor([0.0], dtype=torch.float64),
        "energy_U0": torch.tensor([-40.4789], dtype=torch.float64),
        "energy_U": torch.tensor([-40.4761], dtype=torch.float64),
        "_n_atoms": torch.tensor([5]),
        "_atomic_numbers": torch.tensor([6, 1, 1, 1, 1]),
        "_positions": torch.tensor(
            [
                [-1.2698e-02, 1.0858e00, 8.0010e-03],
                [2.1504e-03, -6.0313e-03, 1.9761e-03],
                [1.0117e00, 1.4638e00, 2.7657e-04],
                [-5.4082e-01, 1.4475e00, -8.7664e-01],
                [-5.2381e-01, 1.4379e00, 9.0640e-01],
            ],
            dtype=torch.float64,
        ),
        "_cell": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64
        ),
        "_pbc": torch.tensor([False, False, False]),
        "_offsets": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        "_idx_i": torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        ),
        "_idx_j": torch.tensor(
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
        ),
        "_Rij": torch.tensor(
            [
                [1.4849e-02, -1.0918e00, -6.0249e-03],
                [1.0244e00, 3.7795e-01, -7.7244e-03],
                [-5.2812e-01, 3.6172e-01, -8.8464e-01],
                [-5.1112e-01, 3.5213e-01, 8.9840e-01],
                [-1.4849e-02, 1.0918e00, 6.0249e-03],
                [1.0096e00, 1.4698e00, -1.6995e-03],
                [-5.4297e-01, 1.4536e00, -8.7862e-01],
                [-5.2596e-01, 1.4440e00, 9.0442e-01],
                [-1.0244e00, -3.7795e-01, 7.7244e-03],
                [-1.0096e00, -1.4698e00, 1.6995e-03],
                [-1.5525e00, -1.6225e-02, -8.7692e-01],
                [-1.5355e00, -2.5819e-02, 9.0612e-01],
                [5.2812e-01, -3.6172e-01, 8.8464e-01],
                [5.4297e-01, -1.4536e00, 8.7862e-01],
                [1.5525e00, 1.6225e-02, 8.7692e-01],
                [1.7001e-02, -9.5940e-03, 1.7830e00],
                [5.1112e-01, -3.5213e-01, -8.9840e-01],
                [5.2596e-01, -1.4440e00, -9.0442e-01],
                [1.5355e00, 2.5819e-02, -9.0612e-01],
                [-1.7001e-02, 9.5940e-03, -1.7830e00],
            ],
            dtype=torch.float64,
        ),
        "scalar_representation": torch.tensor(
            [
                [
                    0.1254,
                    -0.9284,
                    -0.6935,
                    2.2096,
                    -0.0555,
                    -0.1595,
                    -1.1804,
                    0.6562,
                    -0.3001,
                    -0.4318,
                    1.0901,
                    -0.0626,
                ],
                [
                    -0.0200,
                    -1.9309,
                    0.5967,
                    -0.3637,
                    0.2486,
                    0.1331,
                    -0.7700,
                    -1.4115,
                    -0.1196,
                    0.5523,
                    0.0644,
                    -0.4112,
                ],
                [
                    -0.0200,
                    -1.9309,
                    0.5967,
                    -0.3637,
                    0.2486,
                    0.1331,
                    -0.7700,
                    -1.4115,
                    -0.1196,
                    0.5523,
                    0.0645,
                    -0.4112,
                ],
                [
                    -0.0200,
                    -1.9309,
                    0.5967,
                    -0.3637,
                    0.2486,
                    0.1331,
                    -0.7700,
                    -1.4115,
                    -0.1196,
                    0.5523,
                    0.0645,
                    -0.4112,
                ],
                [
                    -0.0200,
                    -1.9309,
                    0.5967,
                    -0.3637,
                    0.2486,
                    0.1331,
                    -0.7700,
                    -1.4115,
                    -0.1196,
                    0.5523,
                    0.0645,
                    -0.4112,
                ],
            ],
            dtype=torch.float64,
        ),
    }

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
        d_ij.unsqueeze(1) / 10
    )  # NOTE: converting to nm

    assert torch.allclose(schnetpack_phi_ij, modelforge_phi_ij, atol=1e-3)
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
    modelforge_results = modelforge_schnet.core_module.forward(
        schnet_nn_input_mf, pairlist_output
    )

    assert (
        schnetpack_results["scalar_representation"].shape
        == modelforge_results["E"].shape
    )

    scalar_spk = schnetpack_results["scalar_representation"]
    scalar_mf = modelforge_results["E"]
    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)

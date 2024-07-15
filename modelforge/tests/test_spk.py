import torch
import pytest


@pytest.mark.xfail
def test_compare_radial_symmetry_features():
    # compare schnetpack RadialSymmetryFunction with modelforge RadialSymmetryFunction
    from modelforge.potential.utils import SchnetRadialSymmetryFunction
    from schnetpack.nn import GaussianRBF as schnetpackGaussianRBF
    from openff.units import unit

    # Initialize the RBFs
    number_of_gaussians = 10
    cutoff = unit.Quantity(5.2, unit.angstrom)
    start = unit.Quantity(0.8, unit.angstrom)
    schnetpack_rbf = schnetpackGaussianRBF(
        n_rbf=number_of_gaussians,
        cutoff=cutoff.to(unit.angstrom).m,
        start=start.to(unit.angstrom).m,
    )
    radial_symmetry_function_module = SchnetRadialSymmetryFunction(
        number_of_radial_basis_functions=number_of_gaussians,
        max_distance=cutoff,
        min_distance=start,
    )

    # compare the output
    r = torch.rand(5, 1)
    assert torch.allclose(
        schnetpack_rbf(r),
        radial_symmetry_function_module(r / 10).unsqueeze(1),
        atol=1e-5,
    )  # NOTE: there is a shape mismatch between the two outputs
    assert (
        schnetpack_rbf.n_rbf
        == radial_symmetry_function_module.number_of_radial_basis_functions
    )


from .precalculated_values import setup_single_methane_input


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
                "function": "from_atom_to_molecule",
                "mode": "sum",
                "in": 'per_atom_energy',
                "index_key": "atomic_subsystem_indices",
                "out": "E",
            }
        ],
    )


@pytest.mark.xfail
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
    # test neighborlist and distance
    # ---------------------------------------- #
    assert torch.allclose(spk_input["_Rij"] / 10, pain_nn_input_mf.r_ij, atol=1e-4)
    assert torch.allclose(spk_input["_idx_i"], pain_nn_input_mf.pair_indices[0])
    assert torch.allclose(spk_input["_idx_j"], pain_nn_input_mf.pair_indices[1])
    idx_i = spk_input["_idx_i"]
    idx_j = spk_input["_idx_j"]

    # ---------------------------------------- #
    # test radial symmetry function
    # ---------------------------------------- #
    r_ij = spk_input["_Rij"]
    d_ij = torch.norm(r_ij, dim=1, keepdim=True)
    dir_ij = r_ij / d_ij
    schnetpack_phi_ij = schnetpack_painn.radial_basis(d_ij)
    modelforge_phi_ij = modelforge_painn.core_module.representation_module.radial_symmetry_function_module(
        d_ij / 10
    ).unsqueeze(
        1
    )  # NOTE: for the sake of comparision, changing the shape here  # NOTE: converting to nm

    assert torch.allclose(schnetpack_phi_ij, modelforge_phi_ij)
    phi_ij = schnetpack_phi_ij
    # ---------------------------------------- #
    # test cutoff
    # ---------------------------------------- #
    fcut_spk = schnetpack_painn.cutoff_fn(d_ij)
    fcut_mf = modelforge_painn.core_module.representation_module.cutoff_module(
        d_ij / 10
    )  # NOTE: converting to nm
    assert torch.allclose(fcut_spk, fcut_mf)

    # ---------------------------------------- #
    # test filter
    # ---------------------------------------- #
    filters_spk = schnetpack_painn.filter_net(phi_ij) * fcut_spk[..., None]
    filters_mf = schnetpack_painn.filter_net(phi_ij) * fcut_mf[..., None]
    assert torch.allclose(filters_spk, filters_mf)

    # ---------------------------------------- #
    # test embedding
    # ---------------------------------------- #
    import schnetpack.properties as properties

    assert torch.allclose(
        spk_input[properties.Z].to(torch.int32), mf_nnp_input.atomic_numbers.squeeze()
    )
    embedding_spk = schnetpack_painn.embedding(spk_input[properties.Z])
    embedding_mf = modelforge_painn.core_module.embedding_module(
        mf_nnp_input.atomic_numbers
    )

    assert torch.allclose(embedding_spk, embedding_mf)
    # ---------------------------------------- #
    # test interaction
    # ---------------------------------------- #
    # compare dimensions of q and mu in spk and modelforge
    q_spk_initial = embedding_spk[:, None]
    spk_qs = q_spk_initial.shape
    q_mf_initial = embedding_mf[:, None]
    mf_qs = q_mf_initial.shape
    assert spk_qs == mf_qs
    assert torch.allclose(q_spk_initial, q_mf_initial)

    mu_spk_initial = torch.zeros((spk_qs[0], 3, spk_qs[2]))
    mu_mf_initial = torch.zeros((mf_qs[0], 3, mf_qs[2]))
    assert mu_spk_initial.shape == mu_mf_initial.shape

    # set up the filter and interaction, pass the input and compare the results
    # ---------------------------------------- #
    assert torch.allclose(q_mf_initial, q_spk_initial)
    # reset parameters
    torch.manual_seed(1234)
    [
        dense.reset_parameters()
        for i in range(nr_of_interactions)
        for dense in modelforge_painn.core_module.interaction_modules[i].interatomic_net
    ]
    torch.manual_seed(1234)
    [
        dense.reset_parameters()
        for i in range(nr_of_interactions)
        for dense in schnetpack_painn.interactions[i].interatomic_context_net
    ]
    print(modelforge_painn.core_module.interaction_modules[0].interatomic_net[0].weight)
    print(schnetpack_painn.interactions[0].interatomic_context_net[0].weight)

    assert torch.allclose(
        modelforge_painn.core_module.interaction_modules[0].interatomic_net[0].weight,
        schnetpack_painn.interactions[0].interatomic_context_net[0].weight,
        atol=1e-4,
    )

    # first layer
    mf_intra_net = modelforge_painn.core_module.interaction_modules[0].interatomic_net
    spk_intra_net = schnetpack_painn.interactions[0].interatomic_context_net
    intra_mf_q = mf_intra_net(q_mf_initial)
    intra_spk_q = spk_intra_net(q_spk_initial)

    assert torch.allclose(intra_mf_q, intra_spk_q)
    # ---------------------------------------- #

    filter_list = torch.split(filters_spk, 3 * schnetpack_painn.n_atom_basis, dim=-1)
    n_atoms = spk_input[properties.Z].shape[0]
    q_spk, mu_spk = schnetpack_painn.interactions[0](
        q_spk_initial, mu_spk_initial, filter_list[0], dir_ij, idx_i, idx_j, n_atoms
    )
    torch.manual_seed(1234)
    pair_indices = pain_nn_input_mf.pair_indices
    filter_list = torch.split(
        filters_mf, 3 * modelforge_painn.core_module.number_of_atom_features, dim=-1
    )

    # test intra-atomic NNP
    q_mf, mu_mf = modelforge_painn.core_module.interaction_modules[0](
        q_mf_initial,
        mu_mf_initial,
        filter_list[0].squeeze(1),  # NOTE: change of shape
        dir_ij,
        pair_indices,
    )

    assert q_spk.shape == q_mf.shape
    assert mu_spk.shape == mu_mf.shape
    assert torch.allclose(q_spk, q_mf)
    assert torch.allclose(mu_spk, mu_mf)

    # ---------------------------------------- #
    # test mixing
    # ---------------------------------------- #
    # reset parameters
    torch.manual_seed(1234)
    [
        (
            modelforge_painn.core_module.mixing_modules[
                i
            ].mu_channel_mix.reset_parameters(),
            modelforge_painn.core_module.mixing_modules[i]
            .intra_atomic_net[0]
            .reset_parameters(),
            modelforge_painn.core_module.mixing_modules[i]
            .intra_atomic_net[1]
            .reset_parameters(),
        )
        for i in range(nr_of_interactions)
    ]

    torch.manual_seed(1234)

    [
        (
            schnetpack_painn.mixing[i].mu_channel_mix.reset_parameters(),
            schnetpack_painn.mixing[i].intraatomic_context_net[0].reset_parameters(),
            schnetpack_painn.mixing[i].intraatomic_context_net[1].reset_parameters(),
        )
        for i in range(nr_of_interactions)
    ]

    # test painn mixing
    for i in range(nr_of_interactions):
        print(i, flush=True)
        assert torch.allclose(
            modelforge_painn.core_module.mixing_modules[i].mu_channel_mix.weight,
            schnetpack_painn.mixing[i].mu_channel_mix.weight,
        )
        assert torch.allclose(
            modelforge_painn.core_module.mixing_modules[i].intra_atomic_net[0].weight,
            schnetpack_painn.mixing[i].intraatomic_context_net[0].weight,
        )
        assert torch.allclose(
            modelforge_painn.core_module.mixing_modules[i].intra_atomic_net[1].weight,
            schnetpack_painn.mixing[i].intraatomic_context_net[1].weight,
        )

    mixed_spk_q, mixed_spk_mu = schnetpack_painn.mixing[0](q_spk, mu_spk)
    mixed_mf_q, mixed_mf_mu = modelforge_painn.core_module.mixing_modules[0](
        q_mf, mu_mf
    )
    assert torch.allclose(mixed_mf_q, mixed_spk_q)
    assert torch.allclose(mixed_mf_mu, mixed_spk_mu)

    # -----------------------------
    # test one interaction and mixing pass
    # -----------------------------
    spk_filter_list = torch.split(
        filters_spk, 3 * schnetpack_painn.n_atom_basis, dim=-1
    )
    # q_spk = q_spk_initial
    # mu_spk = mu_spk_initial
    for i, (interaction, mixing) in enumerate(
        zip(schnetpack_painn.interactions[0:1], schnetpack_painn.mixing[0:1])
    ):
        q_spk, mu_spk = interaction(
            q_spk,
            mu_spk,
            spk_filter_list[i],
            dir_ij,
            idx_i,
            idx_j,
            n_atoms,
        )
        q_spk, mu_spk = mixing(q_spk, mu_spk)

    mf_filter_list = torch.split(
        filters_mf, 3 * modelforge_painn.core_module.number_of_atom_features, dim=-1
    )
    # q_mf = q_mf_initial
    # mu_mf = mu_mf_initial

    for i, (interaction, mixing) in enumerate(
        zip(
            modelforge_painn.core_module.interaction_modules[0:1],
            modelforge_painn.core_module.mixing_modules[0:1],
        )
    ):
        q_mf, mu_mf = interaction(
            q_mf, mu_mf, mf_filter_list[i].squeeze(1), dir_ij, pair_indices
        )
        q_mf, mu_mf = mixing(q_mf, mu_mf)

    assert torch.allclose(q_mf, q_spk)
    assert torch.allclose(mu_mf, mu_spk)

    # -----------------------------
    # test two interaction and mixing pass
    # -----------------------------
    spk_filter_list = torch.split(
        filters_spk, 3 * schnetpack_painn.n_atom_basis, dim=-1
    )
    mf_filter_list = torch.split(
        filters_mf, 3 * modelforge_painn.core_module.number_of_atom_features, dim=-1
    )

    # q_spk = q_spk_initial
    # mu_spk = mu_spk_initial

    for i, (spk_interaction, spk_mixing, mf_interaction, mf_mixing) in enumerate(
        zip(
            schnetpack_painn.interactions,
            schnetpack_painn.mixing,
            modelforge_painn.core_module.interaction_modules,
            modelforge_painn.core_module.mixing_modules,
        )
    ):
        q_spk, mu_spk = spk_interaction(
            q_spk,
            mu_spk,
            spk_filter_list[i],
            dir_ij,
            idx_i,
            idx_j,
            n_atoms,
        )
        q_spk, mu_spk = spk_mixing(q_spk, mu_spk)
        q_mf, mu_mf = mf_interaction(
            q_mf, mu_mf, mf_filter_list[i].squeeze(1), dir_ij, pair_indices
        )
        q_mf, mu_mf = mf_mixing(q_mf, mu_mf)
        assert torch.allclose(q_mf, q_spk)
        assert torch.allclose(mu_mf, mu_spk)

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #

    # reset filter parameters
    torch.manual_seed(1234)
    modelforge_painn.core_module.representation_module.filter_net.reset_parameters()
    torch.manual_seed(1234)
    schnetpack_painn.filter_net.reset_parameters()
    assert torch.allclose(
        modelforge_painn.core_module.representation_module.filter_net.weight,
        schnetpack_painn.filter_net.weight,
        atol=1e-4,
    )
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


@pytest.mark.xfail
def setup_mf_schnet_representation(
    cutoff: float,
    number_of_atom_features: int,
    number_of_radial_basis_functions: int,
    nr_of_interactions: int,
):
    # ------------------------------------ #
    # set up the modelforge Painn representation model
    # which means that we only want to call the
    # _transform_input() method
    from modelforge.potential.schnet import SchNet as mf_SchNET

    return mf_SchNET(
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
                "function": "from_atom_to_molecule",
                "mode": "sum",
                "in": 'per_atom_energy',
                "index_key": "atomic_subsystem_indices",
                "out": "E",
            }
        ],
    )


@pytest.mark.xfail
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
    schnetpack_schnet = setup_spk_schnet_representation(
        cutoff, number_of_atom_features, n_rbf, nr_of_interactions
    ).double()
    torch.manual_seed(1234)
    modelforge_schnet = setup_mf_schnet_representation(
        cutoff, number_of_atom_features, n_rbf, nr_of_interactions
    ).double()
    # ------------------------------------ #
    # set up the input for the spk Schnet model
    input = setup_single_methane_input()
    spk_input = input["spk_methane_input"]
    mf_nnp_input = input["modelforge_methane_input"]

    schnetpack_results = schnetpack_schnet(spk_input)
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
    idx_i = spk_input["_idx_i"]
    idx_j = spk_input["_idx_j"]

    # ---------------------------------------- #
    # test radial symmetry function
    # ---------------------------------------- #
    r_ij = spk_input["_Rij"]
    d_ij = torch.norm(r_ij, dim=1, keepdim=True)
    schnetpack_phi_ij = schnetpack_schnet.radial_basis(d_ij)
    modelforge_phi_ij = modelforge_schnet.core_module.schnet_representation_module.radial_symmetry_function_module(
        d_ij.unsqueeze(1) / 10
    )  # NOTE: converting to nm

    assert torch.allclose(schnetpack_phi_ij, modelforge_phi_ij)
    phi_ij = schnetpack_phi_ij
    # ---------------------------------------- #
    # test cutoff
    # ---------------------------------------- #
    fcut_spk = schnetpack_schnet.cutoff_fn(d_ij)
    fcut_mf = modelforge_schnet.core_module.schnet_representation_module.cutoff_module(
        d_ij / 10
    )  # NOTE: converting to nm
    assert torch.allclose(fcut_spk, fcut_mf)

    # ---------------------------------------- #
    # test embedding
    # ---------------------------------------- #
    import schnetpack.properties as properties

    assert torch.allclose(
        spk_input[properties.Z].to(torch.int),
        schnet_nn_input_mf.atomic_numbers.squeeze(),
    )
    embedding_spk = schnetpack_schnet.embedding(spk_input[properties.Z])
    embedding_mf = modelforge_schnet.core_module.embedding_module(
        schnet_nn_input_mf.atomic_numbers
    )

    assert torch.allclose(embedding_spk, embedding_mf)

    # --------------------------------------- #
    # test representation
    # --------------------------------------- #
    f_ij_mf = modelforge_schnet.core_module.schnet_representation_module.radial_symmetry_function_module(
        d_ij.unsqueeze(1) / 10
    )
    r_cut_ij_mf = (
        modelforge_schnet.core_module.schnet_representation_module.cutoff_module(
            d_ij / 10
        )
    )

    r_ij = spk_input["_Rij"]
    d_ij = torch.norm(r_ij, dim=1)
    f_ij_spk = schnetpack_schnet.radial_basis(d_ij)
    rcut_ij_spk = schnetpack_schnet.cutoff_fn(d_ij)

    f_ij_mf_ = f_ij_mf.squeeze(1)
    r_cut_ij_mf_ = r_cut_ij_mf.squeeze(1)
    assert torch.allclose(f_ij_mf_, f_ij_spk)
    assert torch.allclose(r_cut_ij_mf_, rcut_ij_spk)

    # ---------------------------------------- #
    # test interactions
    # ---------------------------------------- #

    # reset all parameters
    torch.manual_seed(1234)
    for i in range(nr_of_interactions):
        schnetpack_schnet.interactions[i].in2f.reset_parameters()
        for j in range(2):
            schnetpack_schnet.interactions[i].f2out[j].reset_parameters()
            schnetpack_schnet.interactions[i].filter_network[j].reset_parameters()

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

    assert torch.allclose(
        schnetpack_schnet.interactions[0].filter_network[0].weight,
        modelforge_schnet.core_module.interaction_modules[0].filter_network[0].weight,
    )
    assert torch.allclose(
        schnetpack_schnet.interactions[0].filter_network[0].bias,
        modelforge_schnet.core_module.interaction_modules[0].filter_network[0].bias,
    )

    assert torch.allclose(
        schnetpack_schnet.interactions[0].filter_network[1].weight,
        modelforge_schnet.core_module.interaction_modules[0].filter_network[1].weight,
    )
    assert torch.allclose(
        schnetpack_schnet.interactions[0].filter_network[1].bias,
        modelforge_schnet.core_module.interaction_modules[0].filter_network[1].bias,
    )

    assert torch.allclose(embedding_spk, embedding_mf)

    for mf_interaction, spk_interaction in zip(
        modelforge_schnet.core_module.interaction_modules,
        schnetpack_schnet.interactions,
    ):
        v_spk = spk_interaction(
            embedding_spk,
            f_ij_spk,
            spk_input["_idx_i"],
            spk_input["_idx_j"],
            rcut_ij_spk,
        )
        v_mf = mf_interaction(
            embedding_mf, schnet_nn_input_mf.pair_indices, f_ij_mf, r_cut_ij_mf
        )

        assert torch.allclose(v_spk, v_mf)

    # Check full pass
    modelforge_results = modelforge_schnet.core_module._forward(schnet_nn_input_mf)
    schnetpack_results = schnetpack_schnet(spk_input)

    assert (
        schnetpack_results["scalar_representation"].shape
        == modelforge_results["q"].shape
    )

    scalar_spk = schnetpack_results["scalar_representation"]
    scalar_mf = modelforge_results["q"]
    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)

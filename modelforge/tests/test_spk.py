import torch


def test_gaussian_rbf_implementation():
    # compare schnetpack GaussianRBF with modelforge GaussianRBF
    from modelforge.potential.utils import GaussianRBF
    from schnetpack.nn import GaussianRBF as schnetpackGaussianRBF
    from openff.units import unit

    n_rbf = 2
    cutoff = unit.Quantity(5.0, unit.angstrom)
    schnetpack_rbf = schnetpackGaussianRBF(
        n_rbf=n_rbf, cutoff=cutoff.to(unit.angstrom).m
    )
    rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    r = torch.rand(5, 3)
    print(schnetpack_rbf(r))
    print(rbf(r / 10))
    assert torch.allclose(schnetpack_rbf(r), rbf(r / 10), atol=1e-8)
    assert schnetpack_rbf.n_rbf == rbf.n_rbf


def setup_input():
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
    atomic_numbers = torch.tensor([[6], [1], [1], [1], [1]], dtype=torch.int64)

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
    E_labels = torch.tensor([0.0], requires_grad=True)
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    modelforge_methane = {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "E_labels": E_labels,
        "atomic_subsystem_indices": atomic_subsystem_indices,
    }
    # ------------------------------------ #

    return {
        "spk_methane_input": methan_spk,
        "modelforge_methane_input": modelforge_methane,
    }


def setup_spk_painn_representation(cutoff, nr_atom_basis, n_rbf):
    # ------------------------------------ #
    # set up the schnetpack Painn representation model
    from schnetpack.nn import GaussianRBF, CosineCutoff
    from schnetpack.representation import PaiNN as schnetpack_PaiNN
    from openff.units import unit

    radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff.to(unit.angstrom).m)
    return schnetpack_PaiNN(
        n_atom_basis=nr_atom_basis,
        n_interactions=2,
        radial_basis=radial_basis,
        cutoff_fn=CosineCutoff(cutoff.to(unit.angstrom).m),
    )


def setup_modelforge_painn_representation(cutoff, nr_atom_basis, n_rbf):
    # ------------------------------------ #
    # set up the modelforge Painn representation model
    # which means that we only want to call the
    # _transform_input() method
    from modelforge.potential import CosineCutoff, GaussianRBF
    from modelforge.potential.utils import SlicedEmbedding
    from modelforge.potential.painn import PaiNN
    from openff.units import unit

    embedding = SlicedEmbedding(max_Z=100, embedding_dim=nr_atom_basis, sliced_dim=0)
    radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    cutoff = CosineCutoff(cutoff)

    return PaiNN(
        embedding_module=embedding,
        nr_interaction_blocks=2,
        radial_basis_module=radial_basis,
        cutoff_module=cutoff,
    )


def test_painn_representation_implementation():
    # ---------------------------------------- #
    # test the implementation of the representation part of the PaiNN model
    # ---------------------------------------- #
    from openff.units import unit

    cutoff = unit.Quantity(5.0, unit.angstrom)
    nr_atom_basis = 8
    n_rbf = 5
    torch.manual_seed(1234)
    schnetpack_painn = setup_spk_painn_representation(
        cutoff, nr_atom_basis, n_rbf
    ).double()
    torch.manual_seed(1234)
    modelforge_painn = setup_modelforge_painn_representation(
        cutoff, nr_atom_basis, n_rbf
    ).double()
    print(schnetpack_painn)
    print(modelforge_painn)
    # ------------------------------------ #
    # set up the input for the spk Painn model
    input = setup_input()
    spk_input = input["spk_methane_input"]
    modelforge_input = input["modelforge_methane_input"]

    schnetpack_results = schnetpack_painn(spk_input)
    modelforge_painn._set_dtype()
    modelforge_input_1 = modelforge_painn._input_checks(modelforge_input)
    modelforge_input_2 = modelforge_painn.prepare_inputs(modelforge_input_1)

    # ---------------------------------------- #
    # test neighborlist and distance
    # ---------------------------------------- #
    assert torch.allclose(spk_input["_Rij"] / 10, modelforge_input_2["r_ij"], atol=1e-4)
    assert torch.allclose(spk_input["_idx_i"], modelforge_input_2["pair_indices"][0])
    assert torch.allclose(spk_input["_idx_j"], modelforge_input_2["pair_indices"][1])
    idx_i = spk_input["_idx_i"]
    idx_j = spk_input["_idx_j"]

    # ---------------------------------------- #
    # test rbf
    # ---------------------------------------- #
    r_ij = spk_input["_Rij"]
    d_ij = torch.norm(r_ij, dim=1, keepdim=True)
    dir_ij = r_ij / d_ij
    schnetpack_phi_ij = schnetpack_painn.radial_basis(d_ij)
    modelforge_phi_ij = modelforge_painn.radial_basis_module(
        d_ij / 10
    )  # NOTE: converting to nm

    assert torch.allclose(schnetpack_phi_ij, modelforge_phi_ij)
    phi_ij = schnetpack_phi_ij
    # ---------------------------------------- #
    # test cutoff
    # ---------------------------------------- #
    fcut_spk = schnetpack_painn.cutoff_fn(d_ij)
    fcut_mf = modelforge_painn.cutoff_module(d_ij / 10)  # NOTE: converting to nm
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
        spk_input[properties.Z], modelforge_input["atomic_numbers"].squeeze()
    )
    embedding_spk = schnetpack_painn.embedding(spk_input[properties.Z])
    embedding_mf = modelforge_painn.embedding_module(modelforge_input["atomic_numbers"])
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
    mf_intra_net = modelforge_painn.interaction_modules[0].intra_atomic_net
    spk_intra_net = schnetpack_painn.interactions[0].interatomic_context_net
    assert torch.allclose(q_mf_initial, q_spk_initial)
    # reset parameters
    torch.manual_seed(1234)
    [dense.reset_parameters() for dense in mf_intra_net]
    torch.manual_seed(1234)
    [dense.reset_parameters() for dense in spk_intra_net]

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
    pair_indices = modelforge_input_2["pair_indices"]
    filter_list = torch.split(filters_mf, 3 * modelforge_painn.nr_atom_basis, dim=-1)

    # test intra-atomic NNP
    q_mf, mu_mf = modelforge_painn.interaction_modules[0](
        q_mf_initial,
        mu_mf_initial,
        filter_list[0],
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
        dense.reset_parameters()
        for dense in modelforge_painn.mixing_modules[0].intra_atomic_net
    ]
    modelforge_painn.mixing_modules[0].mu_channel_mix.reset_parameters()
    torch.manual_seed(1234)
    [
        dense.reset_parameters()
        for dense in schnetpack_painn.mixing[0].intraatomic_context_net
    ]
    schnetpack_painn.mixing[0].mu_channel_mix.reset_parameters()

    mixed_spk_q, mixed_spk_mu = schnetpack_painn.mixing[0](q_spk, mu_spk)
    mixed_mf_q, mixed_mf_mu = modelforge_painn.mixing_modules[0](q_mf, mu_mf)
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

    mf_filter_list = torch.split(filters_mf, 3 * modelforge_painn.nr_atom_basis, dim=-1)
    # q_mf = q_mf_initial
    # mu_mf = mu_mf_initial

    for i, (interaction, mixing) in enumerate(
        zip(
            modelforge_painn.interaction_modules[0:1],
            modelforge_painn.mixing_modules[0:1],
        )
    ):
        q_mf, mu_mf = interaction(q_mf, mu_mf, mf_filter_list[i], dir_ij, pair_indices)
        q_mf, mu_mf = mixing(q_mf, mu_mf)

    assert torch.allclose(q_mf, q_spk)
    assert torch.allclose(mu_mf, mu_spk)

    # -----------------------------
    # test two interaction and mixing pass
    # -----------------------------
    spk_filter_list = torch.split(
        filters_spk, 3 * schnetpack_painn.n_atom_basis, dim=-1
    )
    mf_filter_list = torch.split(filters_mf, 3 * modelforge_painn.nr_atom_basis, dim=-1)

    # q_spk = q_spk_initial
    # mu_spk = mu_spk_initial

    for i, (spk_interaction, spk_mixing, mf_interaction, mf_mixing) in enumerate(
        zip(
            schnetpack_painn.interactions,
            schnetpack_painn.mixing,
            modelforge_painn.interaction_modules,
            modelforge_painn.mixing_modules,
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
            q_mf, mu_mf, mf_filter_list[i], dir_ij, pair_indices
        )
        q_mf, mu_mf = mf_mixing(q_mf, mu_mf)
        assert torch.allclose(q_mf, q_spk)
        assert torch.allclose(mu_mf, mu_spk)

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #

    modelforge_results = modelforge_painn._forward(modelforge_input_2)
    assert (
        schnetpack_results["scalar_representation"].shape
        == modelforge_results["scalar_representation"].shape
    )

    assert torch.allclose(
        schnetpack_results["scalar_representation"],
        modelforge_results["scalar_representation"],
    )

    # assert torch.allclose(
    #     schnetpack_results["vector_representation"],
    #     modelforge_results["vector_representation"],
    # )

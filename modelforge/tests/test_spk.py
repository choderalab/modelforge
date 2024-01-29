from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn


def test_gaussian_rbf_implementation():
    # compare schnetpack GaussianRBF with modelforge GaussianRBF
    from modelforge.potential.utils import GaussianRBF
    from schnetpack.nn import GaussianRBF as schnetpack_GaussianRBF

    n_rbf = 2
    cutoff = 5.0
    schnetpack_rbf = schnetpack_GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    rbf = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    r = torch.rand(5, 3)
    schnetpack_rbf(r)
    rbf(r)
    assert torch.allclose(schnetpack_rbf(r), rbf(r))
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

    positions = torch.tensor(
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


def setup_spk_painn_representation(cutoff, n_atom_basis, n_rbf):
    # ------------------------------------ #
    # set up the schnetpack Painn representation model
    from schnetpack.nn import GaussianRBF, CosineCutoff
    from schnetpack.representation import PaiNN as schnetpack_PaiNN
    from schnetpack.atomistic import PairwiseDistances

    radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    return schnetpack_PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=CosineCutoff(cutoff),
    )


def setup_modelforge_painn_representation(cutoff, n_atom_basis, n_rbf):
    # ------------------------------------ #
    # set up the modelforge Painn representation model
    # which means that we only want to call the
    # _transform_input() method
    from modelforge.potential import CosineCutoff, GaussianRBF
    from modelforge.potential.utils import SlicedEmbedding
    from modelforge.potential.painn import PaiNN

    embedding = SlicedEmbedding(max_Z=100, embedding_dim=n_atom_basis, sliced_dim=0)
    radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    cutoff = CosineCutoff(cutoff)

    return PaiNN(
        embedding=embedding,
        nr_interaction_blocks=3,
        radial_basis=radial_basis,
        cutoff=cutoff,
    )


def test_painn_representation_implementation():
    # test the implementation of the representation part of the PaiNN model

    from modelforge.potential.painn import PaiNN as modelforge_PaiNN

    cutoff = 5.0
    n_atom_basis = 8
    n_rbf = 5

    schnetpack_painn = setup_spk_painn_representation(
        cutoff, n_atom_basis, n_rbf
    ).double()
    modelforge_painn = setup_modelforge_painn_representation(
        cutoff, n_atom_basis, n_rbf
    ).double()

    # ------------------------------------ #
    # set up the input for the spk Painn model
    input = setup_input()
    spk_input = input["spk_methane_input"]
    modelforge_input = input["modelforge_methane_input"]

    schnetpack_results = schnetpack_painn(spk_input)
    modelforge_painn._set_dtype()
    modelforge_input_1 = modelforge_painn.input_checks(modelforge_input)
    modelforge_input_2 = modelforge_painn._prepare_input(modelforge_input_1)

    assert torch.allclose(spk_input["_Rij"], modelforge_input_2["r_ij"], atol=1e-4)
    assert torch.allclose(spk_input["_idx_i"], modelforge_input_2["pair_indices"][0])
    assert torch.allclose(spk_input["_idx_j"], modelforge_input_2["pair_indices"][1])

    modelforge_results = modelforge_painn(modelforge_results)

    a = 7

from typing import Dict

import torch

from modelforge.potential.schnet import Schnet

from .helper_functinos import generate_methane_input


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    schnet = Schnet(128, 6, 2)
    assert schnet is not None, "Schnet model should be initialized."


def test_schnet_forward():
    """
    Test the forward pass of the Schnet model.
    """
    model = Schnet(128, 3)
    inputs = {
        "Z": torch.tensor([[1, 2], [2, 3]]),
        "R": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        ),
    }
    energy = model(inputs)
    assert energy.shape == (
        2,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_calculate_energies_and_forces():
    """
    Test the calculation of energies and forces for a molecule.
    This test will be adapted once we have a trained model.
    """

    schnet = Schnet(128, 6, 64)
    methane_inputs = generate_methane_input()
    result = schnet(methane_inputs)
    forces = -torch.autograd.grad(
        result, methane_inputs["R"], create_graph=True, retain_graph=True
    )[0]

    assert result.shape == (1, 1)  #  only one molecule
    assert forces.shape == (1, 5, 3)  #  only one molecule


def get_input_for_interaction_block(
    nr_atom_basis: int, nr_embeddings: int
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for testing the SchNet interaction block.

    Parameters
    ----------
    nr_atom_basis : int
        Number of atom basis.
    nr_embeddings : int
        Number of embeddings.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing tensors for the interaction block test.
    """

    import torch.nn as nn

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.potential.utils import GaussianRBF, _distance_to_radial_basis

    from .helper_functinos import prepare_pairlist_for_single_batch, return_single_batch

    embedding = nn.Embedding(nr_embeddings, nr_atom_basis, padding_idx=0)
    batch = return_single_batch(QM9Dataset, "fit")
    pairlist = prepare_pairlist_for_single_batch(batch)
    radial_basis = GaussianRBF(n_rbf=20, cutoff=5.0)

    atom_index12 = pairlist["atom_index12"]
    d_ij = pairlist["d_ij"]
    f_ij, rcut_ij = _distance_to_radial_basis(d_ij, radial_basis)
    return {
        "x": embedding(batch["Z"]),
        "f_ij": f_ij,
        "idx_i": atom_index12[0],
        "idx_j": atom_index12[1],
        "rcut_ij": rcut_ij,
    }


def test_schnet_interaction_layer():
    """
    Test the SchNet interaction layer.
    """
    from modelforge.potential.schnet import SchNetInteractionBlock

    nr_atom_basis = 128
    nr_embeddings = 100
    r = get_input_for_interaction_block(nr_atom_basis, nr_embeddings)
    assert r["x"].shape == (
        64,
        17,
        nr_atom_basis,
    ), "Input shape mismatch for x tensor."
    interaction = SchNetInteractionBlock(nr_atom_basis, 4)
    v = interaction(r["x"], r["f_ij"], r["idx_i"], r["idx_j"], r["rcut_ij"])
    assert v.shape == (
        64,
        17,
        nr_atom_basis,
    ), "Output shape mismatch for v tensor."


# def test_schnet_reimplementation_against_original_implementation():
#    import numpy as np
#    np.load('tests/qm9tut/split.npz')['train_idx']

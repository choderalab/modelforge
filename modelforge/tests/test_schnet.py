from loguru import logger

from modelforge.potential.schnet import Schnet

from .helper_functinos import generate_methane_input
import torch


def test_Schnet_init():
    schnet = Schnet(128, 6, 2)
    assert schnet is not None


def test_schnet_forward():
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
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    schnet = Schnet(128, 6, 64)
    methane_inputs = generate_methane_input()
    result = schnet(methane_inputs)
    forces = -torch.autograd.grad(
        result, methane_inputs["R"], create_graph=True, retain_graph=True
    )[0]

    assert result.shape == (1, 1)  #  only one molecule
    assert forces.shape == (1, 5, 3)  #  only one molecule


def get_input_for_interaction_block(nr_atom_basis: int, nr_embeddings: int):
    from .helper_functinos import prepare_pairlist_for_single_batch, return_single_batch
    from modelforge.potential.schnet import SchNetInteractionBlock, SchNetRepresentation
    from modelforge.dataset.qm9 import QM9Dataset
    import torch.nn as nn

    embedding = nn.Embedding(nr_embeddings, nr_atom_basis, padding_idx=0)
    batch = return_single_batch(QM9Dataset, "fit")
    pairlist = prepare_pairlist_for_single_batch(batch)
    representation = SchNetRepresentation(nr_atom_basis, 4, 3)
    atom_index12 = pairlist["atom_index12"]
    d_ij = pairlist["d_ij"]
    f_ij, rcut_ij = representation._distance_to_radial_basis(d_ij)
    return {
        "x": embedding(batch["Z"]),
        "f_ij": f_ij,
        "idx_i": atom_index12[0],
        "idx_j": atom_index12[1],
        "rcut_ij": rcut_ij,
    }


def test_schnet_interaction_layer():
    from modelforge.potential.schnet import SchNetInteractionBlock, SchNetRepresentation

    nr_atom_basis = 128
    nr_embeddings = 100
    r = get_input_for_interaction_block(nr_atom_basis, nr_embeddings)
    assert r["x"].shape == (64, 17, nr_atom_basis)
    interaction = SchNetInteractionBlock(nr_atom_basis, 4)
    v = interaction(r["x"], r["f_ij"], r["idx_i"], r["idx_j"], r["rcut_ij"])
    assert v.shape == (64, 17, nr_atom_basis)

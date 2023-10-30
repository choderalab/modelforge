import torch

from modelforge.potential.schnet import SchNET

import pytest
from .helper_functions import (
    SIMPLIFIED_INPUT_DATA,
)

nr_atom_basis = 128
nr_embeddings = 100


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    schnet = SchNET(128, 6, 2)
    assert schnet is not None, "Schnet model should be initialized."


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
def test_schnet_forward(input_data):
    """
    Test the forward pass of the Schnet model.
    """
    model = SchNET(128, 3)
    energy = model(input_data)
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert energy.shape == (
        nr_of_mols,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_schnet_interaction_layer():
    """
    Test the SchNet interaction layer.
    """
    from modelforge.potential.schnet import SchNETInteractionBlock

    nr_atom_basis = 128
    nr_embeddings = 100

    interaction_data = generate_interaction_block_data(nr_atom_basis, nr_embeddings)
    nr_of_atoms_per_batch = interaction_data["atomic_subsystem_indices"].shape[0]

    assert interaction_data["x"].shape == (
        nr_of_atoms_per_batch,
        1,
        nr_atom_basis,
    ), "Input shape mismatch for x tensor."
    interaction = SchNETInteractionBlock(nr_atom_basis, 4)
    v = interaction(
        interaction_data["x"],
        interaction_data["pair_indices"],
        interaction_data["f_ij"],
        interaction_data["rcut_ij"],
    )
    assert v.shape == (
        nr_of_atoms_per_batch,
        1,
        nr_atom_basis,
    ), "Output shape mismatch for v tensor."


# def test_schnet_reimplementation_against_original_implementation():
#    import numpy as np
#    np.load('tests/qm9tut/split.npz')['train_idx']

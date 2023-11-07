from modelforge.potential.schnet import SchNET

import pytest
from .helper_functions import (
    setup_simple_model,
    SIMPLIFIED_INPUT_DATA,
    generate_interaction_block_data,
)


@pytest.mark.parametrize("lightning", [True, False])
def test_Schnet_init(lightning):
    """Test initialization of the Schnet model."""
    schnet = setup_simple_model(SchNET, lightning=lightning)
    assert schnet is not None, "Schnet model should be initialized."


@pytest.mark.parametrize("lightning", [True, False])
@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize(
    "model_parameter",
    ([64, 50, 20, 5.0, 2], [32, 60, 10, 7.0, 1], [128, 120, 64, 5.0, 3]),
)
def test_schnet_forward(lightning, input_data, model_parameter):
    """
    Test the forward pass of the Schnet model.
    """
    print(f'model_parameter: {model_parameter}')
    (
        nr_atom_basis,
        max_atomic_number,
        n_rbf,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    schnet = setup_simple_model(
        SchNET,
        lightning=lightning,
        nr_atom_basis=nr_atom_basis,
        max_atomic_number=max_atomic_number,
        n_rbf=n_rbf,
        cutoff=cutoff,
        nr_interaction_blocks=nr_interaction_blocks,
    )
    energy = schnet(input_data)
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

    nr_atom_basis = 127
    nr_embeddings = 97
    nr_rbf = 19

    interaction_data = generate_interaction_block_data(
        nr_atom_basis, nr_embeddings, nr_rbf
    )
    nr_of_atoms_per_batch = interaction_data["atomic_subsystem_indices"].shape[0]

    assert interaction_data["x"].shape == (
        nr_of_atoms_per_batch,
        nr_atom_basis,
    ), "Input shape mismatch for x tensor."
    interaction = SchNETInteractionBlock(
        nr_atom_basis=nr_atom_basis, nr_filters=3, nr_rbf=nr_rbf
    )
    v = interaction(
        interaction_data["x"],
        interaction_data["pair_indices"],
        interaction_data["f_ij"],
        interaction_data["rcut_ij"],
    )
    assert v.shape == (
        nr_of_atoms_per_batch,
        nr_atom_basis,
    ), "Output shape mismatch for v tensor."


# def test_schnet_reimplementation_against_original_implementation():
#    import numpy as np
#    np.load('tests/qm9tut/split.npz')['train_idx']

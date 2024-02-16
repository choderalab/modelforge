import os

import pytest

from modelforge.potential.sake import SAKE
from .helper_functions import (
    setup_simple_model,
    SIMPLIFIED_INPUT_DATA,
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize("lightning", [True, False])
def test_SAKE_init(lightning):
    """Test initialization of the SAKE neural network potential."""

    sake = setup_simple_model(SAKE, lightning=lightning)
    assert sake is not None, "SAKE model should be initialized."


from openff.units import unit


@pytest.mark.parametrize("lightning", [True, False])
@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize(
    "model_parameter",
    (
            [64, 50, 2, unit.Quantity(5.0, unit.angstrom), 2],
            [32, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
            [128, 120, 5, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_sake_forward(lightning, input_data, model_parameter):
    """
    Test the forward pass of the SAKE model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        n_rbf,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    sake = setup_simple_model(
        SAKE,
        lightning=lightning,
        nr_atom_basis=nr_atom_basis,
        max_atomic_number=max_atomic_number,
        n_rbf=n_rbf,
        cutoff=cutoff,
        nr_interaction_blocks=nr_interaction_blocks,
    )
    energy = sake(input_data)
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert energy.shape == (
        nr_of_mols,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_sake_interaction_equivariance():
    import torch
    from .helper_functions import generate_methane_input, setup_simple_model
    from modelforge.potential.sake import SAKE

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sake = setup_simple_model(SAKE)
    methane_input = generate_methane_input()
    perturbed_methane_input = methane_input.copy()
    perturbed_methane_input["positions"] = torch.matmul(
        methane_input["positions"], rotation_matrix
    )

    # prepare reference and perturbed inputs
    reference_prepared_input = sake.prepare_inputs(methane_input)
    reference_v = torch.randn_like(reference_prepared_input["positions"])
    reference_d_ij = reference_prepared_input["d_ij"]
    reference_r_ij = reference_prepared_input["r_ij"]
    reference_dir_ij = reference_r_ij / reference_d_ij

    perturbed_prepared_input = sake.prepare_inputs(perturbed_methane_input)
    perturbed_v = torch.matmul(reference_v, rotation_matrix)
    perturbed_d_ij = perturbed_prepared_input["d_ij"]
    perturbed_r_ij = perturbed_prepared_input["r_ij"]
    perturbed_dir_ij = perturbed_r_ij / perturbed_d_ij

    # check that the invariant properties are preserved
    # d_ij is the distance between atom i and j
    assert torch.allclose(reference_d_ij, perturbed_d_ij)

    # what shoudl not be invariant is the direction
    assert not torch.allclose(reference_dir_ij, perturbed_dir_ij)

    # Check for equivariance
    # rotate the reference dir_ij
    rotated_reference_dir_ij = torch.matmul(reference_dir_ij, rotation_matrix)
    # Compare the rotated original dir_ij with the dir_ij from rotated positions
    assert torch.allclose(rotated_reference_dir_ij, perturbed_dir_ij)

    # Test that the interaction block is equivariant
    sake_interaction = sake.interaction_modules[0]

    reference_r = sake_interaction(
        reference_prepared_input["atomic_embedding"],
        reference_prepared_input["positions"],
        reference_v,
        reference_prepared_input["pair_indices"]
    )

    perturbed_r = sake_interaction(
        perturbed_prepared_input["atomic_embedding"],
        perturbed_prepared_input["positions"],
        perturbed_v,
        perturbed_prepared_input["pair_indices"],
    )

    perturbed_h, perturbed_x, perturbed_v = perturbed_r
    reference_h, reference_x, reference_v = reference_r

    # x and v are equivariant, h is invariant
    assert torch.allclose(reference_h, perturbed_h)
    assert torch.allclose(torch.matmul(reference_x, rotation_matrix), perturbed_x)
    assert torch.allclose(torch.matmul(reference_v, rotation_matrix), perturbed_v)

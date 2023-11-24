import os

import pytest

from modelforge.potential.painn import PaiNN
from .helper_functions import (
    setup_simple_model,
    SIMPLIFIED_INPUT_DATA,
)


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.parametrize("lightning", [True, False])
def test_PaiNN_init(lightning):
    """Test initialization of the PaiNN neural network potential."""

    painn = setup_simple_model(PaiNN, lightning=lightning)
    assert painn is not None, "PaiNN model should be initialized."


@pytest.mark.parametrize("lightning", [True, False])
@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize(
    "model_parameter",
    ([64, 50, 2, 5.0, 2], [32, 60, 10, 7.0, 1], [128, 120, 5, 5.0, 3]),
)
def test_painn_forward(lightning, input_data, model_parameter):
    """
    Test the forward pass of the Schnet model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        n_rbf,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    painn = setup_simple_model(
        PaiNN,
        lightning=lightning,
        nr_atom_basis=nr_atom_basis,
        max_atomic_number=max_atomic_number,
        n_rbf=n_rbf,
        cutoff=cutoff,
        nr_interaction_blocks=nr_interaction_blocks,
    )
    energy = painn(input_data)
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert energy.shape == (
        nr_of_mols,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_painn_interaction_invariance():
    import torch
    from modelforge.potential.painn import PaiNNInteraction
    from torch import nn
    from .helper_functions import generate_methane_input, setup_simple_model
    from modelforge.potential.painn import PaiNN
    from modelforge.potential.utils import _distance_to_radial_basis

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    painn = setup_simple_model(PaiNN)
    methane_input = generate_methane_input()
    perturbed_methane_input = methane_input.copy()
    perturbed_methane_input["positions"] = torch.matmul(
        methane_input["positions"], rotation_matrix
    )

    prepared_input = painn._prepare_input(methane_input)
    reference_d_ij = prepared_input["d_ij"]
    reference_r_ij = prepared_input["r_ij"]
    atomic_numbers_embedding = prepared_input["atomic_numbers_embedding"]
    reference_dir_ij = reference_r_ij / reference_d_ij
    reference_f_ij, _ = _distance_to_radial_basis(reference_d_ij, painn.radial_basis)

    perturbed_prepared_input = painn._prepare_input(perturbed_methane_input)
    perturbed_d_ij = perturbed_prepared_input["d_ij"]
    perturbed_r_ij = perturbed_prepared_input["r_ij"]
    perturbed_dir_ij = perturbed_r_ij / perturbed_d_ij
    perturbed_f_ij, _ = _distance_to_radial_basis(perturbed_d_ij, painn.radial_basis)

    assert torch.allclose(reference_d_ij, perturbed_d_ij)
    assert torch.allclose(reference_f_ij, perturbed_f_ij)
    # Test that the interaction block is invariant to the order of the atoms


    
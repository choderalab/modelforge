from modelforge.potential.schnet import SchNet

import pytest
from .helper_functions import (
    setup_simple_model,
    SIMPLIFIED_INPUT_DATA,
    generate_interaction_block_data,
)


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    schnet = setup_simple_model(SchNet)
    assert schnet is not None, "Schnet model should be initialized."


from openff.units import unit


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize(
    "model_parameter",
    (
        [64, 50, 20, unit.Quantity(5.0, unit.angstrom), 2],
        [32, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
        [128, 120, 64, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_schnet_forward(input_data, model_parameter):
    """
    Test the forward pass of the Schnet model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        number_of_gaussians,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    schnet = setup_simple_model(
        SchNet,
        nr_atom_basis=nr_atom_basis,
        max_atomic_number=max_atomic_number,
        number_of_gaussians=number_of_gaussians,
        cutoff=cutoff,
        nr_interaction_blocks=nr_interaction_blocks,
    )
    energy = schnet(input_data)["E_predict"]
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch

from modelforge.potential.schnet import SchNet

import pytest


def test_Schnet_init():
    """Test initialization of the Schnet model."""
    from modelforge.potential.schnet import SchNet

    schnet = SchNet()
    assert schnet is not None, "Schnet model should be initialized."


from openff.units import unit


@pytest.mark.parametrize(
    "model_parameter",
    (
        [64, 50, 20, unit.Quantity(5.0, unit.angstrom), 2],
        [32, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
        [128, 120, 64, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_schnet_forward(single_batch_with_batchsize_64, model_parameter):
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
    schnet = SchNet(
        number_of_atom_features=nr_atom_basis,
        max_Z=max_atomic_number,
        number_of_radial_basis_functions=number_of_gaussians,
        cutoff=cutoff,
        number_of_interaction_modules=nr_interaction_blocks,
    )
    energy = schnet(single_batch_with_batchsize_64.nnp_input).E
    nr_of_mols = single_batch_with_batchsize_64.nnp_input.atomic_subsystem_indices.unique().shape[
        0
    ]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch

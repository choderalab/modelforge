import os

import pytest

from modelforge.potential.painn import PaiNN


def test_PaiNN_init():
    """Test initialization of the PaiNN neural network potential."""
    # read default parameters
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import potential_defaults

    file_path = resources.files(potential_defaults) / f"painn_defaults.toml"
    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    painn = PaiNN(**potential_parameter)
    assert painn is not None, "PaiNN model should be initialized."


from openff.units import unit


@pytest.mark.parametrize(
    "model_parameter",
    (
        [25, 50, 2, unit.Quantity(5.0, unit.angstrom), 2],
        [50, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
        [100, 120, 5, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_painn_forward(model_parameter, single_batch_with_batchsize_64):
    """
    Test the forward pass of the Schnet model.
    """
    import torch

    print(f"model_parameter: {model_parameter}")
    (
        max_Z,
        embedding_dimensions,
        number_of_gaussians,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    painn = PaiNN(
        max_Z=max_Z,
        number_of_atom_features=embedding_dimensions,
        number_of_radial_basis_functions=number_of_gaussians,
        cutoff=cutoff,
        number_of_interaction_modules=nr_interaction_blocks,
        shared_filters=False,
        shared_interactions=False,
    )
    nnp_input = single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32)
    energy = painn(nnp_input).E
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_painn_interaction_equivariance(single_batch_with_batchsize_64):
    from modelforge.potential.painn import PaiNN
    from dataclasses import replace
    import torch

    # read default parameters
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import potential_defaults

    file_path = resources.files(potential_defaults) / f"painn_defaults.toml"
    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )

    painn = PaiNN(**potential_parameter).to(torch.float64)
    methane_input = single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float64)
    perturbed_methane_input = replace(methane_input)
    perturbed_methane_input.positions = torch.matmul(
        methane_input.positions, rotation_matrix
    )

    # prepare reference and perturbed inputs
    pairlist_output = painn.input_preparation.prepare_inputs(methane_input)
    reference_prepared_input = painn.core_module._model_specific_input_preparation(
        methane_input, pairlist_output
    )

    reference_d_ij = reference_prepared_input.d_ij
    reference_r_ij = reference_prepared_input.r_ij
    reference_dir_ij = reference_r_ij / reference_d_ij
    reference_f_ij = (
        painn.core_module.representation_module.radial_symmetry_function_module(
            reference_d_ij
        )
    )

    pairlist_output = painn.input_preparation.prepare_inputs(perturbed_methane_input)
    perturbed_prepared_input = painn.core_module._model_specific_input_preparation(
        perturbed_methane_input, pairlist_output
    )

    perturbed_d_ij = perturbed_prepared_input.d_ij
    perturbed_r_ij = perturbed_prepared_input.r_ij
    perturbed_dir_ij = perturbed_r_ij / perturbed_d_ij
    perturbed_f_ij = (
        painn.core_module.representation_module.radial_symmetry_function_module(
            perturbed_d_ij
        )
    )

    # check that the invariant properties are preserved
    # d_ij is the distance between atom i and j
    # f_ij is the radial basis function of d_ij
    assert torch.allclose(reference_d_ij, perturbed_d_ij)
    assert torch.allclose(reference_f_ij, perturbed_f_ij)

    # what shoudl not be invariant is the direction
    assert not torch.allclose(reference_dir_ij, perturbed_dir_ij)

    # Check for equivariance
    # rotate the reference dir_ij
    rotated_reference_dir_ij = torch.matmul(reference_dir_ij, rotation_matrix)
    # Compare the rotated original dir_ij with the dir_ij from rotated positions
    assert torch.allclose(rotated_reference_dir_ij, perturbed_dir_ij)

    # Test that the interaction block is equivariant
    # First we test the transformed inputs
    reference_tranformed_inputs = painn.core_module.representation_module(
        reference_prepared_input
    )
    perturbed_tranformed_inputs = painn.core_module.representation_module(
        perturbed_prepared_input
    )

    assert torch.allclose(
        reference_tranformed_inputs["q"], perturbed_tranformed_inputs["q"]
    )
    assert torch.allclose(
        reference_tranformed_inputs["mu"], perturbed_tranformed_inputs["mu"]
    )

    painn_interaction = painn.core_module.interaction_modules[0]

    reference_r = painn_interaction(
        reference_tranformed_inputs["q"],
        reference_tranformed_inputs["mu"],
        reference_tranformed_inputs["filters"][0],
        reference_dir_ij,
        reference_prepared_input.pair_indices,
    )

    perturbed_r = painn_interaction(
        perturbed_tranformed_inputs["q"],
        perturbed_tranformed_inputs["mu"],
        reference_tranformed_inputs["filters"][0],
        perturbed_dir_ij,
        perturbed_prepared_input.pair_indices,
    )

    perturbed_q, perturbed_mu = perturbed_r
    reference_q, reference_mu = reference_r

    # mu is different, q is invariant
    assert torch.allclose(reference_q, perturbed_q)
    assert not torch.allclose(reference_mu, perturbed_mu)

    mixed_reference_q, mixed_reference_mu = painn.core_module.mixing_modules[0](
        reference_q, reference_mu
    )
    mixed_perturbed_q, mixed_perturbed_mu = painn.core_module.mixing_modules[0](
        perturbed_q, perturbed_mu
    )

    # q is a scalar property and invariant
    assert torch.allclose(mixed_reference_q, mixed_perturbed_q, atol=1e-2)
    # mu is a vector property and should not be invariant
    assert not torch.allclose(mixed_reference_mu, mixed_perturbed_mu)

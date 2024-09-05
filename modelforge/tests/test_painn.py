import torch
from modelforge.potential import NeuralNetworkPotentialFactory


def setup_painn_model(potential_seed: int):
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    # read default parameters
    config = load_configs_into_pydantic_models("painn", "qm9")
    # override defaults to match reference implementation in spk
    config["potential"].core_parameter.featurization.maximum_atomic_number = 100
    config["potential"].core_parameter.featurization.number_of_per_atom_features = 8
    config["potential"].core_parameter.number_of_radial_basis_functions = 5

    trainer_painn = NeuralNetworkPotentialFactory.generate_potential(
        use="training",
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
        potential_seed=potential_seed,
    ).model.potential
    return trainer_painn


def test_forward(single_batch_with_batchsize):
    """Test initialization of the PaiNN neural network potential."""
    trainer_painn = setup_painn_model()
    assert trainer_painn is not None, "PaiNN model should be initialized."
    batch = batch = single_batch_with_batchsize(batch_size=64, dataset_name="QM9")

    nnp_input = batch.to(dtype=torch.float32).nnp_input_tuple
    energy = trainer_painn(nnp_input)["per_molecule_energy"]
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_equivariance(single_batch_with_batchsize):
    from dataclasses import replace
    import torch

    batch = single_batch_with_batchsize(batch_size=64, dataset_name="QM9")

    potential = setup_painn_model().double()

    # define a rotation matrix in 3D that rotates by 90 degrees around the
    # z-axis (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )

    batch = batch.to(dtype=torch.float64)
    copy_of_batch = replace(batch)

    copy_of_batch.nnp_input.positions = torch.matmul(
        copy_of_batch.nnp_input.positions, rotation_matrix
    )

    methane_input = batch.nnp_input_tuple
    perturbed_methane_input = copy_of_batch.nnp_input_tuple

    # prepare reference and perturbed inputs
    pairlist_output = potential.neighborlist(methane_input)

    reference_d_ij = pairlist_output.d_ij
    reference_r_ij = pairlist_output.r_ij
    reference_dir_ij = reference_r_ij / reference_d_ij
    reference_f_ij = (
        potential.core_network.representation_module.radial_symmetry_function_module(
            reference_d_ij
        )
    )

    pairlist_output = potential.neighborlist(perturbed_methane_input)

    perturbed_d_ij = pairlist_output.d_ij
    perturbed_r_ij = pairlist_output.r_ij
    perturbed_dir_ij = perturbed_r_ij / perturbed_d_ij
    perturbed_f_ij = (
        potential.core_network.representation_module.radial_symmetry_function_module(
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
    reference_tranformed_inputs = potential.core_network.representation_module(
        reference_prepared_input
    )
    perturbed_tranformed_inputs = trainer_painn.core_module.representation_module(
        perturbed_prepared_input
    )

    assert torch.allclose(
        reference_tranformed_inputs["per_atom_scalar_feature"],
        perturbed_tranformed_inputs["per_atom_scalar_feature"],
    )
    assert torch.allclose(
        reference_tranformed_inputs["per_atom_vector_feature"],
        perturbed_tranformed_inputs["per_atom_vector_feature"],
    )

    painn_interaction = trainer_painn.core_module.interaction_modules[0]

    reference_r = painn_interaction(
        reference_tranformed_inputs["per_atom_scalar_feature"],
        reference_tranformed_inputs["per_atom_vector_feature"],
        reference_tranformed_inputs["filters"][0],
        reference_dir_ij,
        reference_prepared_input.pair_indices,
    )

    perturbed_r = painn_interaction(
        perturbed_tranformed_inputs["per_atom_scalar_feature"],
        perturbed_tranformed_inputs["per_atom_vector_feature"],
        reference_tranformed_inputs["filters"][0],
        perturbed_dir_ij,
        perturbed_prepared_input.pair_indices,
    )

    perturbed_q, perturbed_mu = perturbed_r
    reference_q, reference_mu = reference_r

    # mu is different, q is invariant
    assert torch.allclose(reference_q, perturbed_q)
    assert not torch.allclose(reference_mu, perturbed_mu)

    mixed_reference_q, mixed_reference_mu = trainer_painn.core_module.mixing_modules[0](
        reference_q, reference_mu
    )
    mixed_perturbed_q, mixed_perturbed_mu = trainer_painn.core_module.mixing_modules[0](
        perturbed_q, perturbed_mu
    )

    # q is a scalar property and invariant
    assert torch.allclose(mixed_reference_q, mixed_perturbed_q, atol=1e-2)
    # mu is a vector property and should not be invariant
    assert not torch.allclose(mixed_reference_mu, mixed_perturbed_mu)


import torch
from modelforge.tests.test_schnet import setup_single_methane_input


def test_compare_implementation_against_reference_implementation():
    # ---------------------------------------- #
    # setup the PaiNN model
    # ---------------------------------------- #
    from .precalculated_values import load_precalculated_painn_results

    potential = setup_painn_model(potential_seed=1234).double()

    # ------------------------------------ #
    # set up the input for the Painn model
    input = setup_single_methane_input()
    mf_nnp_input = input["modelforge_methane_input"]

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #

    # reset filter parameters
    torch.manual_seed(1234)
    potential.core_network.representation_module.filter_net.reset_parameters()

    calculated_results = potential.core_module.forward(prepared_input, pairlist_output)

    calculated_results = potential.core_module.forward(prepared_input, pairlist_output)
    reference_results = load_precalculated_painn_results()

    # check that the scalar and vector representations are the same
    # start with scalar representation
    assert (
        reference_results["scalar_representation"].shape
        == calculated_results["per_atom_scalar_representation"].shape
    )

    scalar_spk = reference_results["scalar_representation"].double()
    scalar_mf = calculated_results["per_atom_scalar_representation"].double()

    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)
    # check vector representation
    assert (
        reference_results["vector_representation"].shape
        == calculated_results["per_atom_vector_representation"].shape
    )

    assert torch.allclose(
        reference_results["vector_representation"].double(),
        calculated_results["per_atom_vector_representation"].double(),
        atol=1e-4,
    )

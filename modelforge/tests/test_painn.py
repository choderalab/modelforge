import torch
from modelforge.potential import NeuralNetworkPotentialFactory
import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_painn_temp")
    return fn


def setup_painn_model(potential_seed: int):
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    # read default parameters
    config = load_configs_into_pydantic_models("painn", "qm9")
    # override defaults to match reference implementation in spk
    config[
        "potential"
    ].core_parameter.featurization.atomic_number.maximum_atomic_number = 100
    config[
        "potential"
    ].core_parameter.featurization.atomic_number.number_of_per_atom_features = 8
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


def test_forward(single_batch_with_batchsize, prep_temp_dir):
    """Test initialization of the PaiNN neural network potential."""
    trainer_painn = setup_painn_model(42)
    assert trainer_painn is not None, "PaiNN model should be initialized."
    batch = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )

    nnp_input = batch.to_dtype(dtype=torch.float32).nnp_input
    energy = trainer_painn(nnp_input)["per_system_energy"]
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


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
    nnp_input = input["modelforge_methane_input"]

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #
    # reset filter parameters
    torch.manual_seed(1234)
    potential.core_network.representation_module.filter_net.reset_parameters()

    calculated_results = potential.compute_core_network_output(nnp_input)

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

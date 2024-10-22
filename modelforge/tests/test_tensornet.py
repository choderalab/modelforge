import pytest

from modelforge.tests.helper_functions import setup_potential_for_test


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_tensornet")
    return fn


def test_init(prep_temp_dir):
    """Test initialization of the TensorNet model."""

    # load default parameters
    # read default parameters
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environment="JAX",
        local_cache_dir=str(prep_temp_dir),
    )
    assert model is not None, "TensorNet model should be initialized."


@pytest.mark.parametrize("simulation_environment", ["PyTorch", "JAX"])
def test_forward_with_inference_model(
    simulation_environment, single_batch_with_batchsize, prep_temp_dir
):

    nnp_input = single_batch_with_batchsize(
        batch_size=32, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input

    # load default parameters
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environment=simulation_environment,
        use_training_mode_neighborlist=True,
        local_cache_dir=str(prep_temp_dir),
    )

    if simulation_environment == "JAX":
        from modelforge.jax import convert_NNPInput_to_jax

        model(convert_NNPInput_to_jax(nnp_input))
    else:
        model(nnp_input)


def test_input(prep_temp_dir):
    import torch
    from loguru import logger as log

    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_input,
    )

    # setup model
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        local_cache_dir=str(prep_temp_dir),
    )

    from importlib import resources

    from modelforge.tests import data

    # load reference data
    reference_data = resources.files(data) / "tensornet_input.pt"
    reference_batch = resources.files(data) / "nnp_input.pkl"
    import pickle

    mf_input = pickle.load(open(reference_batch, "rb"))

    # calculate pairlist
    pairlist_output = model.neighborlist.forward(mf_input)

    # compare to torchmd-net pairlist
    if reference_data:
        log.warning('Using reference data for "test_input"')
        edge_index, edge_weight, edge_vec = torch.load(reference_data)
    else:
        log.warning('Calculating reference data from  "test_input"')
        edge_index, edge_weight, edge_vec = prepare_values_for_test_tensornet_input(
            mf_input,
            seed=0,
        )

    # reshape and compare
    pair_indices = pairlist_output.pair_indices.t()
    edge_index = edge_index.t()
    for _, pair_index in enumerate(pair_indices):
        idx = ((edge_index == pair_index).sum(axis=1) == 2).nonzero()[0][
            0
        ]  # select [True, True]
        print(pairlist_output.d_ij[_][0], edge_weight[idx])
        assert torch.allclose(
            pairlist_output.d_ij[_][0],
            edge_weight[idx],
            rtol=1e-3,
        )
        print(pairlist_output.r_ij[_], -edge_vec[idx])
        assert torch.allclose(
            pairlist_output.r_ij[_],
            -1 * edge_vec[idx],
            rtol=1e-3,
        )


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function to the output of the
    # modelforge radial symmetry function TODO: only 'expnorm' from TensorNet
    # implemented
    import torch
    from openff.units import unit

    from modelforge.potential import (
        CosineAttenuationFunction,
        TensorNetRadialBasisFunction,
    )
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_compare_radial_symmetry_features,
    )

    seed = 0
    torch.manual_seed(seed)
    from importlib import resources

    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_radial_symmetry_features.pt"

    # generate a random list of distances, all < 5
    d_ij = unit.Quantity(
        torch.tensor([[2.4813], [3.8411], [0.4424], [0.6602], [1.5371]]), unit.angstrom
    )

    # TensorNet constants
    maximum_interaction_radius = unit.Quantity(5.1, unit.angstrom)
    minimum_interaction_radius = unit.Quantity(0.0, unit.angstrom)
    number_of_per_atom_features = 8
    alpha = (
        (maximum_interaction_radius - minimum_interaction_radius)
        / unit.Quantity(5.0, unit.angstrom)
    ).m / 10

    rsf = TensorNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_per_atom_features,
        max_distance=maximum_interaction_radius.to(unit.nanometer).m,
        min_distance=minimum_interaction_radius.to(unit.nanometer).m,
        alpha=alpha,
    )
    mf_r = rsf(d_ij.to(unit.nanometer).m)  # torch.Size([5, 8])
    cutoff_module = CosineAttenuationFunction(
        maximum_interaction_radius.to(unit.nanometer).m
    )

    rcut_ij = cutoff_module(d_ij.to(unit.nanometer).m)  # torch.Size([5, 1])
    mf_r = (mf_r * rcut_ij).unsqueeze(1)

    from importlib import resources

    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_radial_symmetry_features.pt"

    if reference_data:
        tn_r = torch.load(reference_data)
    else:
        tn_r = prepare_values_for_test_tensornet_compare_radial_symmetry_features(
            d_ij,
            minimum_interaction_radius,
            maximum_interaction_radius,
            number_of_per_atom_features,
            trainable=False,
            seed=seed,
        )

    assert torch.allclose(mf_r, tn_r, atol=1e-4)


def test_representation(prep_temp_dir):
    from importlib import resources

    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.potential.tensornet import TensorNetRepresentation
    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_representation.pt"

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1
    trainable_rbf = False
    highest_atomic_number = 128

    import pickle

    reference_batch = resources.files(data) / "nnp_input.pkl"
    nnp_input = pickle.load(open(reference_batch, "rb"))
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    model = setup_potential_for_test(
        use="inference",
        potential_seed=0,
        potential_name="tensornet",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        local_cache_dir=str(prep_temp_dir),
    )
    pairlist_output = model.neighborlist.forward(nnp_input)

    ################ modelforge TensorNet ################
    torch.manual_seed(0)
    tensornet_representation_module = TensorNetRepresentation(
        number_of_per_atom_features,
        num_rbf,
        act_class(),
        unit.Quantity(cutoff_upper, unit.angstrom).to(unit.nanometer).m,
        unit.Quantity(cutoff_lower, unit.angstrom).to(unit.nanometer).m,
        trainable_rbf,
        highest_atomic_number,
    )
    mf_X, _ = tensornet_representation_module(nnp_input, pairlist_output)
    ################ modelforge TensorNet ################

    ################ torchmd-net TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_representation(
            nnp_input,
            number_of_per_atom_features,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            trainable_rbf,
            highest_atomic_number,
            seed=0,
        )
    ################ torchmd-net TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


def test_interaction(prep_temp_dir):
    import pickle
    from importlib import resources

    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.potential.tensornet import TensorNetInteraction
    from modelforge.tests import data
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_interaction,
    )

    seed = 0

    reference_data = resources.files(data) / "tensornet_interaction.pt"

    reference_batch = resources.files(data) / "nnp_input.pkl"
    nnp_input = pickle.load(open(reference_batch, "rb"))

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1

    # -------------------------------#
    # -------------------------------#
    # initialize model
    model = setup_potential_for_test(
        use="inference",
        potential_seed=seed,
        potential_name="tensornet",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        local_cache_dir=str(prep_temp_dir),
    )

    ################ modelforge TensorNet ################
    tensornet_representation_module = model.core_network.representation_module
    pairlist_output = model.neighborlist.forward(nnp_input)
    X, _ = tensornet_representation_module(nnp_input, pairlist_output)

    radial_feature_vector = tensornet_representation_module.radial_symmetry_function(
        pairlist_output.d_ij
    )
    rcut_ij = tensornet_representation_module.cutoff_module(pairlist_output.d_ij)
    radial_feature_vector = (radial_feature_vector * rcut_ij).unsqueeze(1)

    atomic_charges = torch.zeros_like(nnp_input.atomic_numbers)

    # interaction
    torch.manual_seed(seed)
    interaction_module = TensorNetInteraction(
        number_of_per_atom_features,
        num_rbf,
        act_class(),
        unit.Quantity(cutoff_upper, unit.angstrom).to(unit.nanometer).m,
        "O(3)",
    )
    mf_X = interaction_module(
        X,
        pairlist_output.pair_indices,
        pairlist_output.d_ij.squeeze(-1),
        radial_feature_vector.squeeze(1),
        atomic_charges,
    )
    ################ modelforge TensorNet ################

    ################ TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_interaction(
            X,
            nnp_input,
            radial_feature_vector,
            atomic_charges,
            number_of_per_atom_features,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            seed,
        )
    ################ TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)

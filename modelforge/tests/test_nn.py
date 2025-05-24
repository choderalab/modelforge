from .test_potentials import load_configs_into_pydantic_models
import pytest

import torch


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_nn_temp")
    return fn


def test_embedding(single_batch_with_batchsize, prep_temp_dir, dataset_temp_dir):
    # test the input featurization, including:
    # - nuclear charge embedding
    # - total charge mixing

    import torch  # noqa: F401

    local_cache_dir = str(prep_temp_dir) + "/test_embedding"
    dataset_cache_dir = str(dataset_temp_dir)

    batch = single_batch_with_batchsize(
        batch_size=64,
        dataset_name="QM9",
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
    )

    nnp_input = batch.nnp_input
    model_name = "SchNet"
    # read default parameters and extract featurization
    config = load_configs_into_pydantic_models(f"{model_name.lower()}", "qm9")
    featurization_config = config["potential"].core_parameter.featurization.model_dump()

    # featurize the atomic input (default is only atomic number embedding)
    from modelforge.potential import FeaturizeInput

    featurize_input_module = FeaturizeInput(featurization_config)

    # mixing module should be the identity operation since only atomic number
    # embedding is used
    mixing_module = featurize_input_module.mixing
    assert mixing_module.__module__ == "torch.nn.modules.linear"
    mixing_module_name = str(mixing_module)

    # only atomic number embedded
    assert "atomic_number" in featurize_input_module.registered_embedding_operations
    assert len(featurize_input_module.registered_embedding_operations) == 1
    # no mixing
    assert "Identity()" in mixing_module_name

    # add total charge to the input
    featurization_config["properties_to_featurize"].append("per_system_total_charge")
    featurize_input_module = FeaturizeInput(featurization_config)

    # only atomic number embedded
    assert "atomic_number" in featurize_input_module.registered_embedding_operations
    assert len(featurize_input_module.registered_embedding_operations) == 1
    # total charge is added to feature vector
    assert (
        "per_system_total_charge"
        in featurize_input_module.registered_appended_properties
    )
    assert len(featurize_input_module.registered_appended_properties) == 1

    mixing_module = featurize_input_module.mixing
    assert (
        mixing_module.__module__ == "modelforge.potential.utils"
    )  # this is were Dense lives
    mixing_module_name = str(mixing_module)

    assert "Dense" in mixing_module_name

    # make a forward pass, embedd nuclear charges and add total charge (is expanded from per-molecule to per-atom property). Mix the properties then.
    out = featurize_input_module(nnp_input)
    assert out.shape == torch.Size(
        [557, 32]
    )  # nr_of_atoms, nr_of_per_atom_features (the total charge is mixed in)


def test_add_per_molecule_value(
    prep_temp_dir, single_batch_with_batchsize, dataset_temp_dir
):
    from modelforge.potential.featurization import AddPerMoleculeValue

    temp_prop = AddPerMoleculeValue(key="per_system_total_charge")
    assert temp_prop.key == "per_system_total_charge"

    local_cache_dir = str(prep_temp_dir) + "/test_add_per_molecule_value"
    dataset_cache_dir = str(dataset_temp_dir)

    batch = single_batch_with_batchsize(
        batch_size=1,
        dataset_name="SPICE2",
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
    )

    number_of_per_atom_features = 10
    atomic_number_embedding = torch.nn.Embedding(100, number_of_per_atom_features)
    atomic_numbers = batch.nnp_input.atomic_numbers

    embedding_tensor = atomic_number_embedding(atomic_numbers)
    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert embedding_tensor.shape[1] == number_of_per_atom_features

    embedding_tensor = temp_prop(embedding_tensor, batch.nnp_input)

    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert embedding_tensor.shape[1] == number_of_per_atom_features + 1


def test_add_per_atom_value(
    prep_temp_dir, single_batch_with_batchsize, dataset_temp_dir
):
    from modelforge.potential.featurization import AddPerAtomValue

    temp_prop = AddPerAtomValue(key="per_atom_partial_charge")
    assert temp_prop.key == "per_atom_partial_charge"

    local_cache_dir = str(prep_temp_dir) + "/test_add_per_atom_value"
    dataset_cache_dir = str(dataset_temp_dir)

    batch = single_batch_with_batchsize(
        batch_size=1,
        dataset_name="SPICE2",
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
    )
    # we need to set the per_atom_partial_charge to a tensor of some values
    # because by default we don't load in the values from the spice dataset
    batch.nnp_input.per_atom_partial_charge = torch.zeros(
        batch.nnp_input.atomic_numbers.shape[0], 1
    )

    number_of_per_atom_features = 10
    atomic_number_embedding = torch.nn.Embedding(100, number_of_per_atom_features)
    atomic_numbers = batch.nnp_input.atomic_numbers

    embedding_tensor = atomic_number_embedding(atomic_numbers)
    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert embedding_tensor.shape[1] == number_of_per_atom_features

    embedding_tensor = temp_prop(embedding_tensor, batch.nnp_input)

    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert embedding_tensor.shape[1] == number_of_per_atom_features + 1


def test_group_and_period_embedding(
    prep_temp_dir, single_batch_with_batchsize, dataset_temp_dir
):
    local_cache_dir = str(prep_temp_dir) + "/test_group_and_period_embedding"
    dataset_cache_dir = str(dataset_temp_dir)

    batch = single_batch_with_batchsize(
        batch_size=1,
        dataset_name="tmqm",
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
    )

    number_of_per_atom_features = 10
    atomic_number_embedding = torch.nn.Embedding(100, number_of_per_atom_features)
    atomic_numbers = batch.nnp_input.atomic_numbers

    embedding_tensor = atomic_number_embedding(atomic_numbers)

    from modelforge.potential.featurization import GroupPeriodEmbedding

    maximum_period = 7
    number_of_per_period_features = 5
    temp_period_embedding_tensor = torch.nn.Embedding(
        maximum_period, number_of_per_period_features
    )
    temp_prop = GroupPeriodEmbedding(temp_period_embedding_tensor, key="atomic_periods")
    assert temp_prop.key == "atomic_periods"
    assert temp_prop.embedding_tensor == temp_period_embedding_tensor

    embedding_tensor = temp_prop(embedding_tensor, batch.nnp_input)

    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert (
        embedding_tensor.shape[1]
        == number_of_per_atom_features + number_of_per_period_features
    )

    maximum_group = 18
    number_of_per_group_features = 11
    temp_group_embedding_tensor = torch.nn.Embedding(
        maximum_group, number_of_per_group_features
    )
    temp_prop2 = GroupPeriodEmbedding(temp_group_embedding_tensor, key="atomic_groups")
    assert temp_prop2.key == "atomic_groups"

    embedding_tensor = temp_prop2(embedding_tensor, batch.nnp_input)
    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert (
        embedding_tensor.shape[1]
        == number_of_per_atom_features
        + number_of_per_period_features
        + number_of_per_group_features
    )


def test_featurization_input(single_batch_with_batchsize):
    # less exhaustive test, than test_embedding, but should be sufficient
    # to make sure that atomic period and group embedding can be setup
    from modelforge.potential.featurization import FeaturizeInput

    featurization_dict = {
        "properties_to_featurize": ["atomic_number", "atomic_period", "atomic_group"],
        "atomic_number": {
            "maximum_atomic_number": 101,
            "number_of_per_atom_features": 32,
        },
        "atomic_period": {
            "maximum_period": 8,
            "number_of_per_period_features": 5,
        },
        "atomic_group": {
            "maximum_group": 19,
            "number_of_per_group_features": 11,
        },
    }

    temp_features = FeaturizeInput(featurization_dict)

    assert "atomic_number" in temp_features.registered_embedding_operations
    assert "atomic_period" in temp_features.registered_embedding_operations
    assert "atomic_group" in temp_features.registered_embedding_operations

    assert len(temp_features.registered_embedding_operations) == 3
    assert len(temp_features.embeddings) == 2

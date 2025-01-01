import pytest
import torch

from modelforge.potential.featurization import AddPerMoleculeValue


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_featurization_temp")
    return fn


def test_AddPerMoleculeValue(prep_temp_dir, single_batch_with_batchsize):

    temp_prop = AddPerMoleculeValue(key="per_system_total_charge")
    assert temp_prop.key == "per_system_total_charge"

    batch = single_batch_with_batchsize(
        batch_size=1,
        dataset_name="SPICE2",
        local_cache_dir=str(prep_temp_dir),
        version_select="nc_1000_v0",
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


def test_group_and_period_embedding(prep_temp_dir, single_batch_with_batchsize):
    batch = single_batch_with_batchsize(
        batch_size=1,
        dataset_name="tmqm",
        local_cache_dir=str(prep_temp_dir),
        version_select="nc_1000_v0",
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

    embedding_tensor = temp_prop2(embedding_tensor, batch.nnp_input)
    assert embedding_tensor.shape[0] == atomic_numbers.shape[0]
    assert (
        embedding_tensor.shape[1]
        == number_of_per_atom_features
        + number_of_per_period_features
        + number_of_per_group_features
    )

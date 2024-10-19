from .test_potentials import load_configs_into_pydantic_models
import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_nn_temp")
    return fn


def test_embedding(single_batch_with_batchsize, prep_temp_dir):
    # test the input featurization, including:
    # - nuclear charge embedding
    # - total charge mixing

    import torch  # noqa: F401

    batch = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
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
    assert "total_charge" in featurize_input_module.registered_appended_properties
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

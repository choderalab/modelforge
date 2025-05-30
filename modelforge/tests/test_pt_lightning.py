import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_pt_temp")
    return fn


def test_datamodule(prep_temp_dir, dataset_temp_dir):
    # This is an example script that trains an implemented model on the QM9 dataset.
    from modelforge.dataset.dataset import DataModule

    # Set up dataset

    properties_of_interest = ["atomic_numbers", "positions", "internal_energy_at_0K"]
    properties_assignment = {
        "atomic_numbers": "atomic_numbers",
        "positions": "positions",
        "E": "internal_energy_at_0K",
    }
    dm = DataModule(
        name="QM9",
        batch_size=512,
        local_cache_dir=str(prep_temp_dir),
        dataset_cache_dir=str(dataset_temp_dir),
        properties_of_interest=properties_of_interest,
        properties_assignment=properties_assignment,
    )

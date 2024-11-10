import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_pt_temp")
    return fn


def test_datamodule(prep_temp_dir):
    # This is an example script that trains an implemented model on the QM9 dataset.
    from modelforge.dataset.dataset import DataModule

    # Set up dataset

    dm = DataModule(
        name="QM9",
        batch_size=512,
        local_cache_dir=str(prep_temp_dir),
    )

import pytest
import sys


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("init_setup_temp")
    return fn


from modelforge.dataset import _ImplementedDatasets


# the idea of this is just to run through all the test datasets and download them initially
# so they are all available for the rest of the tests
# if we can't fetch the datasets for whatever reason (e.g. server issues), this should fail immediately.
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_dataset_download_from_zenodo(
    dataset_name, datamodule_factory, prep_temp_dir, dataset_temp_dir
):
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    local_cache_dir = str(prep_temp_dir) + "/dataset_generation"
    dataset_cache_dir = str(dataset_temp_dir)

    dataset = datamodule_factory(
        dataset_name=dataset_name,
        local_cache_dir=local_cache_dir,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        batch_size=64,
        dataset_cache_dir=dataset_cache_dir,
    )
    # download any additional subsets that we may need
    if dataset_name.lower().startswith("spice"):

        versions = {
            "spice1": "nc_1000_HCNOFClS_v1.1",
            "spice2": "nc_1000_HCNOFClS_v1.1",
            "spice1_openff": "nc_1000_HCNOFClS_v2.1",
            "spice2_openff": "nc_1000_HCNOFClS_v1.1",
        }

        dataset = datamodule_factory(
            dataset_name=dataset_name,
            batch_size=64,
            splitting_strategy=FirstComeFirstServeSplittingStrategy(),
            local_cache_dir=local_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            version_select=versions[dataset_name.lower()],
        )

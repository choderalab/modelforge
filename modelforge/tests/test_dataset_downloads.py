import pytest
from modelforge.dataset.dataset import DatasetFactory
from modelforge.dataset import _ImplementedDatasets
from modelforge.dataset.qm9 import QM9Dataset


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data_download_tests")
    return fn


dataset_versions = {
    "QM9": ["nc_1000_v0", "full_dataset_v0"],
    "ANI1X": ["nc_1000_v0", "full_dataset_v0"],
    "ANI2X": ["nc_1000_v0", "full_dataset_v0"],
    "SPICE114": [
        "nc_1000_v0",
        "full_dataset_v0",
        "full_dataset_v0_HCNOFClS",
        "nc_1000_v0_HCNOFClS",
    ],
    "SPICE2": [
        "nc_1000_v0",
        "full_dataset_v0",
        "full_dataset_v0_HCNOFClS",
        "nc_1000_v0_HCNOFClS",
    ],
    "SPICE114_OPENFF": [
        "nc_1000_v0",
        "full_dataset_v0",
        "full_dataset_v0_HCNOFClS",
        "nc_1000_v0_HCNOFClS",
    ],
    "PHALKETHOH": ["nc_1000_v0", "full_dataset_v0"],
}

dataset_and_version = []
for name in _ImplementedDatasets.get_all_dataset_names():
    if name in dataset_versions.keys():
        for version in dataset_versions[name]:
            dataset_and_version.append((name, version))


@pytest.mark.parametrize("dataset_name, version", dataset_and_version)
@pytest.mark.data_download
def test_download_download(dataset_name, version, prep_temp_dir):
    local_cache_dir = str(prep_temp_dir)

    # if version in dataset_versions[dataset_name]:
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        version_select=version, local_cache_dir=local_cache_dir, force_download=True
    )
    e

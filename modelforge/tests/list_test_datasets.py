"""
This script will fetch the datasets used in testing to a central location.

The CI is configured to cache this directory, to minimize the number of downloads that are required.  This will
help to eliminate zenodo downtime as a limiting factor
"""

from modelforge.dataset import _ImplementedDatasets
import os
from importlib import resources
import toml

# define the location to save the cached datasets
dataset_cache_dir = "~/.cache/modelforge_testing_dataset_cache"

dataset_cache_dir = os.path.expanduser(dataset_cache_dir)
print(dataset_cache_dir)

# create the dataset cache directory if it does not exist
os.makedirs(dataset_cache_dir, exist_ok=True)


track_files = []
for dataset_name in _ImplementedDatasets.get_all_dataset_names():
    print("fetching " + dataset_name)

    toml_file = f'{resources.files("modelforge.tests")}/data/dataset_defaults/{dataset_name.lower()}.toml'

    print("using toml file: " + toml_file)

    # check to ensure the toml file exists
    if not os.path.exists(toml_file):
        raise FileNotFoundError(
            f"Dataset toml file {toml_file} not found. Please check the dataset name."
        )

    config_dict = toml.load(toml_file)

    version_select = config_dict["dataset"]["version_select"]

    track_files.append([dataset_name, version_select])
    # download any additional subsets that we may need
    if dataset_name.lower().startswith("spice"):

        versions = {
            "spice1": "nc_1000_HCNOFClS_v1.1",
            "spice2": "nc_1000_HCNOFClS_v1.1",
            "spice1_openff": "nc_1000_HCNOFClS_v2.1",
            "spice2_openff": "nc_1000_HCNOFClS_v1.1",
        }
        track_files.append([dataset_name, versions[dataset_name.lower()]])

# purge anything but the hdf5 files from the modelforge_testing_dataset_cache

# write out a text file listing the datasets and versions downloaded so that we can check that hash of it
# on the CI for cache validation
with open(f"{dataset_cache_dir}/downloaded_datasets.txt", "w") as f:
    for dataset_name, version in track_files:
        f.write(f"{dataset_name}: {version}\n")

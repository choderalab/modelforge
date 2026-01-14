"""
This script will fetch the datasets used in testing to a central location.

The CI is configured to cache this directory, to minimize the number of downloads that are required.  This will
help to eliminate zenodo downtime as a limiting factor
"""

from modelforge.dataset import _ImplementedDatasets
from modelforge.dataset.utils import (
    FirstComeFirstServeSplittingStrategy,
    SplittingStrategy,
)
from modelforge.dataset.dataset import initialize_datamodule
from typing import Optional, Type
import os


def create_datamodule(
    dataset_name: str,
    batch_size: int,
    local_cache_dir: str,
    splitting_strategy: Type[SplittingStrategy] = FirstComeFirstServeSplittingStrategy,
    dataset_cache_dir: Optional[str] = None,
    remove_self_energies: Optional[bool] = True,
    element_filter: Optional[str] = None,
    local_yaml_file: Optional[str] = None,
    shift_center_of_mass_to_origin: bool = True,
    version_select: Optional[str] = None,
    regression_ase: Optional[bool] = False,
    shift_energies: Optional[str] = None,
):
    """
    Calls the initialize_datamodule function, returning a DataModule instance.

    This function will load the .toml file in the dataset_defaults directory under tests, using the default
    version in there, unless otherwise specified.

    This function is effectively the same as datamodule_factory in conftest.py.

    Parameters
    ----------
    dataset_name: str
        This is the name of the dataset to load.
    batch_size: int
        This is the batch size to use.
    local_cache_dir: str
        This is the local cache directory to use for data associated with the run
    splitting_strategy: Type[SplittingStrategy], default = FirstComeFirstServeSplittingStrategy
        The splitting strategy to use, default, FirstComeFirstServeSplittingStrategy.
    dataset_cache_dir: str, Optional[str]
        This is the local cache directory to use for datasets
    remove_self_energies: bool, Optional[str]
        Whether to remove the self-energy from the dataset after loading
    element_filter: Optional[str], Optional[str]
        Elements to include/exclude.
    local_yaml_file: Optional[str]
        Path to the local yaml file to use, if using a dataset not part of modelforge
    shift_center_of_mass_to_origin: bool, default=True
        Whether to shift systems such that their center of mass will be at the origin
    version_select: Optional[str]
        Which version of the dataset to use; if not specified the version defined in the associated dataset .toml file
        will be used.
    regression_ase: Optional[bool], default=False
        If true, the atomic self energies will be calculated via regression for use.
    shift_energies: Optional[str]
        If defined, the dataset will be shifted; options include min, mean, max.   If not defined, no shifting is applied

    Returns
    -------
        DataModule
    """

    from importlib import resources
    import toml
    import os

    toml_file = f"{resources.files('modelforge.tests')}/data/dataset_defaults/{dataset_name.lower()}.toml"

    # check to ensure the yaml file exists
    if not os.path.exists(toml_file):
        raise FileNotFoundError(
            f"Dataset toml file {toml_file} not found. Please check the dataset name."
        )

    config_dict = toml.load(toml_file)

    if dataset_cache_dir is None:
        dataset_cache_dir = local_cache_dir

    if version_select is None:
        version_select = config_dict["dataset"]["version_select"]

    return initialize_datamodule(
        dataset_name=dataset_name,
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        version_select=version_select,
        properties_of_interest=config_dict["dataset"]["properties_of_interest"],
        properties_assignment=config_dict["dataset"]["properties_assignment"],
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        remove_self_energies=remove_self_energies,
        element_filter=element_filter,
        local_yaml_file=local_yaml_file,
        shift_center_of_mass_to_origin=shift_center_of_mass_to_origin,
        regression_ase=regression_ase,
        shift_energies=shift_energies,
    )


# define the location to save the cached datasets
dataset_cache_dir = "~/modelforge_testing_dataset_cache"
# expand the dataset_cache_dir path to get the full path
dataset_cache_dir = os.path.expanduser(dataset_cache_dir)

# define a local cache dir that we will simple remove when we are done
local_cache_dir = f"{dataset_cache_dir}/local_cache"


for dataset_name in _ImplementedDatasets.get_all_dataset_names():
    print("fetching " + dataset_name)

    dataset = create_datamodule(
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

        dataset = create_datamodule(
            dataset_name=dataset_name,
            batch_size=64,
            splitting_strategy=FirstComeFirstServeSplittingStrategy(),
            local_cache_dir=local_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            version_select=versions[dataset_name.lower()],
        )

# purge anything but the hdf5 files from the dataset_cache

# first remove the local_cache_dir
import shutil

shutil.rmtree(local_cache_dir, ignore_errors=True)

# get a list of files in the dataset directory and remove any that don't end in hdf5
for file in os.listdir(dataset_cache_dir):
    if not file.endswith(".hdf5"):
        os.remove(os.path.join(dataset_cache_dir, file))

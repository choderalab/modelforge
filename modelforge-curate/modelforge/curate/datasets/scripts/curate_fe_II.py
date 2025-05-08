"""
This script will generate hdf5 datafiles for the Fe(II) dataset using the FeIICuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.

he Fe(II) dataset includes 28834 total configurations for 383 unique molecules Fe(II) organometallic complexes.
    Specifically, this includes 15568 HS geometries and 13266 LS geometries.
    These complexes originate from the Cambridge Structural Database (CSD) as curated by
    Nandy, et al. (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214),
    and were filtered into “computation-ready” complexes, (those where both oxidation states and charges are already
    specified without hydrogen atoms missing in the structures), following the procedure outlined by Arunachalam, et al.
    (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700)


    The original Fe (II) dataset is available from github:
    https://github.com/Neon8988/Iron_NNPs

    The code uses a fork of the original dataset, to enable clear versioning (i.e. a release):
    https://github.com/chrisiacovella/Iron_NNPs

    Citation to the original  dataset:

    Modeling Fe(II) Complexes Using Neural Networks
    Hongni Jin and Kenneth M. Merz Jr.
    Journal of Chemical Theory and Computation 2024 20 (6), 2551-2558
    DOI: 10.1021/acs.jctc.4c00063
"""

from modelforge.curate.datasets.fe_II_curation import FeIICuration
from openff.units import unit
from modelforge.curate.utils import VersionMetadata
import time
import yaml


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/fe_II_dataset"
    local_cache_dir = f"{local_prefix}/fe_II_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.1"
    # version of the dataset to curate
    version_select = "v_0"

    fe_II = FeIICuration(
        dataset_name="fe_II",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    # fe_II.load_from_db(local_cache_dir, "fe_II.sqlite")
    fe_II.process(force_download=False)

    # Generate a yaml file with the information about each of the dataset versions we curated in this script.
    # The contents can be cut and paste into the appropriate file in the modelforge/dataset/yaml_files
    yaml_file = f"{output_file_dir}/fe_II_dataset_curation_v{version}.yaml"
    with open(yaml_file, "w") as f:
        f.write(
            f"# This file contains the metadata for the curated Fe II dataset v{version}.\n"
        )
        f.write(
            f"# Processed on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.\n"
        )

    ####################
    # full dataset
    ####################
    version_name = f"full_dataset_v{version}"
    hdf5_file_name = f"fe_II_v{version}.hdf5"

    total_records, total_configs = fe_II.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
    )
    about = f"""This provides a curated hdf5 file for the Fe (II) dataset designed
    to be compatible with modelforge. This dataset contains {n_total_records} unique records 
    for {n_total_configs} total configurations. """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "energies",
            "positions",
            "forces",
            "total_charge",
            "spin_multiplicities",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.compress_hdf5()

    # dump the metadata to the yaml file
    with open(yaml_file, "a") as f:
        yaml.dump(metadata.remote_dataset_to_dict(), f)

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    ####################
    # 1000 configuration test set
    ####################

    version_name = f"nc_1000_v{version}"
    hdf5_file_name = f"fe_II_ntc_1000_v{version}.hdf5"

    total_records, total_configs = fe_II.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
    )
    about = f"""This provides a curated hdf5 file for a subset of the Fe (II) dataset designed
    to be compatible with modelforge. This dataset contains {n_total_records} unique records 
    for {n_total_configs} total configurations witha  maximum of 10 configurations per record."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "energies",
            "positions",
            "forces",
            "total_charge",
            "spin_multiplicities",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.compress_hdf5()

    # dump the metadata to the yaml file
    with open(yaml_file, "a") as f:
        yaml.dump(metadata.remote_dataset_to_dict(), f)
    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

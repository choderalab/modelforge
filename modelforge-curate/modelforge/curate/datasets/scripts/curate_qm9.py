"""
This script will generate hdf5 datafiles for the QM9 dataset using the QM9Curation class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.

The QM9 dataset includes 133,885 organic molecules with up to nine total heavy atoms (C,O,N,or F; excluding H).
All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.

Citation: Ramakrishnan, R., Dral, P., Rupp, M. et al.
            "Quantum chemistry structures and properties of 134 kilo molecules."
            Sci Data 1, 140022 (2014).
            https://doi.org/10.1038/sdata.2014.22

DOI for dataset: 10.6084/m9.figshare.c.978904.v5
"""

from modelforge.curate.datasets.qm9_curation import QM9Curation
from modelforge.curate.utils import VersionMetadata
import time
import yaml
import os


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/qm9_dataset"
    local_cache_dir = f"{local_prefix}/qm9_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.1"
    # version of the dataset to curate
    version_select = f"v_0"
    # Curate the test dataset with 1000 total conformers
    from modelforge.curate.datasets.qm9_curation import QM9Curation

    qm9_dataset = QM9Curation(
        dataset_name="qm9",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )
    qm9_dataset.process(force_download=False)

    yaml_file = f"{output_file_dir}/qm9_dataset_curation_v{version}.yaml"
    with open(yaml_file, "w") as f:
        f.write(
            f"# This file contains the metadata for the curated qm9 dataset v{version}.\n"
        )
        f.write(
            f"# Processed on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.\n"
        )

    ###############
    # full dataset
    ##############
    # Curates the full dataset
    hdf5_file_name = f"qm9_dataset_v{version}.hdf5"

    n_total_records, n_total_configs = qm9_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name, output_file_dir=output_file_dir
    )

    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the qm9  dataset designed
        to be compatible with modelforge. This dataset contains {n_total_records} unique records 
        for {n_total_configs} total configurations.  Note, the dataset contains only a single 
        configuration per record."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "partial_charges",
            "polarizability",
            "dipole_moment_per_system",
            "dipole_moment_scalar_per_system" "energy_of_homo",
            "lumo-homo_gap",
            "zero_point_vibrational_energy",
            "internal_energy_at_298.15K",
            "internal_energy_at_0K",
            "enthalpy_at_298.15K",
            "free_energy_at_298.15K",
            "heat_capacity_at_298.15K",
            "rotational_constants",
            "harmonic_vibrational_frequencies",
            "electronic_spatial_extent",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.compress_hdf5()

    # dump the metadata to the yaml file
    with open(yaml_file, "a") as f:
        yaml.dump(metadata.remote_dataset_to_dict(), f)

    print("full dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    # Curates the test dataset with 1000 total conformers
    # only a single config per record

    hdf5_file_name = f"qm9_dataset_v{version}_ntc_1000.hdf5"
    n_total_records, n_total_configs = qm9_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
    )
    version_name = f"nc_1000_{version}"
    about = f"""This provides a curated hdf5 file for a subset of the qm9 dataset designed
        to be compatible with modelforge. This dataset contains {n_total_records} unique records 
        for {n_total_configs} total configurations. Note, the dataset contains only a single 
        configuration per record."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "partial_charges",
            "polarizability",
            "dipole_moment_per_system",
            "dipole_moment_scalar_per_system" "energy_of_homo",
            "lumo-homo_gap",
            "zero_point_vibrational_energy",
            "internal_energy_at_298.15K",
            "internal_energy_at_0K",
            "enthalpy_at_298.15K",
            "free_energy_at_298.15K",
            "heat_capacity_at_298.15K",
            "rotational_constants",
            "harmonic_vibrational_frequencies",
            "electronic_spatial_extent",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.compress_hdf5()

    # dump the metadata to the yaml file
    with open(yaml_file, "a") as f:
        yaml.dump(metadata.remote_dataset_to_dict(), f)
    print(" 1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

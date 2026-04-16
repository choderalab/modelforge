"""
This script will generate hdf5 datafiles for the aimnet2 dataset using the Aimnet2Curation class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.

The aimnet2 dataset contain molecular structures and the properties computed with B97-3c (GGA DFT) or wB97M-def2-TZVPP
(range-separated hybrid DFT) methods. Each data file contains about 20M structures.
DFT calculation performed with ORCA 5.0.3 software.

Properties include energy, forces, atomic charges, and molecular dipole and quadrupole moments.


Dataset Citation: Zubatiuk, Roman; Isayev, Olexandr; Anstine, Dylan (2024).
        Training datasets for AIMNet2 machine-learned neural network potential.
        Carnegie Mellon University.
        https://doi.org/10.1184/R1/27629937.v2

DOI for associated publication:
    publisher: https://doi.org/10.1039/D4SC08572H
    ChemRxiv: https://doi.org/10.26434/chemrxiv-2023-296ch-v3
"""


def main():
    # define the location where to store and output the files
    import os
    from modelforge.curate.datasets.aimnet2_curation import Aimnet2Curation
    from modelforge.curate.utils import VersionMetadata

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/aimnet2_dataset"
    local_cache_dir = f"{local_prefix}/aimnet2_dataset"

    # We'll want to provide some simple means of versioning
    # this will be used in the output hdf5 files
    version = "1.0"

    # version of the dataset to curate
    version_select = "wb97m_v0"

    aimnet2_dataset = Aimnet2Curation(
        dataset_name="aimnet2",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    aimnet2_dataset.process(force_download=False)

    ####################
    # 1000 configuration test set
    ####################

    # Curate aimnet2 test dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"aimnet2_dataset_v{version}_ntc_1000.hdf5"

    n_total_records, n_total_configs = aimnet2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
    )

    version_name = f"nc_1000_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the aimnet2 dataset designed 
    to be compatible with modelforge. This dataset contains {n_total_records} unique records 
    for {n_total_configs} total configurations, with a maximum of 10 configurations per record. 
    Note, individual configurations are partitioned into entries based on the array of atomic 
    species appearing in sequence in the source data file."""

    # create the metadata for the dataset that will be written to the yaml file

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
            "partial_charges",
            "total_charge",
            "dipole_moment_per_system",
            "quadrupole_moment_per_system",
        ],
    )

    # this will also compress the hdf5 file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # curate the full aimnet2 dataset
    ###############
    hdf5_file_name = f"aimnet2_dataset_v{version}.hdf5"

    n_total_records, n_total_configs = aimnet2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
    )

    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the aimnet2 dataset designed
    to be compatible with modelforge. This dataset contains {n_total_records} unique records 
    for {n_total_configs} total configurations. Note, individual configurations are partitioned 
    into entries based on the array of atomic species appearing in sequence in the source data file."""

    # create the metadata for the dataset that will be written to the yaml file

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
            "partial_charges",
            "total_charge",
            "dipole_moment_per_system",
            "quadrupole_moment_per_system",
        ],
    )
    # this will also compress the hdf5 file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("full dataset dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

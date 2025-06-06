"""
This script will generate hdf5 datafiles for the ANI2X dataset using the ANI2xCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.

The ANI-2x data set includes properties for small organic molecules that contain
H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for nearly 200,000 molecules.
This will fetch data generated with the wB97X/631Gd level of theory
used in the original ANI-2x paper, calculated using Gaussian 09

Citation: Devereux, C, Zubatyuk, R., Smith, J. et al.
            "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens."
            Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202.
            https://doi.org/10.1021/acs.jctc.0c00121

DOI for dataset: 10.5281/zenodo.10108941
"""


def main():
    # define the location where to store and output the files
    import os
    from modelforge.curate.datasets.ani2x_curation import ANI2xCuration
    from modelforge.curate.utils import VersionMetadata

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/ani2x_dataset"
    local_cache_dir = f"{local_prefix}/ani2x_dataset"

    # We'll want to provide some simple means of versioning
    # this will be used in the output hdf5 files
    version = "1.1"

    # version of the dataset to curate
    version_select = f"v_0"

    ani2x_dataset = ANI2xCuration(
        dataset_name="ani2x",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    ani2x_dataset.process(force_download=False)

    ####################
    # 1000 configuration test set
    ####################

    # Curate ANI2x test dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"ani2x_dataset_v{version}_ntc_1000.hdf5"

    n_total_records, n_total_configs = ani2x_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
    )

    version_name = f"nc_1000_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the ANI-2x dataset designed 
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
        available_properties=["atomic_numbers", "energies", "positions", "forces"],
    )

    # this will also compress the hdf5 file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # curate the full ANI-2x dataset
    ###############
    hdf5_file_name = f"ani2x_dataset_v{version}.hdf5"

    n_total_records, n_total_configs = ani2x_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
    )

    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the ANI-2x dataset designed
    to be compatible with modelforge. This dataset contains {n_total_records} unique records 
    for {n_total_configs} total configurations. Note, individual configurations are partitioned 
    into entries based on the array of atomic species appearing in sequence in the source data file."""

    # create the metadata for the dataset that will be written to the yaml file

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=["atomic_numbers", "energies", "positions", "forces"],
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

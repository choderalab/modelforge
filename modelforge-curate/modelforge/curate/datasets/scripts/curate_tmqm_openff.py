"""
This script will generate hdf5 datafiles for the tmqm openff dataset.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 configurations, with a maximum of 10 configurations per molecule.


"""

from modelforge.curate.datasets.tmqm_openff_curation import tmQMOpenFFCuration
from modelforge.curate.utils import VersionMetadata


def main():

    from openff.units import unit

    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/tmqm_openff_dataset"
    local_cache_dir = f"{local_prefix}/tmqm_openff_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.3"
    # version of the dataset to curate
    version_select = f"v_0"

    tmqm_openff = tmQMOpenFFCuration(
        dataset_name="tmqm_openff",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    tmqm_openff.process(
        qcportal_view_filename="dataset_419_view.sqlite",
        qcportal_view_path="/home/cri/mf_datasets/tmqm_openff_dataset/dataset_download_Feb26",
        force_download=False,
    )
    # tmqm_openff.load_from_db(
    #     local_db_dir="/home/cri/mf_datasets/tmqm_openff_dataset",
    #     local_db_name="tmqm_openff.sqlite",
    # )
    available_properties = (
        [
            "atomic_numbers",
            "positions",
            "total_charge",
            "per_system_spin_multiplicity",
            "dft_total_energy",
            "dft_total_force",
            "scf_dipole",
            "scf_quadrupole",
            "mulliken_partial_charges",
            "lowdin_partial_charges",
            "spin_multiplicity_per_atom",
        ],
    )
    #################
    # 1000 configuration version
    #################

    # curate dataset with 1000 total configurations, max of 10 configurations per record
    hdf5_file_name = f"tmqm_openff_dataset_v{version}_ntc_1000.hdf5"

    n_total_records, n_total_configs = tmqm_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"nc_1000_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with a maximum of 10 configurations per record.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """
    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # full dataset
    ##############
    # curate the full dataset
    hdf5_file_name = f"tmqm_openff_dataset_v{version}.hdf5"
    print("total dataset")

    n_total_records, n_total_configs = tmqm_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )

    version_name = f"full_dataset_v{version}"

    about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # full dataset, spin multiplicity restrictions
    ##############
    for sm in [1, 3, 5]:
        # curate the full dataset
        hdf5_file_name = f"tmqm_openff_dataset_sm{sm}_v{version}.hdf5"
        print("total dataset")

        n_total_records, n_total_configs = tmqm_openff.to_hdf5(
            hdf5_file_name=hdf5_file_name,
            output_file_dir=output_file_dir,
            max_force=1.0 * unit.hartree / unit.bohr,
            max_force_key="dft_total_force",
            spin_multiplicity_to_limit=sm,
            spin_multiplicity_key="per_system_spin_multiplicity",
        )

        version_name = f"full_dataset_sm{sm}_v{version}"

        about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed 
                    to be compatible with modelforge. This dataset contains {n_total_records} unique records
                    for {n_total_configs} total configurations, restricted to spin multiplicity {sm}.
                    This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
                    """

        metadata = VersionMetadata(
            version_name=version_name,
            about=about,
            hdf5_file_name=hdf5_file_name,
            hdf5_file_dir=output_file_dir,
            available_properties=available_properties,
        )
        # we need to compress the hdf5 file to get the checksum and length for the gzipped file
        metadata.to_yaml(
            file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
        )

        print(f"Total records: {n_total_records}")
        print(f"Total configs: {n_total_configs}")

    ######################################
    # 1000 conformer, last configuration only
    ######################################
    # curate dataset with 1000 total configurations, last only
    hdf5_file_name = f"tmqm_openff_dataset_v{version}_ntc_1000_minimal.hdf5"

    n_total_records, n_total_configs = tmqm_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )
    version_name = f"nc_1000_v{version}_minimal"
    about = f"""This provides a curated hdf5 file for a subset of the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with only the final configuration of the optimization.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration subset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    #######################################3
    # full dataset last configuration only
    #######################################3

    # curate the full dataset last config only

    hdf5_file_name = f"tmqm_openff_dataset_v{version}_minimal.hdf5"

    n_total_records, n_total_configs = tmqm_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )
    version_name = f"full_dataset_v{version}_minimal"
    about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with only the final configuration of the optimization.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("full dataset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

"""
This script will generate hdf5 datafiles for the PhAlkEthOH dataset at the OpenFF level of theory
using the SPICEOpenFFCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 configurations, with a maximum of 10 configurations per molecule.


"""


def main():

    from openff.units import unit

    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/PhAlkEthOH_openff_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.1"
    # version of the dataset to curate
    version_select = f"v_0"

    from modelforge.curate.datasets.phalkethoh_curation import PhAlkEthOHCuration

    PhAlkEthOH_openff = PhAlkEthOHCuration(
        dataset_name="PhAlkEthOH_openff",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    PhAlkEthOH_openff.process(force_download=False)
    # PhAlkEthOH_openff.load_from_db(
    #     local_db_dir="/home/cri/mf_datasets/PhAlkEthOH_openff_dataset",
    #     local_db_name="PhAlkEthOH_openff.sqlite",
    # )

    # record_names = list(PhAlkEthOH_openff.dataset.records.keys())

    # record = PhAlkEthOH_openff.dataset.get_record(record_names[0])

    # print(record)
    # curate dataset with 1000 total configurations, max of 10 configurations per record
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_ntc_1000.hdf5"

    n_total_records, n_total_configs = PhAlkEthOH_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    print("1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    # curate the full dataset
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}.hdf5"
    print("total dataset")

    n_total_records, n_total_configs = PhAlkEthOH_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    # curate dataset with 1000 total configurations, last only
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_ntc_1000_minimal.hdf5"

    n_total_records, n_total_configs = PhAlkEthOH_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )
    print("1000 configuration subset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    # curate the full dataset last config only

    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_minimal.hdf5"

    n_total_records, n_total_configs = PhAlkEthOH_openff.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )

    print("full dataset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

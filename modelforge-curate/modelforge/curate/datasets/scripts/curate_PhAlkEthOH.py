"""
This script will generate hdf5 datafiles for the PhAlkEthOH dataset at the OpenFF level of theory
using the SPICEOpenFFCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.


"""


def PhAlkEthOH_openff_wrapper(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    version_select: str = "latest",
    max_records=None,
    max_conformers_per_record=None,
    total_conformers=None,
    max_force=None,
    final_conformer_only=False,
):
    """
    This curates and processes the SPICE 114 dataset at the OpenFF level of theory into an hdf5 file.


    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated.
    output_file_dir: str, required
        Directory where the hdf5 file will be saved.
    local_cache_dir: str, required
        Directory where the intermediate data will be saved; in this case it will be tarred file downloaded
        from figshare and the expanded archive that contains xyz files for each molecule in the dataset.
    force_download: bool, optional, default=False
        If False, we will use the tarred file that exists in the local_cache_dir (if it exists);
        If True, the tarred file will be downloaded, even if it exists locally.
    version_select: str, optional, default="latest"
        The version of the dataset to use as defined in the associated yaml file.
        If "latest", the most recent version will be used.
    max_records: int, optional, default=None
        The maximum number of records to process.
    max_conformers_per_record: int, optional, default=None
        The maximum number of conformers to process for each record.
    total_conformers: int, optional, default=None
        The total number of conformers to process.
    max_force: float, optional, default=None
        The maximum force to allow in the dataset. Any conformers with forces greater than this value will be ignored.
    final_conformer_only: bool, optional, default=False
        If True, only the final conformer for each molecule will be processed. If False, all conformers will be processed.

    """
    from modelforge.curate.datasets.phalkethoh_curation import PhAlkEthOHCuration

    PhAlkEthOH_dataset = PhAlkEthOHCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )
    PhAlkEthOH_dataset.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
        n_threads=1,
        max_force=max_force,
        final_conformer_only=final_conformer_only,
    )
    print(f"Total records: {PhAlkEthOH_dataset.total_records()}")
    print(f"Total conformers: {PhAlkEthOH_dataset.total_configs()}")


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

    # curate dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_ntc_1000.hdf5"

    PhAlkEthOH_openff_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_records=1000,
        total_conformers=1000,
        max_conformers_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
    )

    # curate the full dataset
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}.hdf5"
    print("total dataset")
    PhAlkEthOH_openff_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_force=1.0 * unit.hartree / unit.bohr,
    )

    # curate dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_ntc_1000_minimal.hdf5"

    PhAlkEthOH_openff_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_records=1000,
        total_conformers=1000,
        max_conformers_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
        final_conformer_only=True,
    )

    # curate the full dataset
    hdf5_file_name = f"PhAlkEthOH_openff_dataset_v{version}_minimal.hdf5"
    print("total dataset")
    PhAlkEthOH_openff_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_force=1.0 * unit.hartree / unit.bohr,
        final_conformer_only=True,
    )


if __name__ == "__main__":
    main()

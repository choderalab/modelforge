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
    limit_atomic_species=None,
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
    limit_atomic_species: list, optional, default=None
        A list of atomic species to limit the dataset to. Any molecules that contain elements outside of this list
        will be ignored. If not defined, no filtering by atomic species will be performed.


    """
    from modelforge.curation.phalkethoh_curation import PhAlkEthOHCuration

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
        limit_atomic_species=limit_atomic_species,
        n_threads=1,
    )
    print(f"Total records: {PhAlkEthOH_dataset.total_records}")
    print(f"Total conformers: {PhAlkEthOH_dataset.total_conformers}")


def main():

    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/PhAlkEthOH_openff_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "0"
    # version of the dataset to curate
    version_select = f"v_{version}"

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
    )


if __name__ == "__main__":
    main()

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


def qm9_wrapper(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    version_select: str = "latest",
    max_records=None,
    max_conformers_per_record=None,
    total_conformers=None,
):
    """
    This instantiates and calls the QM9Curation class to generate the hdf5 file for the QM9 dataset.

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


    """
    from modelforge.curate.datasets.qm9_curation import QM9Curation

    qm9 = QM9Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    qm9.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
    )
    print(f"Total records: {qm9.total_records()}")
    print(f"Total configurations: {qm9.total_configs()}")


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/qm9_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "0"
    # version of the dataset to curate
    version_select = f"v_{version}"
    # Curate the test dataset with 1000 total conformers
    hdf5_file_name = f"qm9_dataset_v{version}_ntc_1.hdf5"

    qm9_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_conformers_per_record=1,  # there is only one conformer per molecule in the QM9 dataset
        total_conformers=1,
    )
    #
    # # Curates the full dataset
    # hdf5_file_name = f"qm9_dataset_v{version}.hdf5"
    #
    # qm9_wrapper(
    #     hdf5_file_name,
    #     output_file_dir,
    #     local_cache_dir,
    #     force_download=False,
    #     version_select=version_select,
    # )


if __name__ == "__main__":
    main()

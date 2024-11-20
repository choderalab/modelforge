"""
This script will generate hdf5 datafiles for the tmQM dataset using the tmQMCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.

    The tmQM dataset contains the geometries and properties of 86,665 mononuclear complexes extracted from the
    Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large
    variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    Citation:

    David Balcells and Bastian Bjerkem Skjelstad,
    tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes
    Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146
    DOI: 10.1021/acs.jcim.0c01041

    Original dataset source: https://github.com/uiocompcat/tmQM
"""


def tmqm_wrapper(
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
    This instantiates and calls the tmQMCuration class to generate the hdf5 file for the tmQM dataset.

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
    from modelforge.curation.tmqm_curation import tmQMCuration

    tmqm = tmQMCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    tmqm.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
    )
    print(f"Total records: {tmqm.total_records}")
    print(f"Total conformers: {tmqm.total_conformers}")


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/tmqm_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "0"
    version_out = "0"
    # version of the dataset to curate
    version_select = f"v_{version}"
    # Curate the test dataset with 1000 total conformers
    hdf5_file_name = f"tmqm_dataset_v{version_out}_ntc_1000.hdf5"

    tmqm_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_conformers_per_record=1,  # there is only one conformer per molecule in the tmqm dataset
        total_conformers=1000,
    )

    # Curates the full dataset
    hdf5_file_name = f"tmqm_dataset_v{version_out}.hdf5"

    tmqm_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
    )


if __name__ == "__main__":
    main()

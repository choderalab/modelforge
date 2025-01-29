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


def ani2x_wrapper(
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
    This fetches and processes the ANI2x dataset into a curated hdf5 file.

    The ANI-2x data set includes properties for small organic molecules that contain
        H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for 200,000
        This will fetch data generated with the wB97X/631Gd level of theory
        used in the original ANI-2x paper, calculated using Gaussian 09

        Citation: Devereux, C, Zubatyuk, R., Smith, J. et al.
                    "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens."
                    Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202.
                    https://doi.org/10.1021/acs.jctc.0c00121

        DOI for dataset: 10.5281/zenodo.10108941

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
    from modelforge.curate.datasets.ani2x_curation import ANI2xCuration

    ani2x = ANI2xCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    ani2x.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
    )
    print(f"Total records: {ani2x.total_records()}")
    print(f"Total configs: {ani2x.total_configs()}")


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/ani2x_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.0"

    # version of the dataset to curate
    version_select = f"v_0"

    # curate ANI2x test dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"ani2x_dataset_v{version}_ntc_1000.hdf5"

    ani2x_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        max_conformers_per_record=10,
        version_select=version_select,
        total_conformers=1000,
    )

    # curate the full ANI-2x dataset
    hdf5_file_name = f"ani2x_dataset_v{version}.hdf5"

    ani2x_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
    )


if __name__ == "__main__":
    main()

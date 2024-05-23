"""
This script will generate hdf5 datafiles for the ANI1X dataset using the ANI1xCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.

This dataset includes ~5 million density function theory calculations
for small organic molecules containing H, C, N, and O.
A subset of ~500k are computed with accurate coupled cluster methods.

References:

ANI-1x dataset:
Smith, J. S.; Nebgen, B.; Lubbers, N.; Isayev, O.; Roitberg, A. E.
Less Is More: Sampling Chemical Space with Active Learning.
J. Chem. Phys. 2018, 148 (24), 241733.
https://doi.org/10.1063/1.5023802
https://arxiv.org/abs/1801.09319

ANI-1ccx dataset:
Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E.
Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential through Transfer Learning. N
at. Commun. 2019, 10 (1), 2903.
https://doi.org/10.1038/s41467-019-10827-4

wB97x/def2-TZVPP data:
Zubatyuk, R.; Smith, J. S.; Leszczynski, J.; Isayev, O.
Accurate and Transferable Multitask Prediction of Chemical Properties with an Atoms-in-Molecules Neural Network.
Sci. Adv. 2019, 5 (8), eaav6490.
https://doi.org/10.1126/sciadv.aav6490


Dataset DOI:
https://doi.org/10.6084/m9.figshare.c.4712477.v1
"""


def ani1x_wrapper(
    hdf5_file_name: str,
    output_file_dir: str,
    local_cache_dir: str,
    force_download: bool = False,
    max_records=None,
    max_conformers_per_record=None,
    total_conformers=None,
):
    """
    This curates and processes the ANI1x dataset into a  hdf5 file.

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
    max_records: int, optional, default=None
        The maximum number of records to process.
    max_conformers_per_record: int, optional, default=None
        The maximum number of conformers to process for each record.
    total_conformers: int, optional, default=None
        The total number of conformers to process.

    """

    from modelforge.curation.ani1x_curation import ANI1xCuration

    ani1x = ANI1xCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
    )
    ani1x.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
    )
    print(f"Total records: {ani1x.total_records}")
    print(f"Total conformers: {ani1x.total_conformers}")


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/ani1x_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "0"

    # curate ANI1x test dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"ani1x_dataset_v{version}_ntc_1000.hdf5"

    ani1x_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        total_conformers=1000,
        max_conformers_per_record=10,
    )

    # curate the full ANI-1x dataset
    hdf5_file_name = f"ani1x_dataset_v{version}.hdf5"
    ani1x_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
    )


if __name__ == "__main__":
    main()

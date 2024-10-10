"""
This script will generate hdf5 datafiles for the SPICE114 dataset using the SPICE114Curation class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule, limited to elements
    that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]
 - the full dataset, limited to elements that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]

Small-molecule/Protein Interaction Chemical Energies (SPICE).
The SPICE dataset contains 1.1 million conformations for a diverse set of small molecules,
dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
uncharged molecules, and a wide range of covalent and non-covalent interactions.
It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.

Reference:
Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

Dataset DOI:
https://doi.org/10.5281/zenodo.8222043

"""


def spice114_wrapper(
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
    This curates and processes the SPICE 114 dataset into an hdf5 file.

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
    from modelforge.curation.spice_1_curation import SPICE1Curation

    spice_114 = SPICE1Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )
    spice_114.process(
        force_download=force_download,
        max_records=max_records,
        max_conformers_per_record=max_conformers_per_record,
        total_conformers=total_conformers,
        limit_atomic_species=limit_atomic_species,
    )
    print(f"Total records: {spice_114.total_records}")
    print(f"Total conformers: {spice_114.total_conformers}")


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/spice_114_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1"
    # version of the dataset to curate
    version_select = f"v_0"

    # version v_0 corresponds to SPICE 1.1.4 release
    # curate ANI1x test dataset with 1000 total conformers, max of 10 conformers per record
    ani2x_elements = ["H", "C", "N", "O", "F", "Cl", "S"]

    # curate SPICE 114 dataset with 1000 total conformers, max of 10 conformers per record
    # limited to elements that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]
    hdf5_file_name = f"spice_114_dataset_v{version}_ntc_1000_HCNOFClS.hdf5"

    spice114_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_conformers_per_record=10,
        total_conformers=1000,
        limit_atomic_species=ani2x_elements,
    )

    # curate the full SPICE 114 dataset limited to elements that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]

    hdf5_file_name = f"spice_114_dataset_v{version}_HCNOFClS.hdf5"

    spice114_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        limit_atomic_species=ani2x_elements,
    )

    # curate SPICE 114 dataset with 1000 total conformers, max of 10 conformers per record
    hdf5_file_name = f"spice_114_dataset_v{version}_ntc_1000.hdf5"

    spice114_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
        max_conformers_per_record=10,
        total_conformers=1000,
    )

    # curate the full SPICE 114 dataset
    hdf5_file_name = f"spice_114_dataset_v{version}.hdf5"

    spice114_wrapper(
        hdf5_file_name,
        output_file_dir,
        local_cache_dir,
        force_download=False,
        version_select=version_select,
    )


if __name__ == "__main__":
    main()

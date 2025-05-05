"""
This script will generate hdf5 datafiles for the SPICE2 dataset using the SPICE2Curation class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule, limited to elements
    that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]
 - the full dataset, limited to elements that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]

Small-molecule/Protein Interaction Chemical Energies (SPICE). The SPICE dataset contains conformations for a diverse
set of small molecules, dimers, dipeptides, and solvated amino acids. It provides both forces and energies calculated
at the ωB97M-D3(BJ)/def2-TZVPPD level of theory using Psi4.

Reference to the original SPICE 1 dataset publication:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

Dataset DOI:
https://doi.org/10.5281/zenodo.8222043

"""


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/spice2"
    local_cache_dir = f"{local_prefix}/spice2_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.1"
    # version of the dataset to curate
    version_select = f"v_0"

    # version v_0 corresponds to SPICE 2.0.1
    # start with processing the full dataset
    from modelforge.curate.datasets.spice_2_curation import SPICE2Curation

    spice2_dataset = SPICE2Curation(
        dataset_name="spice2",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    spice2_dataset.process(force_download=False)

    ani2x_elements = ["H", "C", "N", "O", "F", "Cl", "S"]

    # curate SPICE 2.0.1 dataset with 1000 total configurations, max of 10 conformers per record
    # limited to the elements that will work with ANI2x
    hdf5_file_name = f"spice_2_dataset_v{version}_ntc_1000_HCNOFClS.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        atomic_species_to_limit=ani2x_elements,
    )

    print("SPICE2: 1000 configuration subset limited to ANI2x elements")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # curate the full SPICE 2.0.1 dataset, limited to the elements that will work with ANI2x
    hdf5_file_name = f"spice_2_dataset_v{version}_HCNOFClS.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=ani2x_elements,
    )

    print("SPICE2: full dataset limited to ANI2x elements")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # curate the test SPICE 2.0.1 dataset with 1000 total configurations, max of 10 configurations per record
    hdf5_file_name = f"spice_2_dataset_v{version}_ntc_1000.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
    )

    print("SPICE2: 1000 configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # curate the full SPICE 2.0.1 dataset
    hdf5_file_name = f"spice_2_dataset_v{version}.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name, output_file_dir=output_file_dir
    )
    print("SPICE2: full dataset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

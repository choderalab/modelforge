"""
This script will generate hdf5 datafiles for the SPICE2 OpenFF dataset using the SPICE2OpenFFCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule.
 - a subset of the dataset with 1000 conformers, with a maximum of 10 conformers per molecule, limited to elements
    that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]
 - the full dataset, limited to elements that are compatible with ANI2x ["H", "C", "N", "O", "F", "Cl", "S"]

Small-molecule/Protein Interaction Chemical Energies (SPICE). The SPICE dataset contains conformations for a diverse
set of small molecules, dimers, dipeptides, and solvated amino acids.
All QM datapoints retrieved were generated using B3LYP-D3BJ/DZVP level of theory.
This is the default theory used for force field development by the Open Force Field Initiative.

SPICE 2 github repository:
    https://github.com/openmm/spice-dataset

Reference to SPICE 2 publication:
    Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E.
    Nutmeg and SPICE: models and data for biomolecular machine learning.
    Journal of chemical theory and computation, 20(19), 8583-8593 (2024).
    https://doi.org/10.1021/acs.jctc.4c00794

Reference to the original SPICE 1 dataset publication:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

Dataset DOI:
https://doi.org/10.5281/zenodo.8222043

"""

from modelforge.curate.datasets.spice_2_openff_curation import SPICE2OpenFFCuration
from modelforge.curate.utils import VersionMetadata
from openff.units import unit


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/spice2_openff"
    local_cache_dir = f"{local_prefix}/spice2_openff_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.1"
    # version of the dataset to curate
    version_select = f"v_0"

    # version v_0 corresponds to SPICE 1.1.4
    # start with processing the full dataset

    spice2_dataset = SPICE2OpenFFCuration(
        dataset_name="spice2_openff",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    spice2_dataset.process(force_download=False, n_threads=8)

    ani2x_elements = ["H", "C", "N", "O", "F", "Cl", "S"]

    #######################################################
    # 1000 configuration test set limited to ANI2x elements
    #######################################################

    # curate SPICE 1.1.4 dataset with 1000 total configurations, max of 10 conformers per record
    # limited to the elements that will work with ANI2x
    hdf5_file_name = f"spice_2_openff_dataset_v{version}_ntc_1000_HCNOFClS.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        atomic_species_to_limit=ani2x_elements,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"nc_1000_HCNOFClS_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the SPICE2 openff dataset designed
        to be compatible with modelforge. This dataset contains {total_records} unique records
        for {total_configs} total configurations, with a maximum of 10 configurations per record.
        The dataset is limited to the elements that are compatible with ANI2x: {ani2x_elements}"""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_energy",
            "dispersion_correction_energy",
            "dft_total_energy",
            "dft_force",
            "dispersion_correction_force",
            "dft_total_force",
            "mbis_charges",
            "scf_dipole",
        ],
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("SPICE2_openff: 1000 configuration subset limited to ANI2x elements")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    ########################################################
    # full dataset limited to ANI2x elements
    ########################################################

    hdf5_file_name = f"spice_2_openff_dataset_v{version}_HCNOFClS.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=ani2x_elements,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"full_dataset_HCNOFClS_v{version}"
    about = f"""This provides a curated hdf5 file for the SPICE2 openff dataset designed
        to be compatible with modelforge. This dataset contains {total_records} unique records
        for {total_configs} total configurations.
        The dataset is limited to the elements that are compatible with ANI2x: {ani2x_elements}"""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_energy",
            "dispersion_correction_energy",
            "dft_total_energy",
            "dft_force",
            "dispersion_correction_force",
            "dft_total_force",
            "mbis_charges",
            "scf_dipole",
        ],
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("SPICE2_openff: full dataset limited to ANI2x elements")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    ##########################################
    # 1000 configuration test set
    ##########################################

    hdf5_file_name = f"spice_2_openff_dataset_v{version}_ntc_1000.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the SPICE2 openff dataset designed
            to be compatible with modelforge. This dataset contains {total_records} unique records
            for {total_configs} total configurations, with a maximum of 10 configurations per record.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_energy",
            "dispersion_correction_energy",
            "dft_total_energy",
            "dft_force",
            "dispersion_correction_force",
            "dft_total_force",
            "mbis_charges",
            "scf_dipole",
        ],
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("SPICE2_openff: 1000 configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    #########################################################
    # full dataset
    #########################################################

    # curate the full SPICE 1.1.4 dataset
    hdf5_file_name = f"spice_2_openff_dataset_v{version}.hdf5"

    total_records, total_configs = spice2_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the SPICE2 openff dataset designed
        to be compatible with modelforge. This dataset contains {total_records} unique records
        for {total_configs} total configurations.
        """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_energy",
            "dispersion_correction_energy",
            "dft_total_energy",
            "dft_force",
            "dispersion_correction_force",
            "dft_total_force",
            "mbis_charges",
            "scf_dipole",
        ],
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("SPICE2_openff: full dataset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

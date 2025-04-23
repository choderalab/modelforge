"""
This script will generate hdf5 datafiles for the tmQM-xtb dataset using the tmQMXTBCuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.


Routines to process the tmQM-xtb dataset into a curated hdf5 file.

This dataset uses  configurations from the original tmQM dataset as starting points
for semi-emperical calculations using the GFN2-XTB (calculated using TBlite) that perform MD-based sampling.
MD sampling using the atomic simulation environment (ASE) and was performed using the Langevin thermostat,
with a time step of 1.0 fs, and friction of 0.01 1/fs.

v0 of the dataset performings sampling at T=400K, generating 10 snapshots per molecule (the first corresponding
to the original, energy minimized state in tmQM) with 100 timesteps spacing between each snapshot. The goal
was to primarily capture the fluctuations around the equilibrium, rather than large scale conformation changes.

To remove potentially problematic configurations, a filtering criteria is applied during curation:
- The initial configurations of the molecules undergo bond perception using RDKit, with bond distances recorded.
- The relative change in bond length is calculated for each snapshot.
- If the relative change in any bond length is more than 0.09 the snapshot is removed from the dataset.


The original tmQM dataset contains the geometries and properties of 108,541 (in the 13Aug24 release)
mononuclear complexes extracted from the Cambridge Structural Database, including Werner, bioinorganic, and
organometallic complexes based on a large variety of organic ligands and 30 transition metals
(the 3d, 4d, and 5d from groups 3 to 12).
All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

The scripts used to generate the tmQM-xtb dataset are available at:
https://github.com/chrisiacovella/xtb_Config_gen

The tmQM-xtb dataset is avialble from zenodo:
 10.5281/zenodo.14894964 (v0)

Citation to the original tmQM dataset:

David Balcells and Bastian Bjerkem Skjelstad,
tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes
Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146
DOI: 10.1021/acs.jcim.0c01041

Original dataset source: https://github.com/uiocompcat/tmQM

forked to be able to create releases:  https://github.com/chrisiacovella/tmQM/
"""

from modelforge.curate.datasets.tmqm_xtb_curation import tmQMXTBCuration
from openff.units import unit


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/tmqm_xtb_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version_out = "1.1"
    # version of the dataset to curate
    version_select = "v1_PdZnFeCuNiPtIrRhCrAg_300K"

    tmqm_xtb = tmQMXTBCuration(
        dataset_name="tmqm_xtb",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    cutoff = 0.15 * unit.angstrom

    # tmqm_xtb.load_from_db(local_cache_dir, "tmqm_xtb.sqlite")
    tmqm_xtb.process(force_download=False)
    tmqm_xtb.process(force_download=False, cutoff=cutoff)

    #
    # Curate the test limited to transition metals Pd, Zn, Fe, Cu and
    # organics C, H, P, S, O, N, F, Cl, Br

    hdf5_file_name = f"tmqm_xtb_dataset_PdZnFeCu_T300_v{version_out}.hdf5"

    atomic_species_to_limit = [
        "Pd",
        "Zn",
        "Fe",
        "Cu",
        "C",
        "H",
        "P",
        "S",
        "O",
        "N",
        "F",
        "Cl",
        "Br",
    ]
    total_records, total_configs = tmqm_xtb.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=atomic_species_to_limit,
    )

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # Curate the test limited to transition metals Pd, Zn, Fe, Cu, Ni, Pt, Ir, Rh, Cr, Ag and
    # organics C, H, P, S, O, N, F, Cl, Br

    hdf5_file_name = f"tmqm_xtb_dataset_PdZnFeCuNiPtIrCrAg_T300_v{version_out}.hdf5"

    atomic_species_to_limit = [
        "Pd",
        "Zn",
        "Fe",
        "Cu",
        "Ni",
        "Pt",
        "Ir",
        "Rh",
        "Cr",
        "Ag",
        "C",
        "H",
        "P",
        "S",
        "O",
        "N",
        "F",
        "Cl",
        "Br",
    ]
    total_records, total_configs = tmqm_xtb.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=atomic_species_to_limit,
    )

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # 1000 configurations test set
    hdf5_file_name = f"tmqm_xtb_dataset_PdZnFeCu_T300_ntc_1000_v{version_out}.hdf5"
    atomic_species_to_limit = [
        "Pd",
        "Zn",
        "Fe",
        "Cu",
        "C",
        "H",
        "P",
        "S",
        "O",
        "N",
        "F",
        "Cl",
        "Br",
    ]
    total_records, total_configs = tmqm_xtb.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=atomic_species_to_limit,
        total_configurations=1000,
    )

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # 1000 configurations test set
    hdf5_file_name = f"tmqm_xtb_dataset_PdZnFeCu_T300_first10_v{version_out}.hdf5"
    atomic_species_to_limit = [
        "Pd",
        "Zn",
        "Fe",
        "Cu",
        "C",
        "H",
        "P",
        "S",
        "O",
        "N",
        "F",
        "Cl",
        "Br",
    ]
    total_records, total_configs = tmqm_xtb.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        atomic_species_to_limit=atomic_species_to_limit,
        max_configurations_per_record=10,
    )

    print("primary configuration subset of first 10 configs only")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

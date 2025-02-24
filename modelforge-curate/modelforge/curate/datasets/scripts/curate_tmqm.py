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


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files"
    local_cache_dir = f"{local_prefix}/tmqm_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "0"
    version_out = "1.0"
    # version of the dataset to curate
    version_select = f"v_{version}"

    force_download = False
    from modelforge.curate.datasets.tmqm_curation import tmQMCuration

    tmqm = tmQMCuration(
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    tmqm.process(
        force_download=force_download,
    )

    # curate the full dataset
    hdf5_file_name = f"tmqm_dataset_v{version_out}.hdf5"

    total_records, total_configs = tmqm.to_hdf5(
        hdf5_file_name=hdf5_file_name, output_file_dir=output_file_dir
    )

    print("full dataset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # Curate the test dataset with 1000 total conformers
    # only a single config per record
    hdf5_file_name = f"tmqm_dataset_v{version_out}_ntc_1000.hdf5"

    total_records, total_configs = tmqm.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_conformers=1000,
    )

    print(" 1000 conformer subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

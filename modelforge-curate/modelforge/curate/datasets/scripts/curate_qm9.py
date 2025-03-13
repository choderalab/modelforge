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


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/qm9_dataset"
    local_cache_dir = f"{local_prefix}/qm9_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.0"
    # version of the dataset to curate
    version_select = f"v_0"
    # Curate the test dataset with 1000 total conformers
    from modelforge.curate.datasets.qm9_curation import QM9Curation

    qm9_dataset = QM9Curation(
        dataset_name="qm9",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )
    qm9_dataset.process(force_download=False)

    # Curates the full dataset
    hdf5_file_name = f"qm9_dataset_v{version}.hdf5"

    n_total_records, n_total_configs = qm9_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name, output_file_dir=output_file_dir
    )
    print("full dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    # Curates the test dataset with 1000 total conformers
    # only a single config per record

    hdf5_file_name = f"qm9_dataset_v{version}_ntc_1000.hdf5"
    n_total_records, n_total_configs = qm9_dataset.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
    )
    print(" 1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

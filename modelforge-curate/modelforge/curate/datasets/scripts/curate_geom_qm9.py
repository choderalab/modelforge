"""
This script will generate hdf5 datafiles for the QM9 dataset using the GEOMQM9Curation class.


This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.


The original QM9 dataset includes 133,885 organic molecules with up to nine total heavy atoms (C,O,N,or F; excluding H).

Citation:   Ramakrishnan, R., Dral, P., Rupp, M. et al.
           "Quantum chemistry structures and properties of 134 kilo molecules."
            Sci Data 1, 140022 (2014).
            https://doi.org/10.1038/sdata.2014.22

DOI for original qm9 dataset: 10.6084/m9.figshare.c.978904.v5

The Geometric Ensemble Of Molecules (GEOM) dataset contains conformers for 133,000 species from the QM9 dataset.
Conformer structures were generated with CREST and GFN2-XTB.  Energies were evaluated using
DFT via ORCA 5.0.2  using the r2scan-3c functional and mTZVPP basis.

Citation:   Axelrod, S., Gómez-Bombarelli, R.
            GEOM, energy-annotated molecular conformations for property prediction and molecular generation.
            Sci Data 9, 185 (2022).
            https://doi.org/10.1038/s41597-022-01288-4

DOI for dataset: https://doi.org/10.7910/DVN/JNGTDF
Download link for qm9 subset of the GEOM dataset:  https://dataverse.harvard.edu/api/access/datafile/4327190
"""

from modelforge.curate.datasets.geom_qm9_curation import GEOMQM9Curation
from modelforge.curate.utils import VersionMetadata


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/geom_qm9_data"
    local_cache_dir = f"{local_prefix}/geom_qm9_data"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.0"
    # version of the dataset to curate
    version_select = f"v_4"
    # Curate the test dataset with 1000 total conformers
    # only a single config per record
    geom_qm9_data = GEOMQM9Curation(
        dataset_name="geom_qm9",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )
    # geom_qm9_data.load_from_db(local_cache_dir, "geom_qm9.sqlite")
    geom_qm9_data.process(force_download=False)

    ###############
    # full dataset
    ##############
    # Curates the full dataset
    hdf5_file_name = f"geom_qm9_data_v{version}.hdf5"

    n_total_records, n_total_configs = geom_qm9_data.to_hdf5(
        hdf5_file_name=hdf5_file_name, output_file_dir=output_file_dir
    )

    version_name = f"full_dataset_v{version}"
    about = f"""This provides a curated hdf5 file for the GEOM qm9 dataset designed
        to be compatible with modelforge. This dataset contains {n_total_records} unique records 
        for {n_total_configs} total configurations.  Note, the dataset comes from GEOM which uses 
        gfn2-xtb and CREST to generate conformers."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_total_energy",
        ],
    )

    # this will also compress the hdf5 file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("full dataset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    #################
    # 1000 configuration version
    #################
    # Curates the test dataset with 1000 total configurations
    # only a single config per record

    hdf5_file_name = f"geom_qm9_data_v{version}_ntc_1000.hdf5"
    n_total_records, n_total_configs = geom_qm9_data.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
    )
    version_name = f"nc_1000_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the GEOM qm9 dataset designed
        to be compatible with modelforge. This dataset contains {n_total_records} unique records 
        for {n_total_configs} total configurations.Note, the dataset comes from GEOM which uses 
        gfn2-xtb and CREST to generate conformers."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_total_energy",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print(" 1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    #################
    # 10 configuration version
    #################
    # Curates the test dataset with 10 total configuration

    hdf5_file_name = f"geom_qm9_data_v{version}_ntc_10.hdf5"
    n_total_records, n_total_configs = geom_qm9_data.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=10,
    )
    version_name = f"nc_10_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the GEOM qm9 dataset designed
        to be compatible with modelforge. This dataset contains {n_total_records} unique records 
        for {n_total_configs} total configurations. Note, the dataset comes from GEOM which uses 
        gfn2-xtb and CREST to generate conformers."""

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        remote_dataset=False,
        available_properties=[
            "atomic_numbers",
            "positions",
            "total_charge",
            "dft_total_energy",
        ],
    )

    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("10 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

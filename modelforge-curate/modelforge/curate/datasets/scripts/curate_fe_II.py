"""
This script will generate hdf5 datafiles for the Fe(II) dataset using the FeIICuration class.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with a maximum of 1000 conformers.

he Fe(II) dataset includes 28834 total configurations for 383 unique molecules Fe(II) organometallic complexes.
    Specifically, this includes 15568 HS geometries and 13266 LS geometries.
    These complexes originate from the Cambridge Structural Database (CSD) as curated by
    Nandy, et al. (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214),
    and were filtered into “computation-ready” complexes, (those where both oxidation states and charges are already
    specified without hydrogen atoms missing in the structures), following the procedure outlined by Arunachalam, et al.
    (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700)


    The original Fe (II) dataset is available from github:
    https://github.com/Neon8988/Iron_NNPs

    The code uses a fork of the original dataset, to enable clear versioning (i.e. a release):
    https://github.com/chrisiacovella/Iron_NNPs

    Citation to the original  dataset:

    Modeling Fe(II) Complexes Using Neural Networks
    Hongni Jin and Kenneth M. Merz Jr.
    Journal of Chemical Theory and Computation 2024 20 (6), 2551-2558
    DOI: 10.1021/acs.jctc.4c00063
"""

from modelforge.curate.datasets.fe_II_curation import FeIICuration
from openff.units import unit


def main():
    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/fe_II_dataset"
    local_cache_dir = f"{local_prefix}/tmqm_xtb_dataset"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version_out = "1.1"
    # version of the dataset to curate
    version_select = "v_0"

    fe_II = FeIICuration(
        dataset_name="fe_II",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    # fe_II.load_from_db(local_cache_dir, "fe_II.sqlite")
    fe_II.process(force_download=False)

    hdf5_file_name = f"fe_II_v{version_out}.hdf5"

    total_records, total_configs = fe_II.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
    )

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")

    # 1000 configurations test set
    hdf5_file_name = f"fe_II_ntc_1000_v{version_out}.hdf5"

    total_records, total_configs = fe_II.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
    )

    print("primary configuration subset")
    print(f"Total records: {total_records}")
    print(f"Total configs: {total_configs}")


if __name__ == "__main__":
    main()

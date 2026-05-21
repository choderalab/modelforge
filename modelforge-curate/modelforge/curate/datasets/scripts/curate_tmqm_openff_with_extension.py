"""
This script will generate hdf5 datafiles for the tmqm openff dataset.

This will generate separate HDF5 files for:
 - the full dataset
 - a subset of the dataset with 1000 configurations, with a maximum of 10 configurations per molecule.


"""

from modelforge.curate.datasets.tmqm_openff_curation import tmQMOpenFFCuration
from modelforge.curate.utils import VersionMetadata


def main():

    from openff.units import unit

    # define the location where to store and output the files
    import os

    local_prefix = os.path.expanduser("~/mf_datasets")
    output_file_dir = f"{local_prefix}/hdf5_files/tmqm_openff_dataset"
    local_cache_dir = f"{local_prefix}/tmqm_openff_dataset_cache"

    # We'll want to provide some simple means of versioning
    # if we make updates to either the underlying dataset, curation modules, or parameters given to the code
    version = "1.4"
    # version of the dataset to curate
    version_select = f"v_0"

    # The dataset is composed of our initial "base" dataset, where configurations are sampled at 100K and
    # spin multiplicity = 1, then each configuration is evaluated with DFT at spin multiplicities 1,3,5
    # The extensions contain additional configurations, where the MD sampling is done at different spin multiplicities,
    # and then evaluated with DFT at that same spin mulitiplicity.
    # This helps to ensure that we are sampling configurations around the energy minima associated with different
    # spin multiplicities.

    # set up a instance of the curation class for the "base" dataset
    tmqm_openff_base = tmQMOpenFFCuration(
        dataset_name="tmqm_openff",
        local_cache_dir=local_cache_dir,
        version_select=version_select,
    )

    # rather than fetching the dataset directly, we will use a local view file
    # tmqm_openff_base.process(
    #     qcportal_view_filename="dataset_419_view.sqlite",
    #     qcportal_view_path="/home/cri/mf_datasets/tmqm_openff_dataset/dataset_download_Feb26",
    #     force_download=False,
    # )
    tmqm_openff_base.load_from_db(
        local_db_dir=local_cache_dir,
        local_db_name="tmqm_openff.sqlite",
    )
    # set up the extensions
    # spin multiplicty = 1
    tmqm_openff_ext_sm1 = tmQMOpenFFCuration(
        dataset_name="tmqm_openff_ext_sm1",
        local_cache_dir=local_cache_dir,
        version_select="v0_ext_sm1",
    )

    # tmqm_openff_ext_sm1.process(
    #     qcportal_view_filename="dataset_469_view.sqlite",
    #     qcportal_view_path="/home/cri/mf_datasets/tmqm_openff_dataset/dataset_download_May13",
    #     force_download=False,
    # )

    tmqm_openff_ext_sm1.load_from_db(
        local_db_dir=local_cache_dir,
        local_db_name="tmqm_openff_ext_sm1.sqlite",
    )
    # spin multiplicity = 3
    tmqm_openff_ext_sm3 = tmQMOpenFFCuration(
        dataset_name="tmqm_openff_ext_sm3",
        local_cache_dir=local_cache_dir,
        version_select="v0_ext_sm3",
    )

    # tmqm_openff_ext_sm3.process(
    #     qcportal_view_filename="dataset_468_view.sqlite",
    #     qcportal_view_path="/home/cri/mf_datasets/tmqm_openff_dataset/dataset_download_May13",
    #     force_download=False,
    # )
    tmqm_openff_ext_sm3.load_from_db(
        local_db_dir=local_cache_dir,
        local_db_name="tmqm_openff_ext_sm3.sqlite",
    )
    # spin multiplicity = 5
    tmqm_openff_ext_sm5 = tmQMOpenFFCuration(
        dataset_name="tmqm_openff_ext_sm5",
        local_cache_dir=local_cache_dir,
        version_select="v0_ext_sm5",
    )
    # tmqm_openff_ext_sm5.process(
    #     qcportal_view_filename="dataset_463_view.sqlite",
    #     qcportal_view_path="/home/cri/mf_datasets/tmqm_openff_dataset/dataset_download_May13",
    #     force_download=False,
    # )
    tmqm_openff_ext_sm5.load_from_db(
        local_db_dir=local_cache_dir,
        local_db_name="tmqm_openff_ext_sm5.sqlite",
    )
    # we want to ensure that all records in the extension get added
    # so let us do a quick look to see if any record_names in the extensions are missed in the base

    missed_records = []
    for record_name in tmqm_openff_ext_sm1.dataset.record_names():
        if not record_name in tmqm_openff_base.dataset.record_names():
            print(f"record {record_name} in ext_sm1 not in base")
            missed_records.append(record_name)
    for record_name in tmqm_openff_ext_sm3.dataset.record_names():
        if not record_name in tmqm_openff_base.dataset.record_names():
            print(f"record {record_name} in ext_sm3 not in base")
            missed_records.append(record_name)
    for record_name in tmqm_openff_ext_sm5.dataset.record_names():
        if not record_name in tmqm_openff_base.dataset.record_names():
            print(f"record {record_name} in ext_sm5 not in base")
            missed_records.append(record_name)

    # print the total number of records missed
    print(f"# missed records {len(missed_records)}")

    # loop over all the records in the dataset and see if they are in the extensions and merge them
    ext_sm1_record_names = tmqm_openff_ext_sm1.dataset.record_names()
    ext_sm3_record_names = tmqm_openff_ext_sm3.dataset.record_names()
    ext_sm5_record_names = tmqm_openff_ext_sm5.dataset.record_names()

    for record_name in tmqm_openff_base.dataset.record_names():
        record_base = tmqm_openff_base.dataset.get_record(record_name)
        if record_name in ext_sm1_record_names:
            record_ext_sm1 = tmqm_openff_ext_sm1.dataset.get_record(record_name)
            record_base.merge(record_ext_sm1)
        if record_name in ext_sm3_record_names:
            record_ext_sm3 = tmqm_openff_ext_sm3.dataset.get_record(record_name)
            record_base.merge(record_ext_sm3)
        if record_name in ext_sm5_record_names:
            record_ext_sm5 = tmqm_openff_ext_sm5.dataset.get_record(record_name)
            record_base.merge(record_ext_sm5)

        # update the record in the base dataset
        tmqm_openff_base.dataset.update_record(record_base)

    for record_name in missed_records:
        if record_name in ext_sm1_record_names:
            record_ext = tmqm_openff_ext_sm1.dataset.get_record(record_name)
            # since this will be ecountered first, there is no chance the record already exists
            tmqm_openff_base.dataset.add_record(record_ext)

        if record_name in ext_sm3_record_names:
            record_ext = tmqm_openff_ext_sm3.dataset.get_record(record_name)
            # since we may have already added this record above, we need to check to see if it exists
            # if it does we will merge and update
            # other wise we will add it
            if record_name in tmqm_openff_base.dataset.record_names():
                record_base = tmqm_openff_base.dataset.get_record(record_name)
                record_base.merge(record_ext)
                tmqm_openff_base.dataset.update_record(record_base)
            else:
                tmqm_openff_base.dataset.add_record(record_ext)
        if record_name in ext_sm5_record_names:
            record_ext = tmqm_openff_ext_sm5.dataset.get_record(record_name)
            if record_name in tmqm_openff_base.dataset.record_names():
                record_base = tmqm_openff_base.dataset.get_record(record_name)
                record_base.merge(record_ext)
                tmqm_openff_base.dataset.update_record(record_base)
            else:

                tmqm_openff_base.dataset.add_record(record_ext)

    # we now need to loop over all the records and calculate the self_energy as a function of charge and spin multiplicity
    # this uses the regressed values calculated above

    available_properties = (
        [
            "atomic_numbers",
            "positions",
            "total_charge",
            "per_system_spin_multiplicity",
            "dft_total_energy",
            "dft_total_force",
            "scf_dipole",
            "scf_quadrupole",
            "mulliken_partial_charges",
            "lowdin_partial_charges",
            "spin_multiplicity_per_atom",
        ],
    )
    #################
    # 1000 configuration version
    #################

    # curate dataset with 1000 total configurations, max of 10 configurations per record
    hdf5_file_name = f"tmqm_openff_dataset_v{version}_ntc_1000.hdf5"

    n_total_records, n_total_configs = tmqm_openff_base.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_configurations_per_record=10,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )
    version_name = f"nc_1000_v{version}"
    about = f"""This provides a curated hdf5 file for a subset of the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with a maximum of 10 configurations per record.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """
    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration subset")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # full dataset
    ##############
    # curate the full dataset
    hdf5_file_name = f"tmqm_openff_dataset_v{version}.hdf5"
    print("total dataset")

    n_total_records, n_total_configs = tmqm_openff_base.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
    )

    version_name = f"full_dataset_v{version}"

    about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    ###############
    # full dataset, spin multiplicity restrictions
    ##############
    for sm in [1, 3, 5]:
        for charge in [-1, 0, 1, None]:
            # curate the full dataset
            if not charge is None:
                if charge == -1:
                    charge_str = "neg1"
                elif charge == 0:
                    charge_str = "neutral"
                elif charge == 1:
                    charge_str = "plus1"
                hdf5_file_name = (
                    f"tmqm_openff_dataset_sm{sm}_{charge_str}_v{version}.hdf5"
                )
            else:
                hdf5_file_name = f"tmqm_openff_dataset_sm{sm}_v{version}.hdf5"

            print("total dataset")

            n_total_records, n_total_configs = tmqm_openff_base.to_hdf5(
                hdf5_file_name=hdf5_file_name,
                output_file_dir=output_file_dir,
                max_force=1.0 * unit.hartree / unit.bohr,
                max_force_key="dft_total_force",
                spin_multiplicity_to_limit=sm,
                spin_multiplicity_key="per_system_spin_multiplicity",
                total_charge_to_limit=charge,
                total_charge_key="total_charge",
            )

            version_name = f"full_dataset_sm{sm}_v{version}"

            if charge is not None:
                about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed 
                                to be compatible with modelforge. This dataset contains {n_total_records} unique records
                                for {n_total_configs} total configurations, restricted to spin multiplicity {sm} and charge {charge}.
                                This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
                                """
            else:
                about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed 
                                to be compatible with modelforge. This dataset contains {n_total_records} unique records
                                for {n_total_configs} total configurations, restricted to spin multiplicity {sm}.
                                This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
                                """

            metadata = VersionMetadata(
                version_name=version_name,
                about=about,
                hdf5_file_name=hdf5_file_name,
                hdf5_file_dir=output_file_dir,
                available_properties=available_properties,
            )
            # we need to compress the hdf5 file to get the checksum and length for the gzipped file
            metadata.to_yaml(
                file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
            )

            print(f"Total records: {n_total_records}")
            print(f"Total configs: {n_total_configs}")

    ######################################
    # 1000 conformer, last configuration only
    ######################################
    # curate dataset with 1000 total configurations, last only
    hdf5_file_name = f"tmqm_openff_dataset_v{version}_ntc_1000_minimal.hdf5"

    n_total_records, n_total_configs = tmqm_openff_base.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        total_configurations=1000,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )
    version_name = f"nc_1000_v{version}_minimal"
    about = f"""This provides a curated hdf5 file for a subset of the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with only the final configuration of the optimization.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )

    print("1000 configuration subset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")

    #######################################3
    # full dataset last configuration only
    #######################################3

    # curate the full dataset last config only

    hdf5_file_name = f"tmqm_openff_dataset_v{version}_minimal.hdf5"

    n_total_records, n_total_configs = tmqm_openff_base.to_hdf5(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=output_file_dir,
        max_force=1.0 * unit.hartree / unit.bohr,
        max_force_key="dft_total_force",
        final_configuration_only=True,
    )
    version_name = f"full_dataset_v{version}_minimal"
    about = f"""This provides a curated hdf5 file for the tmqm openff dataset designed
            to be compatible with modelforge. This dataset contains {n_total_records} unique records
            for {n_total_configs} total configurations, with only the final configuration of the optimization.
            This excludes any configurations where the magnitude of any forces on the atoms are greater than 1 hartree/bohr.
            """

    metadata = VersionMetadata(
        version_name=version_name,
        about=about,
        hdf5_file_name=hdf5_file_name,
        hdf5_file_dir=output_file_dir,
        available_properties=available_properties,
    )
    # we need to compress the hdf5 file to get the checksum and length for the gzipped file
    metadata.to_yaml(
        file_name=f"{version_name}_metadata.yaml", file_path=output_file_dir
    )
    print("full dataset last configurations only")
    print(f"Total records: {n_total_records}")
    print(f"Total configs: {n_total_configs}")


if __name__ == "__main__":
    main()

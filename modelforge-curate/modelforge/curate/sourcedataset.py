from modelforge.utils.units import GlobalUnitSystem, chem_context
from modelforge.curate.properties import (
    PropertyBaseModel,
    PropertyClassification,
    PropertyType,
    Positions,
    Energies,
    AtomicNumbers,
)
from modelforge.curate.record import Record

from openff.units import unit

import numpy as np
import os
from typing import Union, List, Type, Optional

from typing_extensions import Self

from loguru import logger as log
from sqlitedict import SqliteDict


class SourceDataset:
    def __init__(
        self,
        name: str,
        append_property: bool = False,
        local_db_dir: Optional[str] = "./",
        local_db_name: Optional[str] = None,
        read_from_local_db=False,
    ):
        """
        Class to hold a dataset of properties for a given dataset name

        Parameters
        ----------
        name: str
            Name of the dataset
        append_property: bool, optional, default=False
            If True, append an array to existing array if a property with the same name is added multiple times to a record.
            If False, an error will be raised if trying to add a property with the same name already exists in a record
        local_db_dir: str, optional, default="./"
            Directory to store the local database
        local_db_name: str, optional, default=None
            Name of the cache database. If None, the dataset name will be used.
        read_from_local_db: bool, optional, default=False
            If True, use an existing database.
            If False, removes the existing database and creates a new one.
        """

        self.name = name
        self.records = {}
        self.append_property = append_property
        self.local_db_dir = local_db_dir
        os.makedirs(self.local_db_dir, exist_ok=True)

        if local_db_name is None:
            self.local_db_name = name.replace(" ", "_") + ".sqlite"
        else:
            self.local_db_name = local_db_name

        self.read_from_local_db = read_from_local_db
        if self.read_from_local_db == False:
            if os.path.exists(f"{self.local_db_dir}/{self.local_db_name}"):
                log.warning(
                    f"Database file {self.local_db_name} already exists in {self.local_db_dir}. Removing it."
                )
                self._remove_local_db()
        else:
            if not os.path.exists(f"{self.local_db_dir}/{self.local_db_name}"):
                log.warning(
                    f"Database file {self.local_db_name} does not exist in {self.local_db_dir}"
                )
                raise FileNotFoundError(
                    f"Database file {self.local_db_name} does not exist in {self.local_db_dir}."
                )
            else:
                # populate the records dict with the keys from the database
                with SqliteDict(
                    f"{self.local_db_dir}/{self.local_db_name}",
                    autocommit=True,
                ) as db:
                    keys = list(db.keys())
                    for key in keys:
                        self.records[key] = key

    def keys(self):
        """
        Return the keys of the records in the dataset

        Returns
        -------
        list: list of record keys
        """
        return list(self.records.keys())

    def total_records(self):
        """
        Get the total number of records in the dataset.

        Returns
        -------

        """
        return len(self.records)

    def total_configs(self):
        """
        Get the total number of configurations in the dataset.
        """
        total_config = 0
        for record in self.records.keys():
            record = self.get_record(record)
            total_config += record.n_configs
        return total_config

    def create_record(
        self,
        name: str,
        properties: Optional[List[Type[PropertyBaseModel]]] = None,
    ):
        """
        Create a record in the dataset. If properties are provided, they will be added to the record.

        Parameters
        ----------
        name: str
            Name of the record/
        properties: List[Type[PropertyBaseModel]], optional, default=None
            List of properties to add to the record. If not provided, an empty record will be created.

        Returns
        -------

        """
        assert isinstance(name, str)

        # I think this should error out if we've already encountered a name, as that would imply
        # some issue with the dataset construction
        if name in self.records.keys():
            raise ValueError(f"Record with name {name} already exists in the dataset")

        self.records[name] = name
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            db[name] = Record(name, self.append_property)
        if properties is not None:
            self.add_properties(name, properties)

    def add_record(self, record: Record):
        """
        Add a record to the dataset.

        Note, this will raise an error if the record already exists in the dataset.
        Parameters
        ----------
        Record: Record
            Instance of the Record class to add to the dataset.

        Returns
        -------

        """

        if not isinstance(record, Record):
            raise ValueError("Input must be an instance of the Record class")

        if record.name in self.records.keys():
            log.warning(
                f"Record with name {record.name} already exists in the dataset."
            )
            raise ValueError(
                f"Record with name {record.name} already exists in the dataset."
            )
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            db[record.name] = record

        self.records[record.name] = record.name

        # self.records[record.name] = copy.deepcopy(record)

    def add_records(self, records: List[Record]):
        """
        Add a list of records to the dataset.

        Parameters
        ----------
        records: List[Record]
            List of records to add to the dataset.

        Returns
        -------

        """
        if not all([isinstance(record, Record) for record in records]):
            raise ValueError("Input must be a list of instances of the Record class")

        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            for i in range(len(records)):
                name = records[i].name

                if name in self.records.keys():
                    log.warning(
                        f"Record with name {name} already exists in the dataset."
                    )
                    raise ValueError(
                        f"Record with name {name} already exists in the dataset."
                    )
                self.records[name] = name
                db[name] = records[i]

    def update_record(self, record: Record):
        """
        Update a record in the dataset by overwriting the existing record with the input.

        This is useful if a record was accessed via get_record and then modified or for copying a record
        from a different dataset.



        Parameters
        ----------
        record: Record
            Record to update.
        Returns
        -------

        """
        if not record.name in self.records.keys():
            log.warning(
                f"Record with name {record.name} does not exist in the dataset. Use the add_record function."
            )
            raise ValueError(
                f"Record with name {record.name} does not exist in the dataset."
            )
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            db[record.name] = record

        # self.records[record.name] = copy.deepcopy(record)

    def remove_record(self, name: str):
        """
        Remove a record from the dataset.

        Parameters
        ----------
        name: str
            Name of the record to remove.

        Returns
        -------

        """
        assert isinstance(name, str)
        if name in self.records.keys():
            self.records.pop(name)
            with SqliteDict(
                f"{self.local_db_dir}/{self.local_db_name}",
                autocommit=True,
            ) as db:
                db.pop(name)
        else:
            log.warning(f"Record with name {name} does not exist in the dataset.")

    def add_properties(self, name: str, properties: List[Type[PropertyBaseModel]]):
        """
        Add a list of properties to a record in the dataset.

        Parameters
        ----------
        name: str
            Name of the record to add the properties to.
        properties: List[Type[PropertyBaseModel]]
            List of properties to add to the record.

        Returns
        -------

        """
        assert isinstance(name, str)
        # check if the record exists; if it does not add it
        if name not in self.records.keys():
            log.info(
                f"Record with name {name} does not exist in the dataset. Creating it now."
            )
            self.create_record(name)

        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            record = db[name]
            record.add_properties(properties)
            db[name] = record

        # for property in properties:
        #     self.add_property(name, property)

    def add_property(self, name: str, property: Type[PropertyBaseModel]):
        """
        Add a property to a record in the dataset.

        Parameters
        ----------
        name: str
            Name of the record to add the property to.
        property: Type[PropertyBaseModel]
            Property to add to the record.

        Returns
        -------

        """
        assert isinstance(name, str)
        # check if the record exists; if it does not add it
        if name not in self.records.keys():
            log.info(
                f"Record with name {name} does not exist in the dataset. Creating it now."
            )
            self.create_record(name)

        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            record = db[name]
            record.add_property(property)
            db[name] = record
        # self.records[name].add_property(property)

    def get_record(self, name: str):
        """
        Get a record from the dataset. Returns an instance of the Record class.

        Parameters
        ----------
        name: str
            Name of the record to get
        Returns
        -------
            Record: instance of the Record class corresponding to the record name

        """
        assert isinstance(name, str)
        from copy import deepcopy

        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            return db[name]

    def slice_record(self, name: str, min: int = 0, max: int = -1) -> Record:
        """
        Slice a record to only include a subset of configs

        Slicing occurs on all per_atom and per_system properties

        Parameters
        ----------
        name: str
            Name of the record to slice.
        min: int
            Starting index for slicing.
        max: int
            Ending index for slicing.

        Returns
        -------
            Record: A copy of the sliced record.
        """
        assert isinstance(name, str)
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            return db[name].slice_record(min=min, max=max)

        # return self.records[name].slice_record(min=min, max=max)

    def subset_dataset(
        self,
        new_dataset_name: str,
        total_records: Optional[int] = None,
        total_configurations: Optional[int] = None,
        max_configurations_per_record: Optional[int] = None,
        atomic_numbers_to_limit: Optional[np.ndarray] = None,
        max_force: Optional[unit.Quantity] = None,
        max_force_key: Optional[str] = "forces",
        final_configuration_only: Optional[bool] = False,
        local_db_dir: Optional[str] = None,
        local_db_name: Optional[str] = None,
    ) -> Self:
        """
        Subset the dataset based on various criteria

        Parameters
        ----------
        new_dataset_name: str
            Name of the new dataset that will be returned. Cannot be the same as the current dataset name.
        total_records: Optional[int], default=None
            Maximum number of records to include in the subset. Cannot be used in conjunction with total_configurations.
        total_configurations: Optional[int], default=None
            Total number of conformers to include in the subset. annot be used in conjunction with total_records
        max_configurations_per_record: Optional[int], default=None
            Maximum number of conformers to include per record. If None, all conformers in a record will be included.
        atomic_numbers_to_limit: Optional[np.ndarray], default=None
            An array of atomic species to limit the dataset to.
            Any molecules that contain elements outside of this list will be igonored
        max_force: Optional[unit.Quantity], default=None
            If set, configurations with forces greater than this value will be removed.
        final_configuration_only: Optional[bool], default=False
            If True, only the final configuration of each record will be included in the subset.
        local_db_dir: str, optional, default=None
            Directory to store the local database for the new dataset.  If not defined, will use the same directory as the current dataset.
        local_db_name: str, optional, default=None
            Name of the cache database for the new dataset. If None, the dataset name will be used.

        Returns
        -------
            SourceDataset: A new dataset that corresponds to the desired subset.
        """
        if new_dataset_name == self.name:
            raise ValueError(
                "New dataset name cannot be the same as the current dataset name."
            )
        if total_records is not None and total_configurations is not None:
            raise ValueError(
                "Cannot set both total_records and total_conformers. Please choose one."
            )

        if (
            final_configuration_only == True
            and max_configurations_per_record is not None
        ):
            raise ValueError(
                "Cannot set final_configuration_only=True and total_conformers. Please choose one."
            )

        if total_configurations is not None:
            if total_configurations > self.total_configs():
                log.warning(
                    f"Requested number of configurations {total_configurations} is greater than the number of configurations in the dataset {self.total_configs()}."
                )
                log.warning(
                    f"Using all configurations ({self.total_configs()} in the dataset."
                )
                total_configurations = self.total_configs()

        if total_records is not None:
            if total_records > len(self.records):
                log.warning(
                    f"Requested number of records {total_records} is greater than the number of records in the dataset {len(self.records)}."
                )
                log.warning("Using all records in the dataset instead.")
                total_records = len(self.records)

        if local_db_dir is None:
            local_db_dir = self.local_db_dir
        if local_db_name is None:
            local_db_name = new_dataset_name.replace(" ", "_") + ".sqlite"

        # create a new empty dataset, we will add records that meet the criteria to this dataset
        new_dataset = SourceDataset(
            name=new_dataset_name,
            append_property=self.append_property,
            local_db_dir=local_db_dir,
            local_db_name=local_db_name,
        )

        # if we are limiting the total conformers
        if total_configurations is not None:

            total_configurations_to_add = total_configurations

            with SqliteDict(
                f"{self.local_db_dir}/{self.local_db_name}",
                autocommit=True,
            ) as db:
                for name in self.records.keys():

                    if total_configurations_to_add > 0:

                        record = db[name]

                        # if we have a max force, we will remove configurations with forces greater than the max_force
                        # we will just overwrite the record with the new record
                        if max_force is not None:
                            record = record.remove_high_force_configs(
                                max_force=max_force, force_key=max_force_key
                            )

                        # we need to set the total configs in the record AFTER we have done any force filtering
                        n_configs = record.n_configs

                        # if the record does not contain the appropriate atomic species, we will skip it
                        # and move on to the next iteration
                        if atomic_numbers_to_limit is not None:
                            if not record.contains_atomic_numbers(
                                atomic_numbers_to_limit
                            ):
                                continue

                        # if we set a max number of configurations we want per record, consider that here
                        if max_configurations_per_record is not None:
                            n_configs_to_add = max_configurations_per_record
                            # if we have fewer than the max, just set to n_configs
                            if n_configs < max_configurations_per_record:
                                n_configs_to_add = n_configs
                            # now check to see if the n_configs_to_add is more than what we still need to hit max conformers
                            if n_configs_to_add > total_configurations_to_add:
                                n_configs_to_add = total_configurations_to_add

                        # even if we don't limit the number of configurations, we may still need to limit the number we
                        # include to satisfy the total number of configurations
                        else:
                            n_configs_to_add = n_configs
                            if n_configs_to_add > total_configurations_to_add:
                                n_configs_to_add = total_configurations_to_add

                        if final_configuration_only:
                            record = record.slice_record(
                                record.n_configs - 1, record.n_configs
                            )
                            new_dataset.add_record(record)
                            total_configurations_to_add -= 1

                        else:
                            record = record.slice_record(0, n_configs_to_add)
                            new_dataset.add_record(record)
                            total_configurations_to_add -= n_configs_to_add
                return new_dataset

        elif total_records is not None:
            with SqliteDict(
                f"{self.local_db_dir}/{self.local_db_name}",
                autocommit=True,
            ) as db:
                total_records_to_add = total_records
                for name in self.records.keys():
                    if total_records_to_add > 0:
                        record = db[name]
                        # if we have a max force, we will remove configurations with forces greater than the max_force
                        # we will just overwrite the record with the new record
                        if max_force is not None:
                            record = record.remove_high_force_configs(
                                max_force=max_force, force_key=max_force_key
                            )
                        # if the record does not contain the appropriate atomic species, we will skip it
                        # and move on to the next iteration
                        if atomic_numbers_to_limit is not None:
                            if not record.contains_atomic_numbers(
                                atomic_numbers_to_limit
                            ):
                                continue
                        # if we have a max number of configurations we want per record, consider that here
                        if max_configurations_per_record is not None:
                            n_to_add = min(
                                max_configurations_per_record, record.n_configs
                            )

                            record = record.slice_record(0, n_to_add)

                        if final_configuration_only:
                            record = record.slice_record(
                                record.n_configs - 1, record.n_configs
                            )

                        new_dataset.add_record(record)
                        total_records_to_add -= 1
                return new_dataset
        # if we are not going to be limiting the total number of configurations or records
        else:
            with SqliteDict(
                f"{self.local_db_dir}/{self.local_db_name}",
                autocommit=True,
            ) as db:
                for name in self.records.keys():
                    record = db[name]
                    # if we have a max force, we will remove configurations with forces greater than the max_force
                    # we will just overwrite the record with the new record
                    if max_force is not None:
                        record = record.remove_high_force_configs(
                            max_force=max_force, force_key=max_force_key
                        )
                    # if the record does not contain the appropriate atomic species, we will skip it
                    # and move on to the next iteration
                    if atomic_numbers_to_limit is not None:
                        if not record.contains_atomic_numbers(atomic_numbers_to_limit):
                            continue
                    if max_configurations_per_record is not None:
                        n_to_add = min(max_configurations_per_record, record.n_configs)
                        record = record.slice_record(0, n_to_add)
                    if final_configuration_only:
                        record = record.slice_record(
                            record.n_configs - 1, record.n_configs
                        )

                    new_dataset.add_record(record)
                return new_dataset

    def validate_record(self, name: str):
        """
        Validate a record to ensure that the number of atoms and configurations are consistent across all properties.

        Issues are reported in the errors log, but no exceptions are raised.

        Parameters
        ----------
        name: str
            Name of the record to validate

        Returns
        -------
            bool: True if the record validated, False otherwise.
        """

        validation_status = True
        # every record should have atomic numbers, positions, and energies
        # make sure atomic_numbers have been set
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            record = db[name]
            if record.atomic_numbers is None:
                validation_status = False
                log.error(
                    f"No atomic numbers set for record {name}. These are required."
                )
                # raise ValueError(
                #     f"No atomic numbers set for record {name}. These are required."
                # )

            # ensure we added positions and energies as these are the minimum requirements for a dataset along with
            # atomic_numbers
            positions_in_properties = False
            for property in record.per_atom.keys():
                if isinstance(record.per_atom[property], Positions):
                    positions_in_properties = True
                    break
            if positions_in_properties == False:
                validation_status = False
                log.error(
                    f"No positions found in properties for record {name}. These are required."
                )
                # raise ValueError(
                #     f"No positions found in properties for record {name}. These are required."
                # )

            # we need to ensure we have some type of energy defined
            energy_in_properties = False
            for property in record.per_system.keys():
                if isinstance(record.per_system[property], Energies):
                    energy_in_properties = True
                    break

            if energy_in_properties == False:
                validation_status = False
                log.error(
                    f"No energies found in properties for record {name}. These are required."
                )
                # raise ValueError(
                #     f"No energies found in properties for record {name}. These are required."
                # )

            # run record validation for number of atoms
            # this will check that all per_atom properties have the same number of atoms as the atomic numbers
            if record._validate_n_atoms() == False:
                validation_status = False
                log.error(
                    f"Number of atoms for properties in record {name} are not consistent."
                )
                # raise ValueError(
                #     f"Number of atoms for properties in record {name} are not consistent."
                # )
            # run record validation for number of configurations
            # this will check that all properties have the same number of configurations
            if record._validate_n_configs() == False:
                validation_status = False
                log.error(
                    f"Number of configurations for properties in record {name} are not consistent."
                )
                # raise ValueError(
                #     f"Number of configurations for properties in record {name} are not consistent."
                # )

            # check that the units provided are compatible with the expected units for the property type
            # e.g., ensure things that should be length have units of length.
            for property in record.per_atom.keys():

                property_record = record.per_atom[property]

                property_type = property_record.property_type
                # first double check that this did indeed get pushed to the correct sub dictionary
                assert property_record.classification == PropertyClassification.per_atom

                property_units = property_record.units
                expected_units = GlobalUnitSystem.get_units(property_type)
                # check to make sure units are compatible with the expected units for the property type
                if not expected_units.is_compatible_with(property_units, "chem"):
                    validation_status = False
                    log.error(
                        f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {name}."
                    )
                    # raise ValueError(
                    #     f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {name}."
                    # )

            for property in record.per_system.keys():
                property_record = record.per_system[property]

                # check that the number of atoms in the property matches the number of atoms in the atomic numbers
                property_type = property_record.property_type

                assert (
                    property_record.classification == PropertyClassification.per_system
                )
                expected_units = GlobalUnitSystem.get_units(property_type)
                property_units = property_record.units
                if not expected_units.is_compatible_with(property_units, "chem"):
                    validation_status = False
                    log.error(
                        f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {name}."
                    )

        return validation_status

    def print_record(self, name):
        """
        Print a record to the console.

        Parameters
        ----------
        name: str
            Name of the record to print.

        Returns
        -------

        """
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            record = db[name]
            print(record)

    def validate_records(self):
        """
        Validate records to ensure that the number of atoms and configurations are consistent across all properties.

        Issues are reported in the errors log, but no exceptions are raised.

        Parameters
        ----------

        Returns
        -------
            bool: True if all the records are validated, False otherwise.
        """
        # check that all properties in a record have the same number configs and number of atoms (where applicable)
        from tqdm import tqdm

        validation_status = []
        for name in tqdm(self.records.keys()):
            validation_status.append(self.validate_record(name))

        if all(validation_status):
            log.info("All records validated successfully.")
            return True
        else:
            sum_failures = sum([1 for status in validation_status if status == False])
            log.error(
                f"{sum_failures} record(s) failed validation. See error logs for more information."
            )
            return False

    def generate_dataset_summary(self, checksum: str = None, file_name: str = None):
        """
        Generate a summary of the dataset.

        Parameters
        ----------
        checksum: str, optional, default=None
            MD5 checksum of the dataset file.
        file_name: str, optional, default=None
            Name of the dataset file.


        Returns
        -------
        Dict
            summary of the dataset
        """
        output_dict = {}
        output_dict["name"] = self.name
        if checksum is not None:
            output_dict["md5_checksum"] = checksum
        if file_name is not None:
            output_dict["filename"] = file_name
        output_dict["total_records"] = self.total_records()
        output_dict["total_configurations"] = self.total_configs()

        key = list(self.records.keys())[0]
        with SqliteDict(
            f"{self.local_db_dir}/{self.local_db_name}",
            autocommit=True,
        ) as db:
            record = db[key]

            temp_props = {}
            temp_props["atomic_numbers"] = {
                "classification": str(record.atomic_numbers.classification),
            }

            for prop in record.per_atom.keys():
                temp_props[prop] = {
                    "classification": str(record.per_atom[prop].classification),
                    "units": str(
                        GlobalUnitSystem.get_units(record.per_atom[prop].property_type)
                    ),
                }

            for prop in record.per_system.keys():
                temp_props[prop] = {
                    "classification": str(record.per_system[prop].classification),
                    "units": str(
                        GlobalUnitSystem.get_units(
                            record.per_system[prop].property_type
                        )
                    ),
                }

            for prop in record.meta_data.keys():
                temp_props[prop] = {"classification": "meta_data"}

            output_dict["properties"] = temp_props

        return output_dict

    def summary_to_json(
        self, file_path: str, file_name: str, hdf5_checksum: str, hdf5_file_name: str
    ):
        """
        Write the dataset summary to a json file.

        Parameters
        ----------
        file_path: str, required
            Path to the directory where the json file will be saved.
        file_name: str, required
            Name of the json file to be saved.
        """
        import json

        dataset_summary = self.generate_dataset_summary(hdf5_checksum, hdf5_file_name)

        with open(f"{file_path}/{file_name}", "w") as f:
            json.dump(dataset_summary, f, indent=4)

    def _remove_local_db(self):
        """
        Remove the local database file.

        Returns
        -------

        """
        os.remove(f"{self.local_db_dir}/{self.local_db_name}")

    def to_hdf5(self, file_path: str, file_name: str):
        """
        Write the dataset to an HDF5 file.

        Parameters
        ----------
        file_path: str
            Path where the file should be written.
        file_name: str
            Name of the file to write. Must end in .hdf5

        Returns
        -------
        chksum: str
            MD5 checksum of the hdf5_ file.
        """
        import h5py
        import os

        import h5py
        from tqdm import tqdm
        import numpy as np

        assert file_name.endswith(".hdf5")
        file_path = os.path.expanduser(file_path)
        os.makedirs(file_path, exist_ok=True)

        full_file_path = f"{file_path}/{file_name}"
        dt = h5py.special_dtype(vlen=str)

        # validate entries
        log.info("Validating records")
        self.validate_records()

        log.info(f"Writing records to {full_file_path}")
        from modelforge.utils.misc import OpenWithLock

        with OpenWithLock(f"{full_file_path}.lockfile", "w") as lockfile:
            with h5py.File(full_file_path, "w") as f:
                with SqliteDict(
                    f"{self.local_db_dir}/{self.local_db_name}",
                ) as db:
                    for name in tqdm(self.records.keys()):
                        record_group = f.create_group(name)

                        record = db[name]

                        record_group.create_dataset(
                            "atomic_numbers",
                            data=record.atomic_numbers.value,
                            shape=record.atomic_numbers.value.shape,
                        )
                        record_group["atomic_numbers"].attrs["format"] = str(
                            record.atomic_numbers.classification
                        )
                        record_group["atomic_numbers"].attrs["property_type"] = str(
                            record.atomic_numbers.property_type
                        )

                        record_group.create_dataset("n_configs", data=record.n_configs)
                        record_group["n_configs"].attrs["format"] = "meta_data"

                        for key, property in record.per_atom.items():

                            target_units = GlobalUnitSystem.get_units(
                                property.property_type
                            )
                            record_group.create_dataset(
                                key,
                                data=unit.Quantity(property.value, property.units)
                                .to(target_units, "chem")
                                .magnitude,
                                shape=property.value.shape,
                            )
                            record_group[key].attrs["u"] = str(target_units)
                            record_group[key].attrs["format"] = str(
                                property.classification
                            )
                            record_group[key].attrs["property_type"] = str(
                                property.property_type
                            )

                        for key, property in record.per_system.items():
                            target_units = GlobalUnitSystem.get_units(
                                property.property_type
                            )
                            record_group.create_dataset(
                                key,
                                data=unit.Quantity(property.value, property.units)
                                .to(target_units, "chem")
                                .magnitude,
                                shape=property.value.shape,
                            )

                            record_group[key].attrs["u"] = str(target_units)
                            record_group[key].attrs["format"] = str(
                                property.classification
                            )
                            record_group[key].attrs["property_type"] = str(
                                property.property_type
                            )

                        for key, property in record.meta_data.items():
                            if isinstance(property.value, str):
                                record_group.create_dataset(
                                    key,
                                    data=property.value,
                                    dtype=dt,
                                )
                                record_group[key].attrs["u"] = str(property.units)
                                record_group[key].attrs["format"] = str(
                                    property.classification
                                )
                                record_group[key].attrs["property_type"] = str(
                                    property.property_type
                                )

                            elif isinstance(property.value, (float, int)):

                                record_group.create_dataset(
                                    key,
                                    data=property.value,
                                )
                                record_group[key].attrs["u"] = str(property.units)
                                record_group[key].attrs["format"] = str(
                                    property.classification
                                )
                                record_group[key].attrs["property_type"] = str(
                                    property.property_type
                                )

                            elif isinstance(property.value, np.ndarray):

                                record_group.create_dataset(
                                    key, data=property.value, shape=property.value.shape
                                )
                                record_group[key].attrs["u"] = str(property.units)
                                record_group[key].attrs["format"] = str(
                                    property.classification
                                )
                                record_group[key].attrs["property_type"] = str(
                                    property.property_type
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported type ({type(property.value)}) for metadata {key}"
                                )
        from modelforge.utils.remote import calculate_md5_checksum

        hdf5_checksum = calculate_md5_checksum(file_path=file_path, file_name=file_name)

        return hdf5_checksum

    def convert_to_global_unit_system(self):
        """Convert all properties in the dataset to the global unit system.

        Note metadata will not be converted.
        """

        for record_name in self.records.keys():
            with SqliteDict(
                f"{self.local_db_dir}/{self.local_db_name}",
                autocommit=True,
            ) as db:
                record = db[record_name]
                record.convert_to_global_unit_system()
                db[record_name] = record


def create_dataset_from_hdf5(
    hdf5_filename: str,
    dataset_name: str,
    dataset_local_db_dir: str = "./",
    dataset_local_db_name: str = None,
    append_property: bool = False,
):
    """
    Create a dataset from an HDF5 file.

    Parameters
    ----------
    hdf5_filename: str
        Name of the HDF5 file to read.
    dataset_name: str
        Name of the dataset to create.
    dataset_local_db_dir: str, optional, default="./"
        Directory to store the local database.
    dataset_local_db_name: str, optional, default=None
        Name of the local database.
    append_property: bool, optional, default=False
        Set to True to append properties to existing properties in a record.
        If False, an error will be raised if a property with the same name is added to a record.

    Returns
    -------
    SourceDataset
        Instance of the SourceDataset class.
    """
    import h5py
    from tqdm import tqdm
    from modelforge.curate.properties import PropertyBaseModel

    dataset = SourceDataset(
        name=dataset_name,
        local_db_dir=dataset_local_db_dir,
        local_db_name=dataset_local_db_name,
        append_property=append_property,
    )

    with h5py.File(hdf5_filename, "r") as f:
        keys = list(f.keys())

        for key in tqdm(keys):

            record = Record(name=key)
            n_configs = 0
            properties_keys = f[key].keys()
            for pk in properties_keys:
                if pk == "n_configs":
                    n_configs = f[key][pk][()]
                else:
                    property_type = f[key][pk].attrs["property_type"]
                    property_classification = f[key][pk].attrs["format"]
                    if "u" in f[key][pk].attrs.keys():
                        unit_str = f[key][pk].attrs["u"]
                    else:
                        unit_str = "dimensionless"
                    value = f[key][pk][()]
                    if type(value) is bytes:
                        value = f[key][pk][()].decode("utf-8")

                    property = PropertyBaseModel(
                        name=pk,
                        value=value,
                        units=unit_str,
                        property_type=property_type,
                        classification=property_classification,
                    )

                    record.add_property(property)
            assert n_configs == record.n_configs
            dataset.add_record(record)

        return dataset

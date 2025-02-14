from modelforge.curate.units import GlobalUnitSystem, chem_context
from modelforge.curate.properties import (
    PropertyBaseModel,
    PropertyClassification,
    PropertyType,
    Positions,
    Energies,
    AtomicNumbers,
)

from openff.units import unit

import numpy as np
import copy

from typing import Union, List, Type, Optional

from typing_extensions import Self

from loguru import logger as log


class Record:
    def __init__(self, name: str, append_property: bool = False):
        assert isinstance(name, str)
        self.name = name
        self.per_atom = {}
        self.per_system = {}
        self.meta_data = {}
        self.atomic_numbers = None
        self._n_atoms = -1
        self._n_configs = -1
        self.append_property = append_property

    def __repr__(self):

        output_string = f"name: {self.name}\n"
        if self.n_atoms == -1:
            output_string += f"* n_atoms: cannot be determined, see warnings log\n"
        else:
            output_string += f"* n_atoms: {self.n_atoms}\n"
        if self.n_configs == -1:
            output_string += f"* n_configs: cannot be determined, see warnings log\n"
        else:
            output_string += f"* n_configs: {self.n_configs}\n"
        output_string += "* atomic_numbers:\n"
        output_string += f" -  {self.atomic_numbers}\n"
        output_string += f"* per-atom properties: ({list(self.per_atom.keys())}):\n"
        for key, value in self.per_atom.items():
            output_string += f" -  {value}\n"
        output_string += f"* per-system properties: ({list(self.per_system.keys())}):\n"
        for key, value in self.per_system.items():
            output_string += f" -  {value}\n"
        output_string += f"* meta_data: ({list(self.meta_data.keys())})\n"
        for key, value in self.meta_data.items():
            output_string += f" -  {value}\n"
        return output_string

    @property
    def n_atoms(self):
        """
        Get the number of atoms in the record

        Returns
        -------
            int: number of atoms in the record
        """
        # the validate function will set self._n_atoms to -1 if the number of atoms cannot be determined
        # or if the number of atoms is inconsistent between properties
        # otherwise will set it to the value from the atomic_numbers
        self._validate_n_atoms()
        return self._n_atoms

    @property
    def n_configs(self):
        """
        Get the number of configurations in the record

        Returns
        -------
            int: number of configurations in the record
        """

        # the validate function will set self._n_configs to -1 if the number of configurations cannot be determined
        # or if the number of configurations is inconsistent between properties.
        # will set it to the value from the properties otherwise
        self._validate_n_configs()
        return self._n_configs

    def slice_record(self, min: int = 0, max: int = -1) -> Self:
        """
        Slice the record to only include a subset of configs

        Slicing occurs on all per_atom and per_system properties

        Parameters
        ----------
        min: int
            Starting index for slicing.
        max: int
            ending index for slicing.

        Returns
        -------
            Record: Slice record.
        """
        new_record = copy.deepcopy(self)
        for key, value in self.per_atom.items():
            new_record.per_atom[key].value = value.value[min:max]
        for key, value in self.per_system.items():
            new_record.per_system[key].value = value.value[min:max]

        return new_record

    def to_dict(self):
        """
        Convert the record to a dictionary

        Returns
        -------
            dict: dictionary representation of the record
        """
        return {
            "name": self.name,
            "n_atoms": self.n_atoms,
            "n_configs": self.n_configs,
            "atomic_numbers": self.atomic_numbers,
            "per_atom": self.per_atom,
            "per_system": self.per_system,
            "meta_data": self.meta_data,
        }

    def _validate_n_atoms(self):
        """
        Validate the number of atoms in the record by checking that all per_atom properties have the same number of atoms as the atomic numbers.

        Returns
        -------
            bool: True if the number of atoms is defined and consistent, False otherwise.
        """
        self._n_atoms = -1
        if self.atomic_numbers is not None:
            for key, value in self.per_atom.items():
                if value.n_atoms != self.atomic_numbers.n_atoms:
                    log.warning(
                        f"Number of atoms for property {key} in record {self.name} does not match the number of atoms in the atomic numbers."
                    )
                    return False
        else:
            log.warning(
                f"No atomic numbers set for record {self.name}. Cannot validate number of atoms."
            )
            return False
        self._n_atoms = self.atomic_numbers.n_atoms
        return True

    def _validate_n_configs(self):
        """
        Validate the number of configurations in the record by checking that all properties have the same number of configurations.

        Returns
        -------
            bool: True if the number of configurations is defined and consistent, False otherwise.
        """
        n_configs = []
        for key, value in self.per_atom.items():
            n_configs.append(value.n_configs)
        for key, value in self.per_system.items():
            n_configs.append(value.n_configs)
        if len(n_configs) != 0:
            if all([n == n_configs[0] for n in n_configs]):
                self._n_configs = n_configs[0]
                return True
            else:
                self._n_configs = -1
                log.warning(
                    f"Number of configurations for properties in record {self.name} are not consistent."
                )
                for key, value in self.per_atom.items():
                    log.warning(f" - {key} : {value.n_configs}")
                for key, value in self.per_system.items():
                    log.warning(f" - {key} : {value.n_configs}")
                return False
        else:
            log.warning(
                f"No properties found in record {self.name}. Cannot determine the number of configurations."
            )
            self._n_configs = -1
            return False

    def validate(self):
        """
        Validate the record to ensure that the number of atoms and configurations are consistent across all properties.

        Returns
        -------
            True if the record validated, False otherwise.
        """
        if self._validate_n_atoms() and self._validate_n_configs():
            return True
        return False

    def add_properties(self, properties: List[Type[PropertyBaseModel]]):
        """
        Add a list of properties to the record.

        Parameters
        ----------
        properties: List[Type[PropertyBaseModel]]
            List of properties to add to the record.
        Returns
        -------

        """
        for property in properties:
            self.add_property(property)

    def add_property(self, property: Type[PropertyBaseModel]):
        """
        Add a property to the record.

        Parameters
        ----------
        property: Type[PropertyBaseModel]
            Property to add to the record.
        Returns
        -------

        """
        if property.classification == PropertyClassification.atomic_numbers:
            # we will not allow atomic numbers to be set twice
            if self.atomic_numbers is not None:
                raise ValueError(f"Atomic numbers already set for record {self.name}")

            self.atomic_numbers = property.model_copy(deep=True)

            # Note, the number of atoms will always be set by the atomic_numbers property.
            # We will later validate that per_atom properties are consistent with this value later
            # since we are not enforcing that atomic_numbers need to be set before any other property

        elif property.classification == PropertyClassification.meta_data:
            if property.name in self.meta_data.keys():
                log.warning(
                    f"Metadata with name {property.name} already exists in the record {self.name}."
                )
                raise ValueError(
                    f"Metadata with name {property.name} already exists in the record {self.name}"
                )

            elif property.name in self.per_atom.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_atom property."
                )
            elif property.name in self.per_system.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_system property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them, not the MetaData class."
                )
            self.meta_data[property.name] = property.model_copy(deep=True)

        elif property.classification == PropertyClassification.per_atom:
            if property.name in self.per_system.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_system property."
                )
            elif property.name in self.meta_data.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a meta_data property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
                )
            elif property.name in self.per_atom.keys():
                if self.append_property == False:
                    error_msg = f"Property with name {property.name} already exists in the record {self.name}."
                    error_msg += (
                        f"Set append_property=True to append to the existing property."
                    )
                    raise ValueError(error_msg)
                # if the property already exists, we will use vstack to add it to the existing array
                # after first checking that the dimensions are consistent
                # note we do not check shape[0], as that corresponds to the number of configurations
                assert (
                    self.per_atom[property.name].value.shape[1]
                    == property.value.shape[1]
                ), f"{self.name}: n_atoms of {property.name} does not: {property.value.shape[1]} != {self.per_atom[property.name].value.shape[1]}."
                assert (
                    self.per_atom[property.name].value.shape[2]
                    == property.value.shape[2]
                )
                # In order to append to the array, everything needs to have the same units
                # We will use the units of the first property that was added

                temp_array = property.value
                if property.units != self.per_atom[property.name].units:
                    temp_array = (
                        unit.Quantity(property.value, property.units)
                        .to(
                            self.per_atom[property.name].units,
                            "chem",
                        )
                        .magnitude
                    )
                self.per_atom[property.name].value = np.vstack(
                    (
                        self.per_atom[property.name].value,
                        temp_array,
                    )
                )

            else:
                self.per_atom[property.name] = property.model_copy(deep=True)
        elif property.classification == PropertyClassification.per_system:
            if property.name in self.per_atom.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_atom property."
                )
            elif property.name in self.meta_data.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a meta_data property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
                )
            elif property.name in self.per_system.keys():
                if self.append_property == False:
                    error_msg = f"Property with name {property.name} already exists in the record {self.name}."
                    error_msg += (
                        f"Set append_property=True to append to the existing property."
                    )
                    raise ValueError(error_msg)

                assert (
                    self.per_system[property.name].value.shape[1]
                    == property.value.shape[1]
                )
                temp_array = property.value
                if property.units != self.per_system[property.name].units:
                    temp_array = (
                        unit.Quantity(property.value, property.units)
                        .to(
                            self.per_system[property.name].units,
                            "chem",
                        )
                        .magnitude
                    )

                self.per_system[property.name].value = np.vstack(
                    (
                        self.per_system[property.name].value,
                        temp_array,
                    )
                )
            else:
                self.per_system[property.name] = property.model_copy(deep=True)


class SourceDataset:
    def __init__(
        self,
        dataset_name: str,
        append_property: bool = False,
    ):
        """
        Class to hold a dataset of properties for a given dataset name

        Parameters
        ----------
        dataset_name: str
            Name of the dataset
        append_property: bool, optional, default=False
            If True, append an array to existing array if a property with the same name is added multiple times to a record.
            If False, an error will be raised if trying to add a property with the same name already exists in a record
            Use True if data for configurations are stored in separate files/database entries and you want to combine them.
        """

        self.dataset_name = dataset_name
        self.records = {}
        self.append_property = append_property

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
        for record in self.records.values():
            total_config += record.n_configs
        return total_config

    def create_record(
        self,
        record_name: str,
        properties: Optional[List[Type[PropertyBaseModel]]] = None,
    ):
        """
        Create a record in the dataset. If properties are provided, they will be added to the record.

        Parameters
        ----------
        record_name: str
            Name of the record/
        properties: List[Type[PropertyBaseModel]], optional, default=None
            List of properties to add to the record. If not provided, an empty record will be created.

        Returns
        -------

        """
        assert isinstance(record_name, str)

        # I think this should error out if we've already encountered a name, as that would imply
        # some issue with the dataset construction
        if record_name in self.records.keys():
            raise ValueError(
                f"Record with name {record_name} already exists in the dataset"
            )

        self.records[record_name] = Record(record_name, self.append_property)
        if properties is not None:
            self.add_properties_to_record(record_name, properties)

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
        if record.name in self.records.keys():
            log.warning(
                f"Record with name {record.name} already exists in the dataset."
            )
            raise ValueError(
                f"Record with name {record.name} already exists in the dataset."
            )

        self.records[record.name] = copy.deepcopy(record)

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

        self.records[record.name] = copy.deepcopy(record)

    def remove_record(self, record_name: str):
        """
        Remove a record from the dataset.

        Parameters
        ----------
        record_name: str
            Name of the record to remove.

        Returns
        -------

        """
        assert isinstance(record_name, str)
        if record_name in self.records.keys():
            self.records.pop(record_name)
        else:
            log.warning(
                f"Record with name {record_name} does not exist in the dataset."
            )

    def add_properties(
        self, record_name: str, properties: List[Type[PropertyBaseModel]]
    ):
        """
        Add a list of properties to a record in the dataset.

        Parameters
        ----------
        record_name: str
            Name of the record to add the properties to.
        properties: List[Type[PropertyBaseModel]]
            List of properties to add to the record.

        Returns
        -------

        """

        for property in properties:
            self.add_property(record_name, property)

    def add_property(self, record_name: str, property: Type[PropertyBaseModel]):
        """
        Add a property to a record in the dataset.

        Parameters
        ----------
        record_name: str
            Name of the record to add the property to.
        property: Type[PropertyBaseModel]
            Property to add to the record.

        Returns
        -------

        """
        assert isinstance(record_name, str)
        # check if the record exists; if it does not add it
        if record_name not in self.records.keys():
            log.info(
                f"Record with name {record_name} does not exist in the dataset. Creating it now."
            )
            self.create_record(record_name)

        self.records[record_name].add_property(property)

    def get_record(self, record_name: str):
        """
        Get a record from the dataset. Returns an instance of the Record class.

        Parameters
        ----------
        record_name: str
            Name of the record to get
        Returns
        -------
            Record: instance of the Record class corresponding to the record name

        """
        assert isinstance(record_name, str)
        from copy import deepcopy

        return deepcopy(self.records[record_name])

    def slice_record(self, record_name: str, min: int = 0, max: int = -1) -> Record:
        """
        Slice a record to only include a subset of configs

        Slicing occurs on all per_atom and per_system properties

        Parameters
        ----------
        record_name: str
            Name of the record to slice.
        min: int
            Starting index for slicing.
        max: int
            Ending index for slicing.

        Returns
        -------
            Record: A copy of the sliced record.
        """
        assert isinstance(record_name, str)

        return self.records[record_name].slice_record(min=min, max=max)

    def validate_record(self, record_name: str):
        """
        Validate a record to ensure that the number of atoms and configurations are consistent across all properties.

        Issues are reported in the errors log, but no exceptions are raised.

        Parameters
        ----------
        record_name: str
            Name of the record to validate

        Returns
        -------
            bool: True if the record validated, False otherwise.
        """

        validation_status = True
        # every record should have atomic numbers, positions, and energies
        # make sure atomic_numbers have been set
        if self.records[record_name].atomic_numbers is None:
            validation_status = False
            log.error(
                f"No atomic numbers set for record {record_name}. These are required."
            )
            # raise ValueError(
            #     f"No atomic numbers set for record {record_name}. These are required."
            # )

        # ensure we added positions and energies as these are the minimum requirements for a dataset along with
        # atomic_numbers
        positions_in_properties = False
        for property in self.records[record_name].per_atom.keys():
            if isinstance(self.records[record_name].per_atom[property], Positions):
                positions_in_properties = True
                break
        if positions_in_properties == False:
            validation_status = False
            log.error(
                f"No positions found in properties for record {record_name}. These are required."
            )
            # raise ValueError(
            #     f"No positions found in properties for record {record_name}. These are required."
            # )

        # we need to ensure we have some type of energy defined
        energy_in_properties = False
        for property in self.records[record_name].per_system.keys():
            if isinstance(self.records[record_name].per_system[property], Energies):
                energy_in_properties = True
                break

        if energy_in_properties == False:
            validation_status = False
            log.error(
                f"No energies found in properties for record {record_name}. These are required."
            )
            # raise ValueError(
            #     f"No energies found in properties for record {record_name}. These are required."
            # )

        # run record validation for number of atoms
        # this will check that all per_atom properties have the same number of atoms as the atomic numbers
        if self.records[record_name]._validate_n_atoms() == False:
            validation_status = False
            log.error(
                f"Number of atoms for properties in record {record_name} are not consistent."
            )
            # raise ValueError(
            #     f"Number of atoms for properties in record {record_name} are not consistent."
            # )
        # run record validation for number of configurations
        # this will check that all properties have the same number of configurations
        if self.records[record_name]._validate_n_configs() == False:
            validation_status = False
            log.error(
                f"Number of configurations for properties in record {record_name} are not consistent."
            )
            # raise ValueError(
            #     f"Number of configurations for properties in record {record_name} are not consistent."
            # )

        # check that the units provided are compatible with the expected units for the property type
        # e.g., ensure things that should be length have units of length.
        for property in self.records[record_name].per_atom.keys():

            property_record = self.records[record_name].per_atom[property]

            property_type = property_record.property_type
            # first double check that this did indeed get pushed to the correct sub dictionary
            assert property_record.classification == PropertyClassification.per_atom

            property_units = property_record.units
            expected_units = GlobalUnitSystem.get_units(property_type)
            # check to make sure units are compatible with the expected units for the property type
            if not expected_units.is_compatible_with(property_units, "chem"):
                validation_status = False
                log.error(
                    f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {record_name}."
                )
                # raise ValueError(
                #     f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {record_name}."
                # )

        for property in self.records[record_name].per_system.keys():
            property_record = self.records[record_name].per_system[property]

            # check that the number of atoms in the property matches the number of atoms in the atomic numbers
            property_type = property_record.property_type

            assert property_record.classification == PropertyClassification.per_system
            expected_units = GlobalUnitSystem.get_units(property_type)
            property_units = property_record.units
            if not expected_units.is_compatible_with(property_units, "chem"):
                validation_status = False
                log.error(
                    f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {record_name}."
                )
                # raise ValueError(
                #     f"Unit of {property_record.name} is not compatible with the expected unit {expected_units} for record {record_name}."
                # )
        return validation_status

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
        for record_name in tqdm(self.records.keys()):
            validation_status.append(self.validate_record(record_name))

        if all(validation_status):
            log.info("All records validated successfully.")
            return True
        else:
            sum_failures = sum([1 for status in validation_status if status == False])
            log.error(
                f"{sum_failures} record(s) failed validation. See error logs for more information."
            )
            return False

    def _generate_dataset_summary(self, checksum: str, file_name: str):
        """
        Generate a summary of the dataset.

        Returns
        -------
        Dict
            summary of the dataset
        """
        output_dict = {}
        output_dict["dataset_name"] = self.dataset_name
        output_dict["md5_checksum"] = checksum
        output_dict["filename"] = file_name
        output_dict["total_records"] = self.total_records()
        output_dict["total_configurations"] = self.total_configs()

        key = list(self.records.keys())[0]

        temp_props = {}
        temp_props["atomic_numbers"] = {
            "classification": str(self.records[key].atomic_numbers.classification),
        }

        for prop in self.records[key].per_atom.keys():
            temp_props[prop] = {
                "classification": str(self.records[key].per_atom[prop].classification),
                "units": str(
                    GlobalUnitSystem.get_units(
                        self.records[key].per_atom[prop].property_type
                    )
                ),
            }

        for prop in self.records[key].per_system.keys():
            temp_props[prop] = {
                "classification": str(
                    self.records[key].per_system[prop].classification
                ),
                "units": str(
                    GlobalUnitSystem.get_units(
                        self.records[key].per_system[prop].property_type
                    )
                ),
            }

        for prop in self.records[key].meta_data.keys():
            temp_props[prop] = "meta_data"

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

        dataset_summary = self._generate_dataset_summary(hdf5_checksum, hdf5_file_name)

        with open(f"{file_path}/{file_name}", "w") as f:
            json.dump(dataset_summary, f, indent=4)

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
                for record in tqdm(self.records.keys()):
                    record_group = f.create_group(record)

                    record_group.create_dataset(
                        "atomic_numbers",
                        data=self.records[record].atomic_numbers.value,
                        shape=self.records[record].atomic_numbers.value.shape,
                    )
                    record_group["atomic_numbers"].attrs["format"] = str(
                        self.records[record].atomic_numbers.classification
                    )
                    record_group["atomic_numbers"].attrs["property_type"] = str(
                        self.records[record].atomic_numbers.property_type
                    )

                    record_group.create_dataset(
                        "n_configs", data=self.records[record].n_configs
                    )
                    record_group["n_configs"].attrs["format"] = "meta_data"

                    for key, property in self.records[record].per_atom.items():

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
                        record_group[key].attrs["format"] = str(property.classification)
                        record_group[key].attrs["property_type"] = str(
                            property.property_type
                        )

                    for key, property in self.records[record].per_system.items():
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
                        record_group[key].attrs["format"] = str(property.classification)
                        record_group[key].attrs["property_type"] = str(
                            property.property_type
                        )

                    for key, property in self.records[record].meta_data.items():
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

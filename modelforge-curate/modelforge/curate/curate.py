import copy

from modelforge.curate.utils import (
    _convert_unit_str_to_unit_unit,
    _convert_list_to_ndarray,
    _convert_float_to_ndarray,
    chem_context,
    NdArray,
)
from openff.units import unit

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    field_validator,
    computed_field,
)
from enum import Enum
from typing import Union, List, Type, Optional

from typing_extensions import Self

from loguru import logger as log


# Define a custom config for the BaseModel to avoid extra duplication of code
class CurateBase(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
    )


# Define a custom enum class that is case insensitive that we can inherit from
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class PropertyClassification(CaseInsensitiveEnum):
    """
    Enum class to classify a property to be able to interpret how to parse the shape

    per_atom: properties have shape [n_configs, n_atoms, -1]
    per_system: properties have shape [n_configs, 1, -1]
    atomic_numbers: special case with shape of [n_atoms, 1].
                    Atomic numbers do not change as configuration changes and thus this reduces memory footprint.
    meta_data: A single entry that may be a string, float, int, or array of any shape.
                    In general, meta_data is generally ignored when reading in a dataset for training.

    """

    per_atom = "per_atom"
    per_system = "per_system"
    atomic_numbers = "atomic_numbers"
    meta_data = "meta_data"


class PropertyType(str, Enum):
    """
    Enum class that enables us to know the type of property, e.g., force, energy, charge, etc.

    This is used for validating and converting units.
    Those things classified as "other" will require manual conversion.
    """

    length = "length"
    force = "force"
    energy = "energy"
    charge = "charge"
    dipole_moment = "dipole_moment"
    quadrupole_moment = "quadrupole_moment"
    polarizability = "polarizability"
    atomic_numbers = "atomic_numbers"
    meta_data = "meta_data"


class UnitSystem:

    def __init__(self, name: str = "default"):
        self.unit_system_name = name
        self.length = unit.nanometer
        self.force = unit.kilojoule_per_mole / unit.nanometer
        self.energy = unit.kilojoule_per_mole
        self.charge = unit.elementary_charge
        self.dipole_moment = unit.elementary_charge * unit.nanometer
        self.quadrupole_moment = unit.elementary_charge * unit.nanometer**2
        self.polarizability = unit.nanometer**3
        self.atomic_numbers = unit.dimensionless
        self.dimensionless = unit.dimensionless

    def add_property_type(self, property_name: str, unit: unit.Unit):
        setattr(self, property_name, unit)

    def __repr__(self):

        return "\n".join([f"{key} : {value}" for key, value in self.__dict__.items()])

    def __getitem__(self, item):
        return getattr(self, item)


class RecordProperty(CurateBase):
    name: str
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification
    property_type: Union[PropertyType, str]

    # if units are passed as spring, convert to unit.Unit object
    converted_units = field_validator("units", mode="before")(
        _convert_unit_str_to_unit_unit
    )
    converted_array = field_validator("value", mode="before")(_convert_list_to_ndarray)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:

        if self.classification == PropertyClassification.atomic_numbers:
            if len(self.value.shape) != 2:
                raise ValueError(
                    f"Shape of '{self.name}' should be [n_atoms,1], found {len(self.value.shape)}"
                )
            if self.value.shape[1] != 1:
                raise ValueError(
                    f"Shape of '{self.name}' should be [n_atoms,1], found {len(self.value.shape)}"
                )
        elif self.classification == PropertyClassification.per_system:
            # shape of a per_system property should be 2d for most properties, but it is possible to have a 3d shape if the property is a tensor.
            if len(self.value.shape) < 2:
                raise ValueError(
                    f"Shape of property '{self.name}' should have at least 2 dimensions (per_system), found {len(self.value.shape)}"
                )

        elif self.classification == PropertyClassification.per_atom:
            # shape of a per_atom property must be at least, 3d [n_configs, n_atoms, 1], but a property could be a tensor and have more dimensions.
            if len(self.value.shape) < 3:
                raise ValueError(
                    f"Shape of property '{self.name}' should have at least 3 dimensions (per_atom), found {len(self.value.shape)}"
                )

        return self

    @model_validator(mode="after")
    def _check_unit_type(self) -> Self:

        if self.classification != "meta_data":
            if not self.units.is_compatible_with(
                UnitSystem()[self.property_type], "chem"
            ):
                raise ValueError(
                    f"Unit {self.units} of {self.name} are not compatible with the property type {self.property_type}.\n"
                )
            return self

    @computed_field
    @property
    def n_configs(self) -> int:
        if (
            self.classification == PropertyClassification.per_system
            or self.classification == PropertyClassification.per_atom
        ):
            return self.value.shape[0]
        return None

    @computed_field
    @property
    def n_atoms(self) -> int:
        if self.classification == PropertyClassification.per_atom:
            return self.value.shape[1]
        elif self.classification == PropertyClassification.atomic_numbers:
            return self.value.shape[0]
        return None


class Positions(RecordProperty):
    name: str = "positions"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.length

    @model_validator(mode="after")
    def _check_position_shape(self) -> Self:
        # we already validate that any per_atom property cannot have less than 3 dimensions
        # but we know that positions must be 3d.
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of position should be [n_configs, n_atoms, 3], found {len(self.value.shape)}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of position should be [n_configs, n_atoms, 3], found {len(self.value.shape)}"
            )
        return self


class Energies(RecordProperty):
    name: str = "energies"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.energy

    # if a float is passed, convert to a numpy array of shape (1,1)
    # note if list given, the field validator in the RecordProperty parent class
    # will conver to an ndarray.
    convert_energy_array = field_validator("value", mode="before")(
        _convert_float_to_ndarray
    )

    @model_validator(mode="after")
    def _check_energy_shape(self) -> Self:
        # we already validate that any per_atom property cannot have less than 2 dimensions
        # but energies is a special case that should always be 2.
        if len(self.value.shape) != 2:
            raise ValueError(
                f"Shape of energy should be [n_configs, 1], found {len(self.value.shape)}"
            )
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of energy should be [n_configs, 1], found {len(self.value.shape)}"
            )
        return self


class Forces(RecordProperty):
    name: str = "forces"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.force

    @model_validator(mode="after")
    def _check_force_shape(self) -> Self:
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of force should be [n_configs, n_atoms, 3], found {len(self.value.shape)}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of force should be [n_configs, n_atoms, 3], found {len(self.value.shape)}"
            )
        return self


class PartialCharges(RecordProperty):
    name: str = "partial_charges"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.charge

    @model_validator(mode="after")
    def _check_charge_shape(self) -> Self:
        if self.value.shape[2] != 1:
            raise ValueError(
                f"Shape of charge should be [n_configs, n_atoms, 1], found {len(self.value.shape)}"
            )
        return self


class TotalCharge(RecordProperty):
    name: str = "total_charge"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.charge

    converted_array = field_validator("value", mode="before")(_convert_float_to_ndarray)

    convert_charge_array = field_validator("value", mode="before")(
        _convert_float_to_ndarray
    )

    @model_validator(mode="after")
    def _check_charge_shape(self) -> Self:
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of charge should be [n_configs, 1], found {len(self.value.shape)}"
            )
        return self


class DipoleMoment(RecordProperty):
    name: str = "dipole_moment"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.dipole_moment

    @model_validator(mode="after")
    def _check_dipole_moment_shape(self) -> Self:
        if self.value.shape[1] != 3:
            raise ValueError(
                f"Shape of dipole moment should be [n_configs, 3], found {len(self.value.shape)}"
            )
        return self


class QuadrupoleMoment(RecordProperty):
    name: str = "quadrupole_moment"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.quadrupole_moment


class Polarizability(RecordProperty):
    name: str = "polarizability"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.polarizability


class MetaData(RecordProperty):
    name: str
    value: Union[str, int, float, List, NdArray]
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.meta_data
    property_type: PropertyType = PropertyType.meta_data


class AtomicNumbers(RecordProperty):
    """
    Class to define the atomic numbers of a record.

    The atomic numbers must be a 2d array of shape (n_atoms, 1).
    """

    name: str = "atomic_numbers"
    value: NdArray
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.atomic_numbers
    property_type: PropertyType = PropertyType.atomic_numbers


class Record:
    def __init__(self, name: str, append_property: bool = False):
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
        output_string += f"* per-atom properties ({list(self.per_atom.keys())}):\n"
        for key, value in self.per_atom.items():
            output_string += f" -  {value}\n"
        output_string += f"* per-system properties ({list(self.per_system.keys())}):\n"
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

    def add_properties(self, properties: List[Type[CurateBase]]):
        """
        Add a list of properties to the record.

        Parameters
        ----------
        properties: List[Type[CurateBase]]
            List of properties to add to the record.
        Returns
        -------

        """
        for property in properties:
            self.add_property(property)

    def add_property(self, property: Type[CurateBase]):
        """
        Add a property to the record.

        Parameters
        ----------
        property: Type[CurateBase]
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
                )
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
        unit_system: UnitSystem = UnitSystem(),
        append_property: bool = False,
    ):
        """
        Class to hold a dataset of properties for a given dataset name

        Parameters
        ----------
        dataset_name: str
            Name of the dataset
        unit_system: UnitSystem, optional, default=UnitSystem()
            Unit system to use for the dataset.  Will use default units if not provided.
        append_property: bool, optional, default=False
            If True, append an array to existing array if a property with the same name is added multiple times to a record.
            If False, an error will be raised if trying to add a property with the same name already exists in a record
            Use True if data for configurations are stored in separate files/database entries and you want to combine them.
        """

        self.dataset_name = dataset_name
        self.records = {}
        self.unit_system = unit_system
        self.append_property = append_property

    def create_record(
        self, record_name: str, properties: Optional[List[Type[CurateBase]]] = None
    ):
        """
        Create a record in the dataset. If properties are provided, they will be added to the record.

        Parameters
        ----------
        record_name: str
            Name of the record/
        properties: List[Type[CurateBase]], optional, default=None
            List of properties to add to the record. If not provided, an empty record will be created.

        Returns
        -------

        """
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
        if record_name in self.records.keys():
            self.records.pop(record_name)
        else:
            log.warning(
                f"Record with name {record_name} does not exist in the dataset."
            )

    def add_properties(self, record_name: str, properties: List[Type[CurateBase]]):
        """
        Add a list of properties to a record in the dataset.

        Parameters
        ----------
        record_name: str
            Name of the record to add the properties to.
        properties: List[Type[CurateBase]]
            List of properties to add to the record.

        Returns
        -------

        """
        for property in properties:
            self.add_property(record_name, property)

    def add_property(self, record_name: str, property: Type[CurateBase]):
        """
        Add a property to a record in the dataset.

        Parameters
        ----------
        record_name: str
            Name of the record to add the property to.
        property: Type[CurateBase]
            Property to add to the record.

        Returns
        -------

        """
        # check if the record exists; if it does not add it
        if record_name not in self.records.keys():
            log.info(
                f"Record with name {record_name} does not exist in the dataset. Creating it now."
            )
            self.create_record(record_name)

        self.records[record_name].add_property(property)

        # if property.classification == PropertyClassification.atomic_numbers:
        #
        #     # we will not allow atomic numbers to be set twice
        #     if self.records[record_name].atomic_numbers is not None:
        #         raise ValueError(f"Atomic numbers already set for record {record_name}")
        #
        #     self.records[record_name].atomic_numbers = property.model_copy(deep=True)
        #
        #     # Note, the number of atoms will always be set by the atomic_numbers property.
        #     # We will later validate that per_atom properties are consistent with this value later
        #     # since we are not enforcing that atomic_numbers need to be set before any other property
        #
        # elif property.classification == PropertyClassification.meta_data:
        #     if property.name in self.records[record_name].meta_data.keys():
        #         log.warning(
        #             f"Metadata with name {property.name} already exists in the record {record_name}."
        #         )
        #         raise ValueError(
        #             f"Metadata with name {property.name} already exists in the record {record_name}"
        #         )
        #
        #     elif property.name in self.records[record_name].per_atom.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a per_atom property."
        #         )
        #     elif property.name in self.records[record_name].per_system.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a per_system property."
        #         )
        #     elif property.name == "atomic_numbers":
        #         raise ValueError(
        #             f"The name atomic_numbers is reserved. Use AtomicNumbers to define them, not the MetaData class."
        #         )
        #     self.records[record_name].meta_data[property.name] = property.model_copy(
        #         deep=True
        #     )
        #
        # elif property.classification == PropertyClassification.per_atom:
        #     if property.name in self.records[record_name].per_system.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a per_system property."
        #         )
        #     elif property.name in self.records[record_name].meta_data.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a meta_data property."
        #         )
        #     elif property.name == "atomic_numbers":
        #         raise ValueError(
        #             f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
        #         )
        #     elif property.name in self.records[record_name].per_atom.keys():
        #         if self.append_property == False:
        #             error_msg = f"Property with name {property.name} already exists in the record {record_name}."
        #             error_msg += (
        #                 f"Set append_property=True to append to the existing property."
        #             )
        #             raise ValueError(error_msg)
        #         # if the property already exists, we will use vstack to add it to the existing array
        #         # after first checking that the dimensions are consistent
        #         # note we do not check shape[0], as that corresponds to the number of configurations
        #         assert (
        #             self.records[record_name].per_atom[property.name].value.shape[1]
        #             == property.value.shape[1]
        #         )
        #         assert (
        #             self.records[record_name].per_atom[property.name].value.shape[2]
        #             == property.value.shape[2]
        #         )
        #         # In order to append to the array, everything needs to have the same units
        #         # We will use the units of the first property that was added
        #
        #         temp_array = property.value
        #         if (
        #             property.units
        #             != self.records[record_name].per_atom[property.name].units
        #         ):
        #             temp_array = (
        #                 unit.Quantity(property.value, property.units)
        #                 .to(
        #                     self.records[record_name].per_atom[property.name].units,
        #                     "chem",
        #                 )
        #                 .magnitude
        #             )
        #         self.records[record_name].per_atom[property.name].value = np.vstack(
        #             (
        #                 self.records[record_name].per_atom[property.name].value,
        #                 temp_array,
        #             )
        #         )
        #
        #     else:
        #         self.records[record_name].per_atom[property.name] = property.model_copy(
        #             deep=True
        #         )
        # elif property.classification == PropertyClassification.per_system:
        #     if property.name in self.records[record_name].per_atom.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a per_atom property."
        #         )
        #     elif property.name in self.records[record_name].meta_data.keys():
        #         raise ValueError(
        #             f"Property with name {property.name} already exists in the record {record_name}, but as a meta_data property."
        #         )
        #     elif property.name == "atomic_numbers":
        #         raise ValueError(
        #             f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
        #         )
        #     elif property.name in self.records[record_name].per_system.keys():
        #         if self.append_property == False:
        #             error_msg = f"Property with name {property.name} already exists in the record {record_name}."
        #             error_msg += (
        #                 f"Set append_property=True to append to the existing property."
        #             )
        #             raise ValueError(error_msg)
        #
        #         assert (
        #             self.records[record_name].per_system[property.name].value.shape[1]
        #             == property.value.shape[1]
        #         )
        #         temp_array = property.value
        #         if (
        #             property.units
        #             != self.records[record_name].per_system[property.name].units
        #         ):
        #             temp_array = (
        #                 unit.Quantity(property.value, property.units)
        #                 .to(
        #                     self.records[record_name].per_system[property.name].units,
        #                     "chem",
        #                 )
        #                 .magnitude
        #             )
        #
        #         self.records[record_name].per_system[property.name].value = np.vstack(
        #             (
        #                 self.records[record_name].per_system[property.name].value,
        #                 temp_array,
        #             )
        #         )
        #     else:
        #         self.records[record_name].per_system[property.name] = (
        #             property.model_copy(deep=True)
        #         )

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
        from copy import deepcopy

        return deepcopy(self.records[record_name])

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
            expected_units = self.unit_system[property_type]
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
            expected_units = self.unit_system[property_type]
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

        log.info("Writing records to HDF5 file")
        with h5py.File(full_file_path, "w") as f:
            for record in tqdm(self.records.keys()):
                record_group = f.create_group(record)

                record_group.create_dataset(
                    "atomic_numbers",
                    data=self.records[record].atomic_numbers.value,
                    shape=self.records[record].atomic_numbers.value.shape,
                )
                record_group.create_dataset(
                    "n_configs", data=self.records[record].n_configs
                )

                for key, property in self.records[record].per_atom.items():

                    target_units = self.unit_system[property.property_type]
                    record_group.create_dataset(
                        key,
                        data=unit.Quantity(property.value, property.units)
                        .to(target_units, "chem")
                        .magnitude,
                        shape=property.value.shape,
                    )
                    record_group[key].attrs["u"] = str(target_units)
                    record_group[key].attrs["format"] = "per_atom"

                for key, property in self.records[record].per_system.items():
                    target_units = self.unit_system[property.property_type]
                    record_group.create_dataset(
                        key,
                        data=unit.Quantity(property.value, property.units)
                        .to(target_units, "chem")
                        .magnitude,
                        shape=property.value.shape,
                    )
                    record_group[key].attrs["u"] = str(target_units)
                    record_group[key].attrs["format"] = "per_system"

                for key, property in self.records[record].meta_data.items():
                    if isinstance(property.value, str):
                        record_group.create_dataset(
                            key,
                            data=property.value,
                            dtype=dt,
                        )
                        record_group[key].attrs["u"] = str(target_units)
                        record_group[key].attrs["format"] = "meta_data"
                    elif isinstance(property.value, (float, int)):
                        target_units = self.unit_system[property.property_type]

                        if target_units == unit.dimensionless:
                            record_group.create_dataset(
                                key,
                                data=property.value,
                            )
                        else:
                            record_group.create_dataset(
                                key,
                                data=unit.Quantity(property.value, property.units)
                                .to(target_units, "chem")
                                .magnitude,
                            )
                        record_group[key].attrs["u"] = str(target_units)
                        record_group[key].attrs["format"] = "meta_data"
                    elif isinstance(property.value, np.ndarray):
                        target_units = self.unit_system[property.property_type]

                        if target_units == unit.dimensionless:
                            record_group.create_dataset(
                                key, data=property.value, shape=property.shape
                            )
                        else:
                            record_group.create_dataset(
                                key,
                                data=unit.Quantity(property.value, property.units)
                                .to(target_units, "chem")
                                .magnitude,
                                shape=property.value.shape,
                            )
                        record_group[key].attrs["u"] = str(target_units)
                        record_group[key].attrs["format"] = "meta_data"
                    else:
                        raise ValueError(
                            f"Unsupported type ({type(property.value)})for metadata {key}"
                        )

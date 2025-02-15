"""
This file contains all the pydantic models of the various properties in the curation API
"""

from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
    field_validator,
    computed_field,
)
from enum import Enum
from typing import Union, List
from typing_extensions import Self

from modelforge.curate.utils import (
    NdArray,
    _convert_list_to_ndarray,
    _convert_unit_str_to_unit_unit,
)

from openff.units import unit
from modelforge.curate.units import GlobalUnitSystem, chem_context

# # Define a custom config for the BaseModel to avoid extra duplication of code
# class CurateBase(BaseModel):
#     model_config = ConfigDict(
#         use_enum_values=True,
#         arbitrary_types_allowed=True,
#         validate_assignment=True,
#         extra="forbid",
#         validate_default=True,
#     )


# Define a custom enum class that is case insensitive that we will inherit from
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class PropertyClassification(CaseInsensitiveEnum):
    """
    Enum class to classify a property as per_atom, per_system, atomic_numbers, or meta_data.

    per_atom: properties have shape [n_configs, n_atoms, X]
    per_system: properties have shape [n_configs, X]
    atomic_numbers: special case with shape of [n_atoms, 1].
                    Atomic numbers do not change as configuration changes and thus this reduces memory footprint.
    meta_data: A single entry that may be a string, float, int, or array of any shape.
                    In general, meta_data is ignored when reading in a dataset for training in modelforge

    """

    per_atom = "per_atom"
    per_system = "per_system"
    atomic_numbers = "atomic_numbers"
    meta_data = "meta_data"


class PropertyType(str, Enum):
    """
    Enum class that enables us to know the type of property, e.g., force, energy, charge, etc.

    This is used for validating and converting units.
    """

    length = "length"
    force = "force"
    energy = "energy"
    charge = "charge"
    dipole_moment = "dipole_moment"
    quadrupole_moment = "quadrupole_moment"
    octupole_moment = "octupole_moment"
    polarizability = "polarizability"
    atomic_numbers = "atomic_numbers"
    meta_data = "meta_data"
    frequency = "frequency"
    wavenumber = "wavenumber"
    area = "area"
    heat_capacity = "heat_capacity"
    dimensionless = "dimensionless"


class PropertyBaseModel(BaseModel):
    """
    Base class for all record properties.

    This class defines the basic structure of a record property, which includes:
    name, value, units, classification, and property type.
    This also includes some universal validation checks that are common to all properties.
    Classes for specific properties will inherit from this, although this class can be used directly.
    """

    name: str
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification
    property_type: Union[PropertyType, str]

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
    )

    # if units are passed as string, convert to a unit.Unit object
    converted_units = field_validator("units", mode="before")(
        _convert_unit_str_to_unit_unit
    )
    # if a list is provided, convert to an ndarray
    converted_array = field_validator("value", mode="before")(_convert_list_to_ndarray)

    # general validation of the shape of the value array
    # we can only validate so much since the underlying shape will depend on the property type itself
    # but this should help us to catch some common errors early.
    @model_validator(mode="after")
    def _check_shape(self) -> Self:

        if self.classification == PropertyClassification.atomic_numbers:
            if len(self.value.shape) != 2:
                raise ValueError(
                    f"Shape of '{self.name}' should be [n_atoms,1], found {self.value.shape}"
                )
            if self.value.shape[1] != 1:
                raise ValueError(
                    f"Shape of '{self.name}' should be [n_atoms,1], found {self.value.shape}"
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

    # check compatibility of units with the property type
    # this uses the GlobalUnitSystem to get the expected units for the property type
    @model_validator(mode="after")
    def _check_unit_type(self) -> Self:

        if self.classification != "meta_data":
            if not self.units.is_compatible_with(
                GlobalUnitSystem.get_units(self.property_type), "chem"
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


class Positions(PropertyBaseModel):
    """
    Class to define the positions of a record.

    The positions must be a 3d array of shape (n_configs, n_atoms, 3).

    Parameters
    ----------
    name : str, default="positions"
        The name of the property
    value : np.ndarray
        The array of positions
    units : unit.Unit
        The units of the positions
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.length
        The type of the property

    """

    name: str = "positions"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.length

    # specific validation of the shape of the value array
    @model_validator(mode="after")
    def _check_position_shape(self) -> Self:
        # we already validate that any per_atom property cannot have less than 3 dimensions
        # but we know that positions must be 3d.
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of position should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of position should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        return self


class Energies(PropertyBaseModel):
    """
    Class to define the energies of a record.

    The energies must be a 2d array of shape (n_configs, 1).

    Parameters
    ----------
    name : str, default="energies"
        The name of the property
    value : np.ndarray
        The array of energies
    units : unit.Unit
        The units of the energies
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.energy
        The type of the property

    """

    name: str = "energies"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.energy

    # specific validation of the shape of the value array, which must be 2d for energies
    # with shape [n_configs, 1]
    @model_validator(mode="after")
    def _check_energy_shape(self) -> Self:
        # we already validate that any per_atom property cannot have less than 2 dimensions
        # but energies is a special case that should always be 2.
        if len(self.value.shape) != 2:
            raise ValueError(
                f"Shape of energy should be [n_configs, 1], found {self.value.shape}"
            )
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of energy should be [n_configs, 1], found {self.value.shape}"
            )
        return self


class Forces(PropertyBaseModel):
    """
    Class to define the forces of a record.

    The forces must be a 3d array of shape (n_configs, n_atoms, 3).

    Parameters
    ----------
    name : str, default="forces"
        The name of the property
    value : np.ndarray
        The array of forces
    units : unit.Unit
        The units of the forces
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.force
        The type of the property

    """

    name: str = "forces"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.force

    # specific validation of the value array, which must be 3d for forces, with shape [n_configs, n_atoms, 3]
    @model_validator(mode="after")
    def _check_force_shape(self) -> Self:
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of force should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of force should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        return self


class PartialCharges(PropertyBaseModel):
    """
    Class to define the partial charges of a record.

    The partial charges must be a 3d array of shape (n_configs, n_atoms, 1).

    Parameters
    ----------
    name : str, default="partial_charges"
        The name of the property
    value : np.ndarray
        The array of partial charges
    units : unit.Unit
        The units of the partial charges
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.charge
        The type of the property

    """

    name: str = "partial_charges"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.charge

    # specific validation of the value array, which must be 3d for partial charges, with shape [n_configs, n_atoms, 1]
    @model_validator(mode="after")
    def _check_charge_shape(self) -> Self:
        if self.value.shape[2] != 1:
            raise ValueError(
                f"Shape of charge should be [n_configs, n_atoms, 1], found {self.value.shape}"
            )
        return self


class TotalCharge(PropertyBaseModel):
    """
    Class to define the total charge of a record.

    The total charge must be a 2d array of shape (n_configs, 1).

    Parameters
    ----------
    name : str, default="total_charge"
        The name of the property
    value : np.ndarray
        The array of total charges
    units : unit.Unit
        The units of the total charges
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.charge
        The type of the property
    """

    name: str = "total_charge"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.charge

    # specific validation of the value array, which must be 2d for total charges, with shape [n_configs, 1]
    @model_validator(mode="after")
    def _check_charge_shape(self) -> Self:
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of charge should be [n_configs, 1], found {self.value.shape}"
            )
        if len(self.value.shape) != 2:
            raise ValueError(
                f"Shape of charge should be 2d, found {len(self.value.shape)}"
            )
        return self


class SpinMultiplicities(PropertyBaseModel):
    """
    Class to define the spin multiplicities of a record.

    The spin multiplicities must be a 2d array of shape (n_configs, 1).

    Parameters
    ----------
    name : str, default="spin_multiplicities"
        The name of the property
    value : np.ndarray
        The array of spin multiplicities
    units : unit.Unit
        The units of the spin multiplicities
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.dimensionless
        The type of the property

    """

    name: str = "spin_multiplicities"
    value: NdArray
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.dimensionless

    # specific validation of the value array, which must be 2d for spin multiplicities, with shape [n_configs, 1]
    @model_validator(mode="after")
    def _check_spin_multiplicity_shape(self) -> Self:
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of spin multiplicities should be [n_configs, 1], found {self.value.shape}"
            )
        return self


class DipoleMomentPerSystem(PropertyBaseModel):
    """
    Class to define the per-system dipole moment of a record.

    The per-system dipole moment must be a 2d array of shape (n_configs, 3).

    Parameters
    ----------
    name : str, default="dipole_moment_per_system"
        The name of the property
    value : np.ndarray
        The array of dipole moments
    units : unit.Unit
        The units of the dipole moments
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.dipole_moment
        The type of the property

    """

    name: str = "dipole_moment_per_system"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.dipole_moment

    # specific validation of the value array, which must be 2d for dipole moments, with shape [n_configs, 3]
    @model_validator(mode="after")
    def _check_dipole_moment_shape(self) -> Self:
        if self.value.shape[1] != 3:
            raise ValueError(
                f"Shape of dipole moment should be [n_configs, 3], found {self.value.shape}"
            )
        return self


class DipoleMomentScalarPerSystem(PropertyBaseModel):
    """
    Class to define the per-system scalar dipole moment of a record.

    The per-system scalar dipole moment must be a 2d array of shape (n_configs, 1).

    Parameters
    ----------
    name : str, default="dipole_moment_scalar_per_system"
        The name of the property
    value : np.ndarray
        The array of scalar dipole moments
    units : unit.Unit
        The units of the scalar dipole moments
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.dipole_moment
        The type of the property

    """

    name: str = "dipole_moment_scalar_per_system"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.dipole_moment

    # specific validation of the value array, which must be 2d for scalar dipole moments, with shape [n_configs, 1]
    @model_validator(mode="after")
    def _check_dipole_moment_shape(self) -> Self:
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of scalar dipole moment should be [n_configs, 1], found {self.value.shape}"
            )
        return self


class QuadrupoleMomentPerSystem(PropertyBaseModel):
    """
    Class to define the per-system quadrupole moment of a record.

    The quadrupole moment must be a 2d array of shape (n_configs, 3, 3).
    Parameters
    ----------
    name : str, default="quadrupole_moment_per_system"
        The name of the property
    value : np.ndarray
        The array of quadrupole moments
    units : unit.Unit
        The units of the quadrupole moments
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.quadrupole_moment
        The type of the property

    """

    name: str = "quadrupole_moment_per_system"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.quadrupole_moment

    @model_validator(mode="after")
    def _check_quadrupole_moment_shape(self) -> Self:
        if self.value.shape[1] != 3:
            raise ValueError(
                f"Shape of quadrupole moment should be [n_configs, 3, 3], found {self.value.shape}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of quadrupole moment should be [n_configs, 3, 3], found {self.value.shape}"
            )
        return self


class DipoleMomentPerAtom(PropertyBaseModel):
    """
    Class to define the per-atom dipole moment of a record.

    The per-atom dipole moment must be a 3d array of shape (n_configs, n_atoms, 3).

    Parameters
    ----------
    name : str, default="dipole_moment_per_atom"
        The name of the property
    value : np.ndarray
        The array of dipole moments
    units : unit.Unit
        The units of the dipole moments
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.dipole_moment
        The type of the property

    """

    name: str = "dipole_moment_per_atom"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.dipole_moment

    # specific validation of the value array, which must be 3d for dipole moments, with shape [n_configs, n_atoms, 3]
    @model_validator(mode="after")
    def _check_dipole_moment_shape(self) -> Self:
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of dipole moment should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of dipole moment should be [n_configs, n_atoms, 3], found {self.value.shape}"
            )
        return self


class QuadrupoleMomentPerAtom(PropertyBaseModel):
    """
    Class to define the per-atom quadrupole moment of a record.

    The per-atom quadrupole moment must be a 3d array of shape (n_configs, n_atoms, 3, 3).

    Parameters
    ----------
    name : str, default="quadrupole_moment_per_atom"
        The name of the property
    value : np.ndarray
        The array of quadrupole moments
    units : unit.Unit
        The units of the quadrupole moments
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.quadrupole_moment
        The type of the property

    """

    name: str = "quadrupole_moment_per_atom"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.quadrupole_moment

    @model_validator(mode="after")
    def _check_quadrupole_moment_shape(self) -> Self:
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of quadrupole moment should be [n_configs, n_atoms, 3, 3], found {self.value.shape}"
            )
        if self.value.shape[3] != 3:
            raise ValueError(
                f"Shape of quadrupole moment should be [n_configs, n_atoms, 3, 3], found {self.value.shape}"
            )
        return self


class OctupoleMomentPerAtom(PropertyBaseModel):
    """
    Class to define the per-atom octupole moment of a record.

    The per-atom octupole moment must be a 3d array of shape (n_configs, n_atoms, 3, 3, 3).
    Parameters
    ----------
    name : str, default="octupole_moment_per_atom"
        The name of the property
    value : np.ndarray
        The array of octupole moments
    units : unit.Unit
        The units of the octupole moments
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.octupole_moment
        The type of the property
    """

    name: str = "octupole_moment_per_atom"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.octupole_moment

    @model_validator(mode="after")
    def _check_octupole_moment_shape(self) -> Self:
        if self.value.shape[2] != 3:
            raise ValueError(
                f"Shape of octupole moment should be [n_configs, n_atoms, 3, 3, 3], found {self.value.shape}"
            )
        if self.value.shape[3] != 3:
            raise ValueError(
                f"Shape of octupole moment should be [n_configs, n_atoms, 3, 3, 3], found {self.value.shape}"
            )
        if self.value.shape[4] != 3:
            raise ValueError(
                f"Shape of octupole moment should be [n_configs, n_atoms, 3, 3, 3], found {self.value.shape}"
            )
        return self


class Polarizability(PropertyBaseModel):
    """
    Class to define the polarizability of a record.

    The polarizability must be a 2d array of shape (n_configs, 1).

    Parameters
    ----------
    name : str, default="polarizability"
        The name of the property
    value : np.ndarray
        The array of polarizabilities
    units : unit.Unit
        The units of the polarizabilities
    classification : PropertyClassification, default=PropertyClassification.per_system
        The classification of the property
    property_type : PropertyType, default=PropertyType.polarizability
        The type of the property

    """

    name: str = "polarizability"
    value: NdArray
    units: unit.Unit
    classification: PropertyClassification = PropertyClassification.per_system
    property_type: PropertyType = PropertyType.polarizability

    # specific validation of the value array, which must be 2d for polarizabilities, with shape [n_configs, 1]
    @model_validator(mode="after")
    def _check_polarizability_shape(self) -> Self:
        if self.value.shape[1] != 1:
            raise ValueError(
                f"Shape of polarizability should be [n_configs, 1], found {self.value.shape}"
            )
        if len(self.value.shape) != 2:
            raise ValueError(
                f"Shape of polarizability should be 2d, found {len(self.value.shape)}"
            )
        return self


class MetaData(PropertyBaseModel):
    """
    Class to define the metadata of a record.

    The metadata can be a string, float, int, or array of any shape.

    Parameters
    ----------
    name : str
        The name of the property
    value : Union[str, int, float, List, np.ndarray]
        The metadata value
    units : unit.Unit, default=unit.dimensionless
        The units of the metadata
    classification : PropertyClassification, default=PropertyClassification.meta_data
        The classification of the property
    property_type : PropertyType, default=PropertyType.meta_data
        The type of the property

    """

    name: str
    value: Union[str, int, float, List, NdArray]
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.meta_data
    property_type: PropertyType = PropertyType.meta_data


class AtomicNumbers(PropertyBaseModel):
    """
    Class to define the atomic numbers of a record.

    The atomic numbers must be a 2d array of shape (n_atoms, 1).

    Parameters
    ----------
    name : str, default="atomic_numbers"
        The name of the property
    value : np.ndarray
        The array of atomic numbers
    units : unit.Unit, default=unit.dimensionless
        The units of the atomic numbers
    classification : PropertyClassification, default=PropertyClassification.atomic_numbers
        The classification of the property
    property_type : PropertyType, default=PropertyType.atomic_numbers
        The type of the property

    """

    name: str = "atomic_numbers"
    value: NdArray
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.atomic_numbers
    property_type: PropertyType = PropertyType.atomic_numbers


class BondOrders(PropertyBaseModel):
    """
    Class to define the bond orders of a record.

    The bond orders must be a 3d array of shape (n_configs, n_atoms, n_atoms).

    Parameters
    ----------
    name : str, default="bond_orders"
        The name of the property
    value : np.ndarray
        The array of bond orders
    units : unit.Unit, default=unit.dimensionless
        The units of the bond orders
    classification : PropertyClassification, default=PropertyClassification.per_atom
        The classification of the property
    property_type : PropertyType, default=PropertyType.dimensionless
        The type of the property

    """

    name: str = "bond_orders"
    value: NdArray
    units: unit.Unit = unit.dimensionless
    classification: PropertyClassification = PropertyClassification.per_atom
    property_type: PropertyType = PropertyType.dimensionless

    @model_validator(mode="after")
    def _check_bond_order_shape(self) -> Self:
        if len(self.value.shape) != 3:
            raise ValueError(
                f"Shape of bond orders should be length 3,, found {len(self.value.shape)}"
            )
        if self.value.shape[1] != self.value.shape[2]:
            raise ValueError(
                f"Shape of bond orders should be [n_configs, n_atoms, n_atoms], found {self.value.shape}"
            )
        return self

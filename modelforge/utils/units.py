"""
Module that handles unit system definitions and conversions.

This module defines a custom unit context for converting between various
energy units and includes utility functions for handling units within
the model forge framework.
"""

from typing import Union

from openff.units import unit

# Define a chemical context for unit transformations
# This allows conversions between energy units like hartree and kJ/mol
__all__ = ["chem_context"]
chem_context = unit.Context("chem")

# Add transformations to handle conversions between energy units per substance
# (mole) and other forms
chem_context.add_transformation(
    "[force] * [length]",
    "[force] * [length]/[substance]",
    lambda unit, x: x * unit.avogadro_constant,
)
chem_context.add_transformation(
    "[force] * [length]/[substance]",
    "[force] * [length]",
    lambda unit, x: x / unit.avogadro_constant,
)
chem_context.add_transformation(
    "[force] * [length]/[length]",
    "[force] * [length]/[substance]/[length]",
    lambda unit, x: x * unit.avogadro_constant,
)
chem_context.add_transformation(
    "[force] * [length]/[substance]/[length]",
    "[force] * [length]/[length]",
    lambda unit, x: x / unit.avogadro_constant,
)

chem_context.add_transformation(
    "[force] * [length]/[length]/[length]",
    "[force] * [length]/[substance]/[length]/[length]",
    lambda unit, x: x * unit.avogadro_constant,
)
chem_context.add_transformation(
    "[force] * [length]/[substance]/[length]/[length]",
    "[force] * [length]/[length]/[length]",
    lambda unit, x: x / unit.avogadro_constant,
)

# Register the custom chemical context for use with the unit system
unit.add_context(chem_context)


def _convert_str_or_unit_to_unit_length(val: Union[unit.Quantity, str]) -> float:
    """
    Convert a string or unit.Quantity representation of a length to Global length unit (default nanometers).

    This function ensures that any input, whether a string or an OpenFF
    unit.Quantity, is converted to a unit.Quantity in nanometers and returns the
    magnitude.

    Parameters
    ----------
    val : Union[unit.Quantity, str]
        The value to convert to a unit length (nanometers).

    Returns
    -------
    float
        The value in the Global length units (default: nanometers).

    Examples
    --------
    >>> _convert_str_or_unit_to_unit_length("1.0 * nanometer")
    1.0
    >>> _convert_str_or_unit_to_unit_length(unit.Quantity(1.0, unit.angstrom))
    0.1
    """
    if isinstance(val, str):
        val = unit.Quantity(val)
    return val.to(GlobalUnitSystem.get_units("length")).m


def _convert_str_to_unit(val: Union[unit.Quantity, str]) -> unit.Quantity:
    """
    Convert a string representation of a unit to an OpenFF unit.Quantity.

    If the input is already a unit.Quantity, it is returned as is.
    Parameters
    ----------
    val : Union[unit.Quantity, str]
        The value to convert to a unit.Quantity

    Returns
    -------
    unit.Quantity
        The value and unit as a unit.Quantity

    Examples
    --------
    >>> _convert_str_to_unit("1.0 * kilocalorie / mole")
    Quantity(value=1.0, unit=kilocalorie/mole)
    >>> _convert_str_to_unit(unit.Quantity(1.0, unit.kilocalorie / unit.mole))
    Quantity(value=1.0, unit=kilocalorie/mole)

    """
    if isinstance(val, str):
        return unit.Quantity(val)
    return val


# if units are given as openff.unit compatible strings, convert them to openff unit.Unit objects
def _convert_unit_str_to_unit_unit(value: Union[str, unit.Unit]):
    """
    This will convert a string representation of a unit to an openff unit.Unit object

    If the input is a unit.Unit, nothing will be changed.

    Parameters
    ----------
    value: Union[str, unit.Unit]
        The value to convert to a unit.Unit object

    Returns
    -------
        unit.Unit

    """
    if isinstance(value, str):
        return unit.Unit(value)
    return value


class GlobalUnitSystem:
    """
    Class that defines the global unit system for the modelforge.


    """

    name = "default"
    length = unit.nanometer
    area = unit.nanometer**2
    force = unit.kilojoule_per_mole / unit.nanometer
    energy = unit.kilojoule_per_mole
    charge = unit.elementary_charge
    dipole_moment = unit.elementary_charge * unit.nanometer
    quadrupole_moment = unit.elementary_charge * unit.nanometer**2
    octupole_moment = unit.elementary_charge * unit.nanometer**3
    frequency = unit.gigahertz
    wavenumber = unit.cm**-1
    polarizability = unit.nanometer**3
    heat_capacity = unit.kilojoule_per_mole / unit.kelvin
    atomic_numbers = unit.dimensionless
    dimensionless = unit.dimensionless

    @classmethod
    def set_global_units(cls, property_type: str, units: Union[str, unit.Unit]):
        """
        This can be used to add a new property/unit combination to the class
        or change the default units for a property in the class.

        Parameters
        ----------
        property_type, str:
            type of the property (e.g., length, force, energy, charge, etc.)
        units, openff.units.Unit or str:
            openff.units object or compatible string that defines the units of the property type

        """
        if isinstance(units, str):
            units = _convert_unit_str_to_unit_unit(units)

        if not isinstance(units, unit.Unit):
            raise ValueError(
                "Units must be an openff.units object or compatible string."
            )

        setattr(cls, property_type, units)

    @classmethod
    def get_units(cls, key):

        return getattr(cls, key)

    def __repr__(self):

        attributes_to_print = {
            attr: getattr(self, attr) for attr in dir(self) if not attr.startswith("__")
        }
        attributes_to_print.pop("get_units")
        attributes_to_print.pop("set_global_units")
        return "\n".join(
            [f"{key} : {value}" for key, value in attributes_to_print.items()]
        )

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise AttributeError(f"Unit {item} not found in the unit system.")


def print_modelforge_unit_system():
    """
    Provide details about the used unit systems.
    """
    from loguru import logger as log

    log.info(GlobalUnitSystem)

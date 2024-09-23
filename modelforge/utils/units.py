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
    Convert a string or unit.Quantity representation of a length to nanometers.

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
        The value in nanometers.

    Examples
    --------
    >>> _convert_str_or_unit_to_unit_length("1.0 * nanometer")
    1.0
    >>> _convert_str_or_unit_to_unit_length(unit.Quantity(1.0, unit.angstrom))
    0.1
    """
    if isinstance(val, str):
        val = unit.Quantity(val)
    return val.to(unit.nanometer).m


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


def print_modelforge_unit_system():
    """
    Provide details about the used unit systems.
    """
    from loguru import logger as log

    log.info("Distance are in nanometer.")
    log.info("Energies are in kJ/mol")
    log.info("Forces are in kJ/mol/nm**2")

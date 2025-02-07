from openff.units import unit
from openff.units import unit
from typing import Union


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


class GlobalUnitSystem:
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
            from modelforge.curate.utils import _convert_unit_str_to_unit_unit

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
        # if item in cls.__dict__.keys():
        #     return getattr(cls, item)
        # else:
        #     return None
        try:
            return getattr(self, item)
        except AttributeError:
            raise AttributeError(f"Unit {item} not found in the unit system.")

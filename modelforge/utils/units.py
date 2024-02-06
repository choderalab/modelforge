from openff.units import unit, Quantity

# define new context for converting energy (e.g., hartree)
# to energy/mol (e.g., kJ/mol)

c = unit.Context("chem")
c.add_transformation(
    "[force] * [length]",
    "[force] * [length]/[substance]",
    lambda unit, x: x * unit.avogadro_constant,
)
c.add_transformation(
    "[force] * [length]/[substance]",
    "[force] * [length]",
    lambda unit, x: x / unit.avogadro_constant,
)
c.add_transformation(
    "[force] * [length]/[length]",
    "[force] * [length]/[substance]/[length]",
    lambda unit, x: x * unit.avogadro_constant,
)
c.add_transformation(
    "[force] * [length]/[substance]/[length]",
    "[force] * [length]/[length]",
    lambda unit, x: x / unit.avogadro_constant,
)

c.add_transformation(
    "[force] * [length]/[length]/[length]",
    "[force] * [length]/[substance]/[length]/[length]",
    lambda unit, x: x * unit.avogadro_constant,
)
c.add_transformation(
    "[force] * [length]/[substance]/[length]/[length]",
    "[force] * [length]/[length]/[length]",
    lambda unit, x: x / unit.avogadro_constant,
)


unit.add_context(c)


def provide_details_about_used_unitsystem():
    """
    Provide details about the used unit systems.
    """
    from loguru import logger as log

    log.info("Distance are in nanometer.")
    log.info("Energies are in kJ/mol")
    log.info("Forces are in kJ/mol/nm**2")

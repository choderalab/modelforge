import pytest
import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset
from modelforge.curate.units import GlobalUnitSystem
from modelforge.curate.properties import *


def test_units_representation(capsys):
    print(GlobalUnitSystem())

    out, _ = capsys.readouterr()

    assert "area : nanometer ** 2" in out
    assert "atomic_numbers : dimensionless" in out
    assert "charge : elementary_charge" in out
    assert "dimensionless : dimensionless" in out
    assert "dipole_moment : elementary_charge * nanometer" in out
    assert "energy : kilojoule_per_mole" in out
    assert "force : kilojoule_per_mole / nanometer" in out
    assert "frequency : gigahertz" in out
    assert "heat_capacity : kilojoule_per_mole / kelvin" in out
    assert "length : nanometer" in out
    assert "octupole_moment : elementary_charge * nanometer ** 3" in out
    assert "polarizability : nanometer ** 3" in out
    assert "quadrupole_moment : elementary_charge * nanometer ** 2" in out
    assert "wavenumber : 1 / centimeter" in out

    assert "get_units" not in out
    assert "set_global_units" not in out

    # add in a unit and assert it shows up in the print
    GlobalUnitSystem.set_global_units("pressure", unit.bar)
    print(GlobalUnitSystem())
    out, _ = capsys.readouterr()
    assert "pressure : bar" in out


def test_default_unit_system():

    assert GlobalUnitSystem.name == "default"
    assert GlobalUnitSystem.length == unit.nanometer
    assert GlobalUnitSystem.force == unit.kilojoule_per_mole / unit.nanometer
    assert GlobalUnitSystem.energy == unit.kilojoule_per_mole
    assert GlobalUnitSystem.charge == unit.elementary_charge
    assert GlobalUnitSystem.dipole_moment == unit.elementary_charge * unit.nanometer
    assert (
        GlobalUnitSystem.quadrupole_moment == unit.elementary_charge * unit.nanometer**2
    )
    assert GlobalUnitSystem.polarizability == unit.nanometer**3
    assert GlobalUnitSystem.atomic_numbers == unit.dimensionless
    assert GlobalUnitSystem.dimensionless == unit.dimensionless

    assert GlobalUnitSystem.get_units("length") == unit.nanometer
    assert (
        GlobalUnitSystem.get_units("force") == unit.kilojoule_per_mole / unit.nanometer
    )
    assert GlobalUnitSystem.get_units("energy") == unit.kilojoule_per_mole
    assert GlobalUnitSystem.get_units("charge") == unit.elementary_charge
    assert (
        GlobalUnitSystem.get_units("dipole_moment")
        == unit.elementary_charge * unit.nanometer
    )
    assert (
        GlobalUnitSystem.get_units("quadrupole_moment")
        == unit.elementary_charge * unit.nanometer**2
    )
    assert GlobalUnitSystem.get_units("polarizability") == unit.nanometer**3
    assert GlobalUnitSystem.get_units("atomic_numbers") == unit.dimensionless
    assert GlobalUnitSystem.get_units("dimensionless") == unit.dimensionless


def test_set_global_units():
    GlobalUnitSystem.length = unit.angstrom
    assert GlobalUnitSystem.length == unit.angstrom

    GlobalUnitSystem.set_global_units("pressure", unit.bar)
    assert GlobalUnitSystem.get_units("pressure") == unit.bar

    GlobalUnitSystem.set_global_units("length", unit.nanometer)
    assert GlobalUnitSystem.length == unit.nanometer

    GlobalUnitSystem.set_global_units("length", "nanometer")
    assert GlobalUnitSystem.length == unit.nanometer

    import pint

    with pytest.raises(pint.errors.UndefinedUnitError):
        GlobalUnitSystem.set_global_units("length", "not_a_unit")

    with pytest.raises(ValueError):
        GlobalUnitSystem.set_global_units("length", 123.234)

    assert GlobalUnitSystem.length == unit.nanometer

    # test setting a unit that is not in the global unit system
    # will produce an error
    with pytest.raises(AttributeError):
        GlobalUnitSystem.tacos == unit.nanometer


def test_conversion():
    from modelforge.curate.utils import _convert_unit_str_to_unit_unit

    output = _convert_unit_str_to_unit_unit("nanometer")
    assert output.is_compatible_with(unit.nanometer)
    assert output == unit.nanometer
    assert isinstance(output, unit.Unit)

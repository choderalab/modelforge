import pytest
import numpy as np
from openff.units import unit

from modelforge.curate.properties import *


def test_initialize_positions():
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")

    assert type(positions.value) == np.ndarray
    assert positions.units == unit.nanometer
    assert positions.value.shape == (1, 2, 3)
    assert positions.classification == "per_atom"
    assert positions.name == "positions"
    assert positions.property_type == "length"

    assert positions.n_atoms == 2
    assert positions.n_configs == 1

    # wrong value.shape[2]
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[[1.0, 1.0, 1.0, 2.0], [2.0, 2.0, 2.0, 3.0]]], units="nanometer"
        )
    # wrong shape
    with pytest.raises(ValueError):
        positions = Positions(value=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], units="meter")

    # no units, we don't assume, must specify
    with pytest.raises(ValueError):
        positions = Positions(value=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    # incompatible units
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units=unit.hartree
        )
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], units=unit.kilojoule_per_mole
        )


def test_initialize_energies():
    energies = Energies(value=np.array([[0.1], [0.3]]), units=unit.hartree)

    assert type(energies.value) == np.ndarray
    assert energies.units == unit.hartree
    assert energies.value.shape == (2, 1)
    assert energies.classification == "per_system"
    assert energies.name == "energies"
    assert energies.property_type == "energy"
    assert energies.n_configs == 2

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([0.1]), units=unit.hartree)

    # incompatible units
    with pytest.raises(ValueError):
        energies = Energies(
            value=np.array([[0.1], [0.3]]), units=unit.kilojoule_per_mole**2
        )

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([[0.1, 0.3]]), units=unit.hartree)

    with pytest.raises(ValueError):
        energies = Energies(value=np.array([[[0.1]]]), units=unit.hartree)


def test_initialize_atomic_numbers():
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    assert type(atomic_numbers.value) == np.ndarray
    assert atomic_numbers.value.shape == (2, 1)
    assert atomic_numbers.classification == "atomic_numbers"
    assert atomic_numbers.name == "atomic_numbers"
    assert atomic_numbers.property_type == "atomic_numbers"
    assert atomic_numbers.n_atoms == 2

    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([1, 6]))
        # wrong shape
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6]]))

        # wrong shape
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6], [1, 6]]))

    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(
            value=np.array([[1.0], [6.0]]), units="nanometers"
        )


def test_initialize_meta_data():
    meta_data = MetaData(name="smiles", value="[CH]")
    assert meta_data.name == "smiles"
    assert meta_data.value == "[CH]"
    assert meta_data.classification == "meta_data"
    assert meta_data.property_type == "meta_data"


def test_initialize_partial_charges():
    partial_charges = PartialCharges(
        value=np.array([[[0.1], [-0.1]]]), units=unit.elementary_charge
    )
    assert partial_charges.value.shape == (1, 2, 1)
    assert partial_charges.units == unit.elementary_charge
    assert partial_charges.classification == "per_atom"
    assert partial_charges.name == "partial_charges"
    assert partial_charges.property_type == "charge"

    # wrong shape
    with pytest.raises(ValueError):
        partial_charges = PartialCharges(
            value=np.array([[[0.1, -0.1]]]), units=unit.elementary_charge
        )

    with pytest.raises(ValueError):
        partial_charges = PartialCharges(
            value=np.array([0.1]), units=unit.elementary_charge
        )


def test_initialize_forces():
    forces = Forces(
        value=np.array([[[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]]),
        units=unit.kilojoule_per_mole / unit.nanometer,
    )
    assert forces.value.shape == (1, 2, 3)
    assert forces.units == unit.kilojoule_per_mole / unit.nanometer
    assert forces.classification == "per_atom"
    assert forces.name == "forces"
    assert forces.property_type == "force"

    with pytest.raises(ValueError):
        forces = Forces(
            value=np.array([[0.1, 0.1, 0.1, 0.1], [-0.1, -0.1, -0.1, 0.1]]),
            units=unit.kilojoule_per_mole / unit.nanometer,
        )
    with pytest.raises(ValueError):
        forces = Forces(
            value=np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]),
            units=unit.kilojoule_per_mole / unit.nanometer,
        )
    with pytest.raises(ValueError):
        forces = Forces(
            value=np.array([0.1, 0.1, 0.1, -0.1, -0.1, -0.1]),
            units=unit.kilojoule_per_mole / unit.nanometer,
        )


def test_initialize_total_charge():
    total_charge = TotalCharge(value=np.array([[0.1]]), units=unit.elementary_charge)
    assert total_charge.value.shape == (1, 1)
    assert total_charge.units == unit.elementary_charge
    assert total_charge.classification == "per_system"
    assert total_charge.name == "total_charge"
    assert total_charge.property_type == "charge"

    with pytest.raises(ValueError):
        total_charge = TotalCharge(value=np.array([0.1]), units=unit.elementary_charge)
    with pytest.raises(ValueError):
        total_charge = TotalCharge(
            value=np.array([[0.1, 0.1]]), units=unit.elementary_charge
        )
    with pytest.raises(ValueError):
        total_charge = TotalCharge(
            value=np.array([[[0.1, 0.1, 0.1]]]), units=unit.elementary_charge
        )


def test_initialize_dipole_moment_per_system():
    dipole_moment = DipoleMomentPerSystem(
        value=np.array([[0.1, 0.1, 0.1]]), units=unit.debye
    )
    assert dipole_moment.value.shape == (1, 3)
    assert dipole_moment.units == unit.debye
    assert dipole_moment.classification == "per_system"
    assert dipole_moment.name == "dipole_moment_per_system"
    assert dipole_moment.property_type == "dipole_moment"

    with pytest.raises(ValueError):
        dipole_moment = DipoleMomentPerSystem(
            value=np.array([0.1, 0.1, 0.1]), units=unit.debye
        )
    with pytest.raises(ValueError):
        dipole_moment = DipoleMomentPerSystem(
            value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]), units=unit.debye
        )


def test_initialize_quadruople_moment_per_system():
    quadrupole_moment = QuadrupoleMomentPerSystem(
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
        units=unit.debye * unit.nanometer,
    )
    assert quadrupole_moment.value.shape == (1, 3, 3)
    assert quadrupole_moment.units == unit.debye * unit.nanometer
    assert quadrupole_moment.classification == "per_system"
    assert quadrupole_moment.name == "quadrupole_moment_per_system"
    assert quadrupole_moment.property_type == "quadrupole_moment"


def test_initialize_dipole_moment_scalar_per_system():
    dipole_moment_scalar = DipoleMomentScalarPerSystem(
        value=np.array([[0.1]]), units=unit.debye
    )
    assert dipole_moment_scalar.value.shape == (1, 1)
    assert dipole_moment_scalar.units == unit.debye
    assert dipole_moment_scalar.classification == "per_system"
    assert dipole_moment_scalar.name == "dipole_moment_scalar_per_system"
    assert dipole_moment_scalar.property_type == "dipole_moment"


def test_initialize_dipole_moment_per_atom():
    dipole_moment_per_atom = DipoleMomentPerAtom(
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]), units=unit.debye
    )
    assert dipole_moment_per_atom.value.shape == (1, 2, 3)
    assert dipole_moment_per_atom.units == unit.debye
    assert dipole_moment_per_atom.classification == "per_atom"
    assert dipole_moment_per_atom.name == "dipole_moment_per_atom"
    assert dipole_moment_per_atom.property_type == "dipole_moment"
    assert dipole_moment_per_atom.n_atoms == 2
    assert dipole_moment_per_atom.n_configs == 1


def test_initialize_quadrupole_moment_per_atom():
    quadrupole_moment_per_atom = QuadrupoleMomentPerAtom(
        value=np.array(
            [
                [
                    [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                    [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                ]
            ]
        ),
        units=unit.debye * unit.nanometer,
    )
    assert quadrupole_moment_per_atom.value.shape == (1, 2, 3, 3)
    assert quadrupole_moment_per_atom.units == unit.debye * unit.nanometer
    assert quadrupole_moment_per_atom.classification == "per_atom"
    assert quadrupole_moment_per_atom.name == "quadrupole_moment_per_atom"
    assert quadrupole_moment_per_atom.property_type == "quadrupole_moment"
    assert quadrupole_moment_per_atom.n_atoms == 2
    assert quadrupole_moment_per_atom.n_configs == 1


def test_initialize_octupole_moment_per_atom():
    # octupole moment has value of shape (M, N, 3,3,3)
    octupole_moment_per_atom = OctupoleMomentPerAtom(
        value=np.array(
            [
                [
                    [
                        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                    ]
                ]
            ]
        ),
        units=unit.debye * unit.nanometer**2,
    )

    assert octupole_moment_per_atom.value.shape == (1, 1, 3, 3, 3)
    assert octupole_moment_per_atom.units == unit.debye * unit.nanometer**2
    assert octupole_moment_per_atom.classification == "per_atom"
    assert octupole_moment_per_atom.name == "octupole_moment_per_atom"
    assert octupole_moment_per_atom.property_type == "octupole_moment"
    assert octupole_moment_per_atom.n_atoms == 1
    assert octupole_moment_per_atom.n_configs == 1


def test_initialize_polarizability():
    polarizability = Polarizability(
        value=np.array([[0.1]]),
        units=unit.bohr**3,
    )
    assert polarizability.value.shape == (1, 1)
    assert polarizability.units == unit.bohr**3
    assert polarizability.classification == "per_system"
    assert polarizability.name == "polarizability"
    assert polarizability.property_type == "polarizability"

    with pytest.raises(ValueError):
        polarizability = Polarizability(
            value=np.array([0.1]),
            units=unit.bohr**3,
        )
    with pytest.raises(ValueError):
        polarizability = Polarizability(
            value=np.array([[0.1, 0.1]]),
            units=unit.bohr**3,
        )
    with pytest.raises(ValueError):
        polarizability = Polarizability(
            value=np.array([[[0.1]]]),
            units=unit.bohr**3,
        )


def test_initialize_base_model():
    base_prop = PropertyBaseModel(
        name="test_prop",
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
        units=unit.nanometer,
        property_type="length",
        classification="per_atom",
    )
    assert base_prop.value.shape == (1, 3, 3)
    assert base_prop.units == unit.nanometer
    assert base_prop.classification == "per_atom"
    assert base_prop.name == "test_prop"
    assert base_prop.property_type == "length"


def test_initialize_bond_orders():
    bond_orders = BondOrders(value=np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]))
    assert bond_orders.value.shape == (1, 3, 3)

    with pytest.raises(ValueError):
        bond_orders = BondOrders(value=np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        bond_orders = BondOrders(value=np.array([[1, 2, 3]]))
    with pytest.raises(ValueError):
        bond_orders = BondOrders(value=np.array([[[1, 2, 3]]]))
    with pytest.raises(ValueError):
        bond_orders = BondOrders(value=np.array([[[1, 2, 3], [1, 2, 3]]]))

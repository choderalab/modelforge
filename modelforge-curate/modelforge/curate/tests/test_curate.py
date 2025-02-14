import pytest
import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset
from modelforge.curate.units import GlobalUnitSystem
from modelforge.curate.properties import *


def test_source_dataset_init():
    new_dataset = SourceDataset("test_dataset")
    assert new_dataset.dataset_name == "test_dataset"

    new_dataset.create_record("mol1")
    assert "mol1" in new_dataset.records

    new_dataset.create_record("mol2")

    assert len(new_dataset.records) == 2


def test_dataset_create_record():
    # test creating a record that already exists
    # this will fail
    new_dataset = SourceDataset("test_dataset")
    new_dataset.create_record("mol1")
    assert "mol1" in new_dataset.records
    with pytest.raises(ValueError):
        new_dataset.create_record("mol1")

    new_dataset.create_record("mol2")
    assert "mol2" in new_dataset.records
    assert "mol1" in new_dataset.records
    assert len(new_dataset.records) == 2

    record = Record(name="mol1")

    with pytest.raises(ValueError):
        new_dataset.add_record(record)

    record = Record(name="mol3")
    new_dataset.add_record(record)
    assert "mol3" in new_dataset.records
    assert len(new_dataset.records) == 3

    new_dataset.remove_record("mol3")
    assert "mol3" not in new_dataset.records

    # we already removed it so it doesn't exist, this will do nothing
    # but will log a warning
    new_dataset.remove_record("mol4")

    # let us make sure we create a new record if we try to add a record that doesn't exist
    property = AtomicNumbers(value=np.array([[1], [6]]))
    assert "mol4" not in new_dataset.records
    new_dataset.add_properties("mol4", [property])
    assert "mol4" in new_dataset.records


def test_initialize_properties():
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")

    assert type(positions.value) == np.ndarray
    assert positions.units == unit.nanometer
    assert positions.value.shape == (1, 2, 3)
    assert positions.classification == "per_atom"
    assert positions.name == "positions"
    assert positions.property_type == "length"

    assert positions.n_atoms == 2
    assert positions.n_configs == 1

    energies = Energies(value=np.array([[0.1], [0.3]]), units=unit.hartree)

    assert type(energies.value) == np.ndarray
    assert energies.units == unit.hartree
    assert energies.value.shape == (2, 1)
    assert energies.classification == "per_system"
    assert energies.name == "energies"
    assert energies.property_type == "energy"
    assert energies.n_configs == 2

    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    assert type(atomic_numbers.value) == np.ndarray
    assert atomic_numbers.value.shape == (2, 1)
    assert atomic_numbers.classification == "atomic_numbers"
    assert atomic_numbers.name == "atomic_numbers"
    assert atomic_numbers.property_type == "atomic_numbers"
    assert atomic_numbers.n_atoms == 2

    meta_data = MetaData(name="smiles", value="[CH]")
    assert meta_data.name == "smiles"
    assert meta_data.value == "[CH]"
    assert meta_data.classification == "meta_data"
    assert meta_data.property_type == "meta_data"

    partial_charges = PartialCharges(
        value=np.array([[[0.1], [-0.1]]]), units=unit.elementary_charge
    )
    assert partial_charges.value.shape == (1, 2, 1)
    assert partial_charges.units == unit.elementary_charge
    assert partial_charges.classification == "per_atom"
    assert partial_charges.name == "partial_charges"
    assert partial_charges.property_type == "charge"

    forces = Forces(
        value=np.array([[[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]]),
        units=unit.kilojoule_per_mole / unit.nanometer,
    )
    assert forces.value.shape == (1, 2, 3)
    assert forces.units == unit.kilojoule_per_mole / unit.nanometer
    assert forces.classification == "per_atom"
    assert forces.name == "forces"
    assert forces.property_type == "force"

    total_charge = TotalCharge(value=np.array([[[0.1]]]), units=unit.elementary_charge)
    assert total_charge.value.shape == (1, 1, 1)
    assert total_charge.units == unit.elementary_charge
    assert total_charge.classification == "per_system"
    assert total_charge.name == "total_charge"
    assert total_charge.property_type == "charge"

    dipole_moment = DipoleMomentPerSystem(
        value=np.array([[0.1, 0.1, 0.1]]), units=unit.debye
    )
    assert dipole_moment.value.shape == (1, 3)
    assert dipole_moment.units == unit.debye
    assert dipole_moment.classification == "per_system"
    assert dipole_moment.name == "dipole_moment_per_system"
    assert dipole_moment.property_type == "dipole_moment"

    quadrupole_moment = QuadrupoleMomentPerSystem(
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
        units=unit.debye * unit.nanometer,
    )
    assert quadrupole_moment.value.shape == (1, 3, 3)
    assert quadrupole_moment.units == unit.debye * unit.nanometer
    assert quadrupole_moment.classification == "per_system"
    assert quadrupole_moment.name == "quadrupole_moment_per_system"
    assert quadrupole_moment.property_type == "quadrupole_moment"

    dipole_moment_scalar = DipoleMomentScalarPerSystem(
        value=np.array([[0.1]]), units=unit.debye
    )
    assert dipole_moment_scalar.value.shape == (1, 1)
    assert dipole_moment_scalar.units == unit.debye
    assert dipole_moment_scalar.classification == "per_system"
    assert dipole_moment_scalar.name == "dipole_moment_scalar_per_system"
    assert dipole_moment_scalar.property_type == "dipole_moment"

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

    polarizability = Polarizability(
        value=np.array([[0.1]]),
        units=unit.bohr**3,
    )
    assert polarizability.value.shape == (1, 1)
    assert polarizability.units == unit.bohr**3
    assert polarizability.classification == "per_system"
    assert polarizability.name == "polarizability"
    assert polarizability.property_type == "polarizability"

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

    # various tests that should fail based on wrong dimensions or units
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[[1.0, 1.0, 1.0, 2.0], [2.0, 2.0, 2.0, 3.0]]], units="nanometer"
        )
    with pytest.raises(ValueError):
        positions = Positions(value=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], units="meter")
    # not units! we don't assume, must specify
    with pytest.raises(ValueError):
        positions = Positions(value=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([0.1]), units=unit.hartree)

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([[0.1, 0.3]]), units=unit.hartree)
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([1, 6]))
    # wrong shape
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6]]))

    # wrong shape
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6], [1, 6]]))

    # incompatible units
    with pytest.raises(ValueError):
        energies = Energies(
            value=np.array([[0.1], [0.3]]), units=unit.kilojoule_per_mole**2
        )
    # incompatible units
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units=unit.hartree
        )


def test_add_properties_to_records_directly():
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    assert "positions" in record.per_atom
    assert "energies" in record.per_system
    assert "smiles" in record.meta_data
    assert record.atomic_numbers is not None
    assert record.n_atoms == 2
    assert record.n_configs == 1
    assert record.validate() == True
    assert record._validate_n_atoms() == True
    assert record._validate_n_configs() == True

    with pytest.raises(ValueError):
        record.add_property(property=positions)

    record = Record(name="mol1", append_property=True)
    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    positions2 = Positions(
        value=[[[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]], units="nanometer"
    )

    record.add_properties([positions2, energies])

    assert record.n_configs == 2

    new_dataset = SourceDataset("test_dataset")
    new_dataset.add_record(record)

    assert "mol1" in new_dataset.records.keys()


def test_record_repr(capsys):
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    smiles = MetaData(name="smiles", value="[CH]")
    print(record)
    out, err = capsys.readouterr()

    assert "n_atoms: cannot be determined" in out
    assert "n_configs: cannot be determined" in out

    record.add_properties([positions, energies, atomic_numbers, smiles])

    print(record)
    out, err = capsys.readouterr()
    assert "name: mol1" in out
    assert "n_atoms: 2" in out
    assert "n_configs: 1" in out
    assert " per-atom properties: (['positions'])" in out
    assert " per-system properties: (['energies'])" in out
    assert " meta_data: (['smiles'])" in out
    assert " atomic_numbers" in out
    assert " name='atomic_numbers' value=array([[1]" in out
    assert (
        " [6]]) units=<Unit('dimensionless')> classification='atomic_numbers' property_type='atomic_numbers' n_configs=None n_atoms=2"
        in out
    )
    assert "name='positions' value=array([[[1., 1., 1.]" in out
    assert (
        "[2., 2., 2.]]]) units=<Unit('nanometer')> classification='per_atom' property_type='length' n_configs=1 n_atoms=2"
        in out
    )
    assert (
        "name='energies' value=array([[0.1]]) units=<Unit('hartree')> classification='per_system' property_type='energy' n_configs=1 n_atoms=None"
        in out
    )
    assert (
        " name='smiles' value='[CH]' units=<Unit('dimensionless')> classification='meta_data' property_type='meta_data' n_configs=None n_atoms=None"
        in out
    )
    print(record)


def test_record_to_dict():
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    smiles = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, smiles])
    record_dict = record.to_dict()

    assert record_dict["name"] == "mol1"
    assert record_dict["n_atoms"] == 2
    assert record_dict["n_configs"] == 1
    assert np.all(record_dict["atomic_numbers"].value == atomic_numbers.value)
    assert np.all(record_dict["per_atom"]["positions"].value == positions.value)
    assert np.all(record_dict["per_system"]["energies"].value == energies.value)


def test_record_validation():
    record = Record(name="mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])
    assert record._validate_n_configs() == True
    assert record._validate_n_atoms() == True
    assert record.validate() == True

    # this will fail because we will have different number of n_configs
    # note failure doesn't raise an error, but logs a warning and returns False
    record2 = Record(name="mol2")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1], [0.2]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    record2.add_properties([positions, energies, atomic_numbers])

    assert record2._validate_n_configs() == False
    assert record2.validate() == False

    # this will fail because we will have different number of n_atoms
    record3 = Record(name="mol3")
    positions = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]], units="nanometer"
    )
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    record3.add_properties([positions, energies, atomic_numbers])
    assert record3._validate_n_atoms() == False
    assert record3.validate() == False

    # this will fail because we haven't set atomic numbers

    record4 = Record(name="mol4")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)

    record4.add_properties([positions, energies])
    assert record4._validate_n_atoms() == False
    assert record4.validate() == False
    # this will fail because we don't have any properties that will dictate number of configs
    record5 = Record(name="mol5")
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    record5.add_property(atomic_numbers)

    assert record5._validate_n_configs() == False
    assert record5.validate() == False


def test_add_properties_failures():
    # test to ensure we can't add the same property multiple times
    record = Record(name="mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])
    with pytest.raises(ValueError):
        record.add_property(positions)
    with pytest.raises(ValueError):
        record.add_property(energies)
    with pytest.raises(ValueError):
        record.add_property(atomic_numbers)
    with pytest.raises(ValueError):
        record.add_property(meta_data)

    # we cannot have any property with the name "atomic_numbers" as it is reserved
    # so let us set up a bunch with that name and try to set them to a new record
    record = Record(name="mol1")
    meta_data = MetaData(name="atomic_numbers", value="[1,2]")
    positions = Positions(
        name="atomic_numbers",
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
        units="nanometer",
    )
    energies = Energies(
        name="atomic_numbers", value=np.array([[0.1]]), units=unit.hartree
    )
    atomic_numbers = AtomicNumbers(name="atomic_numbers", value=np.array([[1], [6]]))

    with pytest.raises(ValueError):
        record.add_property(meta_data)
    with pytest.raises(ValueError):
        record.add_property(positions)
    with pytest.raises(ValueError):
        record.add_property(energies)


def test_add_properties():
    new_dataset = SourceDataset("test_dataset")
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])

    record = new_dataset.get_record("mol1")
    assert "positions" in record.per_atom
    assert "energies" in record.per_system
    assert "smiles" in record.meta_data
    assert record.atomic_numbers is not None

    assert np.all(record.atomic_numbers.value == atomic_numbers.value)
    assert np.all(record.per_atom["positions"].value == positions.value)
    assert np.all(record.per_system["energies"].value == energies.value)
    assert record.meta_data["smiles"].value == meta_data.value

    assert record.validate() == True
    assert record.n_atoms == 2
    assert record.n_configs == 1

    with pytest.raises(ValueError):
        new_dataset.add_properties("mol1", [positions, energies])

    with pytest.raises(ValueError):
        new_dataset.add_properties("mol1", [atomic_numbers])


def test_slicing_properties():
    record = Record(name="mol1")

    positions = Positions(
        value=[
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
            [[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]],
        ],
        units="nanometer",
    )
    energies = Energies(
        value=np.array([[0.1], [0.2], [0.3], [0.4]]), units=unit.hartree
    )
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    sliced1 = record.slice_record(0, 1)

    assert sliced1.n_configs == 1
    assert sliced1.per_system["energies"].value == [[0.1]]

    # check dataset level slicing, that just calls the record level slicing
    new_dataset = SourceDataset("test_dataset")
    new_dataset.add_record(record)

    sliced2 = new_dataset.slice_record("mol1", 0, 1)
    assert sliced2.n_configs == 1
    assert sliced2.per_system["energies"].value == [[0.1]]

    # let us try to break this by passing the record, not record name

    with pytest.raises(AssertionError):
        new_dataset.slice_record(record, 0, 1)


def test_counting_records():
    new_dataset = SourceDataset("test_dataset")

    new_dataset.create_record("mol1")
    new_dataset.create_record("mol2")
    new_dataset.create_record("mol3")
    new_dataset.create_record("mol4")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers])
    new_dataset.add_properties("mol2", [positions, energies, atomic_numbers])
    new_dataset.add_properties("mol3", [positions, energies, atomic_numbers])

    positions = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
        units="nanometer",
    )
    energies = Energies(value=np.array([[0.1], [0.2]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    new_dataset.add_properties("mol4", [positions, energies, atomic_numbers])

    assert new_dataset.total_records() == 4
    assert new_dataset.total_configs() == 5

    new_dataset.validate_records()


def test_append_properties():
    new_dataset = SourceDataset("test_dataset", append_property=True)

    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])

    record = new_dataset.get_record("mol1")
    assert record.n_atoms == 2
    assert record.n_configs == 1

    positions = Positions(value=[[[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.5]]), units=unit.hartree)

    new_dataset.add_properties("mol1", [positions, energies])
    record = new_dataset.get_record("mol1")

    assert record.n_atoms == 2
    assert record.n_configs == 2

    assert np.all(
        record.per_atom["positions"].value
        == np.array(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]]
        )
    )

    assert np.all(record.per_system["energies"].value == np.array([[0.1], [0.5]]))

    assert record.validate() == True

    with pytest.raises(AssertionError):
        positions22 = Positions(value=[[[3.0, 1.0, 1.0]]], units="nanometer")

        new_dataset.add_properties("mol1", [positions22])

    # modify the record and then call the update_record method
    new_pos = np.array([[[1.0, 1.0, 1.0], [3.0, 30, 3.0]]])
    record.per_atom["positions"].value = new_pos
    new_dataset.update_record(record)

    assert np.all(record.per_atom["positions"].value == new_pos)
    assert np.all(new_dataset.get_record("mol1").per_atom["positions"].value == new_pos)

    with pytest.raises(ValueError):
        new_dataset.add_property("mol1", atomic_numbers)

    # modify the name; this should fail because the record doesn't exist now
    # since search is done by name
    record.name = "mol2"
    with pytest.raises(ValueError):
        new_dataset.update_record(record)

    # since name changed, need to create the record
    new_dataset.add_record(record)

    assert "mol2" in new_dataset.records.keys()

    assert new_dataset.get_record("mol2")

    new_dataset.remove_record("mol2")
    assert "mol2" not in new_dataset.records.keys()

    record_new = Record(name="mol1", append_property=True)
    record_new.add_properties([positions, energies, atomic_numbers, meta_data])
    assert record_new.validate()

    record_new.add_properties([positions, energies])

    assert record_new.n_configs == 2
    with pytest.raises(ValueError):
        record_new.add_property(atomic_numbers)


def test_write_hdf5(prep_temp_dir):
    new_dataset = SourceDataset("test_dataset")
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])

    new_dataset.create_record("mol2")
    positions = Positions(
        value=[
            [[2.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            [[3.0, 1.0, 1.0], [5.0, 2.0, 2.0], [1.0, 3.0, 3.0]],
        ],
        units="nanometer",
    )
    energies = Energies(value=np.array([[0.2], [0.4]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6], [8]]))
    meta_data = MetaData(name="smiles", value="COH")
    new_dataset.add_properties("mol2", [positions, energies, atomic_numbers, meta_data])

    new_dataset.create_record("mol3")
    positions = Positions(value=[[[3.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.3]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")
    new_dataset.add_properties("mol3", [positions, energies, atomic_numbers, meta_data])

    checksum = new_dataset.to_hdf5(
        file_path=str(prep_temp_dir), file_name="test_dataset.hdf5"
    )

    new_dataset.summary_to_json(
        file_path=str(prep_temp_dir),
        file_name="test_dataset.json",
        hdf5_checksum=checksum,
        hdf5_file_name="test_dataset.hdf5",
    )

    import os

    assert os.path.exists(str(prep_temp_dir / "test_dataset.hdf5")) == True
    assert os.path.exists(str(prep_temp_dir / "test_dataset.json")) == True

    import h5py

    with h5py.File(str(prep_temp_dir / "test_dataset.hdf5"), "r") as f:
        assert "mol1" in f.keys()
        assert "mol2" in f.keys()
        assert "mol3" in f.keys()

        mol1 = f["mol1"]
        assert "atomic_numbers" in mol1.keys()
        assert "positions" in mol1.keys()
        assert "energies" in mol1.keys()
        assert "smiles" in mol1.keys()

        assert mol1["atomic_numbers"].shape == (2, 1)
        assert mol1["positions"].shape == (1, 2, 3)
        assert mol1["positions"].attrs["u"] == "nanometer"
        assert mol1["positions"].attrs["format"] == "per_atom"
        assert mol1["positions"].attrs["property_type"] == "length"

        assert mol1["energies"].shape == (1, 1)
        assert mol1["energies"].attrs["u"] == "kilojoule_per_mole"
        assert mol1["energies"].attrs["format"] == "per_system"
        assert mol1["energies"].attrs["property_type"] == "energy"

        assert mol1["smiles"][()].decode("utf-8") == "[CH]"

        mol2 = f["mol2"]
        assert "atomic_numbers" in mol2.keys()
        assert "positions" in mol2.keys()
        assert "energies" in mol2.keys()
        assert "smiles" in mol2.keys()

        assert mol2["atomic_numbers"].shape == (3, 1)
        assert mol2["positions"].shape == (2, 3, 3)
        assert mol2["energies"].shape == (2, 1)
        assert mol2["smiles"][()].decode("utf-8") == "COH"

        mol3 = f["mol3"]
        assert "atomic_numbers" in mol3.keys()
        assert "positions" in mol3.keys()
        assert "energies" in mol3.keys()
        assert "smiles" in mol3.keys()

        assert mol3["atomic_numbers"].shape == (2, 1)
        assert mol3["positions"].shape == (1, 2, 3)
        assert mol3["energies"].shape == (1, 1)
        assert mol3["smiles"][()].decode("utf-8") == "[CH]"

    import json

    with open(str(prep_temp_dir / "test_dataset.json"), "r") as f:
        data = json.load(f)
        assert data["dataset_name"] == "test_dataset"
        assert data["total_records"] == new_dataset.total_records()
        assert data["total_configurations"] == new_dataset.total_configs()
        assert data["md5_checksum"] == checksum
        assert data["filename"] == "test_dataset.hdf5"


def test_unit_system():

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

    GlobalUnitSystem.length = unit.angstrom
    assert GlobalUnitSystem.length == unit.angstrom

    GlobalUnitSystem.set_global_units("pressure", unit.bar)
    assert GlobalUnitSystem.get_units("pressure") == unit.bar

    GlobalUnitSystem.set_global_units("length", unit.nanometer)
    assert GlobalUnitSystem.length == unit.nanometer


def test_dataset_validation():
    new_dataset = SourceDataset("test_dataset")
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])

    assert new_dataset.validate_records() == True

    new_dataset.create_record("mol2")
    assert new_dataset.validate_records() == False
    assert new_dataset.validate_record("mol1") == True
    assert new_dataset.validate_record("mol2") == False

    new_dataset.add_property("mol2", positions)
    assert new_dataset.validate_record("mol2") == False
    new_dataset.add_property("mol2", atomic_numbers)
    assert new_dataset.validate_record("mol2") == False
    new_dataset.add_property("mol2", energies)
    assert new_dataset.validate_record("mol2") == True
    assert new_dataset.validate_records() == True

import pytest
import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset
from modelforge.curate.units import GlobalUnitSystem
from modelforge.curate.properties import *


def test_source_dataset_init(prep_temp_dir):
    new_dataset = SourceDataset(
        name="test_dataset1",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset1.sqlite",
    )
    assert new_dataset.name == "test_dataset1"

    new_dataset.create_record("mol1")
    assert "mol1" in new_dataset.records

    new_dataset.create_record("mol2")

    assert len(new_dataset.records) == 2


def test_automatic_naming_of_db(prep_temp_dir):

    new_dataset = SourceDataset(
        name="test_dataset2",
        local_db_dir=str(prep_temp_dir),
    )
    assert new_dataset.local_db_name == "test_dataset2.sqlite"
    # test that we sub _ for spaces
    new_dataset = SourceDataset(
        name="test dataset2",
        local_db_dir=str(prep_temp_dir),
    )
    assert new_dataset.local_db_name == "test_dataset2.sqlite"


def test_dataset_create_record(prep_temp_dir):

    new_dataset = SourceDataset(
        name="test_dataset2",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset2.sqlite",
    )
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
    record_from_db = new_dataset.get_record("mol3")
    assert record_from_db.name == "mol3"

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

    # create record with properties
    new_dataset.create_record("mol5", properties=[property])
    assert "mol5" in new_dataset.records

    # test adding multiple records
    record6 = Record(name="mol6")
    record7 = Record(name="mol7")

    new_dataset.add_records([record6, record7])
    assert "mol6" in new_dataset.records
    assert "mol7" in new_dataset.records
    record6_from_db = new_dataset.get_record("mol6")
    assert record6_from_db.name == "mol6"
    record7_from_db = new_dataset.get_record("mol7")
    assert record7_from_db.name == "mol7"
    # try adding again, it should fail
    with pytest.raises(ValueError):
        new_dataset.add_records([record6, record7])

    # test adding somethign that is not a record
    with pytest.raises(ValueError):
        new_dataset.add_records([1, 2, 3])
    with pytest.raises(ValueError):
        new_dataset.add_record(1)


def test_convert_record_units():
    record = Record(name="mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="angstrom")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    record.add_properties([positions, energies, atomic_numbers])

    record.convert_to_global_unit_system()
    assert record.get_property("positions").units == "nanometer"
    assert np.allclose(
        record.get_property("positions").value,
        np.array([[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]),
    )
    assert record.get_property("energies").units == "kilojoule_per_mole"
    assert np.allclose(record.get_property("energies").value, np.array([[262.5499639]]))


def test_add_properties_to_records_directly(prep_temp_dir):
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

    assert "positions" in record.keys()
    assert "energies" in record.keys()
    assert "smiles" in record.keys()
    assert "atomic_numbers" in record.keys()

    with pytest.raises(ValueError):
        record.add_property(property=positions)

    # test return of keys where we dont' add atomic numbers
    record = Record(name="mol1_no_atom")
    record.add_properties([positions, energies, meta_data])
    assert "positions" in record.keys()
    assert "energies" in record.keys()
    assert "smiles" in record.keys()
    assert "atomic_numbers" not in record.keys()

    record = Record(name="mol1", append_property=True)
    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    positions2 = Positions(
        value=[[[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]], units="nanometer"
    )

    record.add_properties([positions2, energies])

    assert record.n_configs == 2

    new_dataset = SourceDataset(
        name="test_dataset3",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset3.sqlite",
    )
    new_dataset.add_record(record)

    assert "mol1" in new_dataset.records.keys()

    # add a property when a record hasn't already been created
    new_dataset.add_property("mol3", atomic_numbers)
    assert "mol3" in new_dataset.records.keys()
    assert new_dataset.get_record("mol3").atomic_numbers is not None


def test_record_failures():
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    # this will fail because the property already exists
    with pytest.raises(ValueError):
        record.add_property(energies)

    # this will fail because the property already exists, but with different type
    with pytest.raises(ValueError):
        positions = Positions(
            name="energies",
            value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
            units="nanometer",
        )
        record.add_property(positions)

    print(record.per_system.keys())
    # this will fail because the property already exists, but with different type
    with pytest.raises(ValueError):
        energies = Energies(
            name="positions", value=np.array([[0.1]]), units=unit.hartree
        )
        record.add_property(energies)

    # this will fail because the property already exists, but with different type
    with pytest.raises(ValueError):
        positions = Positions(
            name="atomic_numbers",
            value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
            units="nanometer",
        )
        record.add_property(positions)


def test_record_to_rdkit():
    record = Record(name="mol1")

    positions = Positions(
        value=[[[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]], [[2.1, 1.0, 1.0], [2.0, 1.0, 1.0]]],
        units="nanometer",
    )
    energies = Energies(value=np.array([[0.1], [0.2]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])

    rdkit_mol = record.to_rdkit()
    assert rdkit_mol.GetNumAtoms() == 2
    assert rdkit_mol.GetNumConformers() == 2
    print(rdkit_mol.GetConformer(0).GetAtomPosition(0))
    assert np.allclose(
        np.array(rdkit_mol.GetConformer(0).GetAtomPosition(0)),
        np.array([10.0, 10.0, 10.0]),
    )
    assert np.allclose(
        np.array(rdkit_mol.GetConformer(0).GetAtomPosition(1)),
        np.array([11.0, 10.0, 10.0]),
    )
    assert np.allclose(
        np.array(rdkit_mol.GetConformer(1).GetAtomPosition(0)),
        np.array([21.0, 10.0, 10.0]),
    )
    assert np.allclose(
        np.array(rdkit_mol.GetConformer(1).GetAtomPosition(1)),
        np.array([20.0, 10.0, 10.0]),
    )

    assert rdkit_mol.GetNumBonds() == 1

    rdkit_mol = record.to_rdkit(infer_bonds=False)
    assert rdkit_mol.GetNumBonds() == 0

    # test that we can't convert to rdkit if we don't have positions
    record = Record(name="mol1")
    record.add_properties([energies, atomic_numbers, meta_data])
    with pytest.raises(ValueError):
        record.to_rdkit()

    # we cannot convert without atomic numbers
    record = Record(name="mol1")
    record.add_properties([positions, energies, meta_data])
    with pytest.raises(ValueError):
        record.to_rdkit()


def test_record_remove_property():
    record = Record(name="mol1")
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    meta_data = MetaData(name="smiles", value="[CH]")
    record.add_properties([positions, energies, atomic_numbers, meta_data])

    record.remove_property("positions")
    assert "positions" not in record.keys()
    assert "positions" not in record.per_atom.keys()
    record.remove_property("energies")
    assert "energies" not in record.keys()
    assert "energies" not in record.per_system.keys()

    record.remove_property("smiles")
    assert "smiles" not in record.keys()
    assert "smiles" not in record.meta_data.keys()

    record.remove_property("atomic_numbers")
    assert "atomic_numbers" not in record.keys()
    assert record.atomic_numbers is None

    with pytest.raises(ValueError):
        record.remove_property("positions")


def test_get_property():
    record = Record(name="mol1")
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])

    pos = record.get_property("positions")
    assert np.all(pos.value == positions.value)
    assert pos.units == positions.units
    assert pos.name == positions.name

    atomic = record.get_property("atomic_numbers")
    assert np.all(atomic.value == atomic_numbers.value)
    assert atomic.units == atomic_numbers.units
    assert atomic.name == atomic_numbers.name

    en = record.get_property("energies")
    assert np.all(en.value == energies.value)
    assert en.units == energies.units
    assert en.name == energies.name

    smiles = record.get_property("smiles")
    assert smiles.value == meta_data.value
    assert smiles.units == meta_data.units
    assert smiles.name == meta_data.name

    with pytest.raises(ValueError):
        record.get_property("non_existent")


def test_get_property_value():
    record = Record(name="mol1")
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])

    pos = record.get_property_value("positions")
    assert np.all(pos.m == positions.value)
    assert pos.units == positions.units

    en = record.get_property_value("energies")
    assert np.all(en.m == energies.value)
    assert en.units == energies.units

    atomic = record.get_property_value("atomic_numbers")
    assert np.all(atomic == atomic_numbers.value)

    smiles = record.get_property_value("smiles")
    assert smiles == meta_data.value

    with pytest.raises(ValueError):
        record.get_property_value("non_existent")


def test_infer_bonds_and_length():
    from modelforge.curate.record import infer_bonds, calculate_max_bond_length_change

    record = Record(name="mol1")
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    positions = Positions(
        value=[[[1.0, 1.0, 1.0], [1.1, 1.0, 1.0]], [[2.2, 1.0, 1.0], [2.0, 1.0, 1.0]]],
        units="nanometer",
    )
    energies = Energies(value=np.array([[0.1], [0.2]]), units=unit.hartree)

    record.add_properties([positions, energies, atomic_numbers])

    bonds = infer_bonds(record)

    assert len(bonds) == 1
    assert bonds[0][0] in [1, 0]
    assert bonds[0][1] in [1, 0]

    max_changes = calculate_max_bond_length_change(record, bonds=bonds)
    assert len(max_changes) == 2

    assert np.allclose(max_changes[0].m, 0)
    assert np.allclose(max_changes[1].m, 0.1)


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
    # try adding the same properties again
    with pytest.raises(ValueError):
        record.add_property(positions)
    with pytest.raises(ValueError):
        record.add_property(energies)
    with pytest.raises(ValueError):
        record.add_property(atomic_numbers)
    with pytest.raises(ValueError):
        record.add_property(meta_data)

    # try adding properties with same names, but different types, i.e., per_atom and per_system
    # energies is already added as per_system, so this will fail
    positions2 = Positions(
        name="energies", value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer"
    )

    with pytest.raises(ValueError):
        record.add_property(positions2)

    energies = Energies(name="positions", value=np.array([[0.1]]), units=unit.hartree)
    with pytest.raises(ValueError):
        record.add_property(energies)

    meta_data = MetaData(name="positions", value="[CH]")
    with pytest.raises(ValueError):
        record.add_property(meta_data)
    meta_data = MetaData(name="energies", value="[CH]")
    with pytest.raises(ValueError):
        record.add_property(meta_data)

    energies = Energies(name="smiles", value=np.array([[0.1]]), units=unit.hartree)
    with pytest.raises(ValueError):
        record.add_property(energies)
    positions = Positions(
        name="smiles", value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer"
    )
    with pytest.raises(ValueError):
        record.add_property(positions)

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


def test_add_properties(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset4",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset4.sqlite",
    )
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


def test_convert_dataset_to_global_unit_system(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset4",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset4.sqlite",
    )
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="angstrom")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])
    # test unit conversion
    new_dataset.convert_to_global_unit_system()
    assert new_dataset.get_record("mol1").get_property("positions").units == "nanometer"
    assert (
        new_dataset.get_record("mol1").get_property("energies").units
        == "kilojoule_per_mole"
    )


def test_slicing_properties(prep_temp_dir):
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
    new_dataset = SourceDataset(
        "test_dataset5",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset5.sqlite",
    )
    new_dataset.add_record(record)

    sliced2 = new_dataset.slice_record("mol1", 0, 1)
    assert sliced2.n_configs == 1
    assert sliced2.per_system["energies"].value == [[0.1]]

    # let us try to break this by passing the record, not record name

    with pytest.raises(AssertionError):
        new_dataset.slice_record(record, 0, 1)


def test_counting_records(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset6",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset6.sqlite",
    )

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

    assert new_dataset.keys() == ["mol1", "mol2", "mol3", "mol4"]


def test_append_properties(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset7",
        append_property=True,
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset7.sqlite",
    )

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


def test_recorder():
    import copy

    record1 = Record(name="mol1", append_property=True)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6], [8]]))
    positions = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]], units="nanometer"
    )

    record1.add_property(atomic_numbers)
    record1.add_property(positions)
    assert record1.n_atoms == 3

    record2 = copy.deepcopy(record1)

    # create a mapping that doesn't change anything
    mapping = [0, 1, 2]
    record2.reorder(mapping)
    assert record2.n_atoms == 3
    assert np.all(
        record2.per_atom["positions"].value == record1.per_atom["positions"].value
    )
    assert np.all(record2.atomic_numbers.value == record1.atomic_numbers.value)

    # create a mapping that changes the order
    mapping = [2, 1, 0]
    record2 = copy.deepcopy(record1)
    record2.reorder(mapping)
    assert record2.n_atoms == 3
    assert np.all(record2.atomic_numbers.value == np.array([[8], [6], [1]]))
    assert np.all(
        record2.per_atom["positions"].value
        == np.array([[[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]])
    )

    # let's add another config to the record to ensure this works for multiple configs

    record2 = copy.deepcopy(record1)
    record2.add_property(positions)

    assert record2.n_atoms == 3
    assert record2.n_configs == 2

    record2.reorder(mapping)
    assert np.all(record2.atomic_numbers.value == np.array([[8], [6], [1]]))
    assert np.all(
        record2.per_atom["positions"].value
        == np.array(
            [
                [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
                [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
            ]
        )
    )


def test_merge_records(prep_temp_dir):

    # merge function works basically the same as append function
    record1 = Record(name="mol1")
    positions1 = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer"
    )
    energies1 = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))

    meta_data1 = MetaData(name="smiles", value="[CH]")
    record1.add_properties([positions1, energies1, atomic_numbers1, meta_data1])
    assert record1.append_property == False

    record2 = Record(name="mol2")
    positions2 = Positions(
        value=[[[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]], units="nanometer"
    )
    energies2 = Energies(value=np.array([[0.5]]), units=unit.hartree)
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data2 = MetaData(name="smiles", value="[CH]")

    record2.add_properties([positions2, energies2, atomic_numbers2, meta_data2])

    # Merge records
    record1.merge(record2)

    assert record1.n_configs == 2
    assert record1.n_atoms == 2
    assert np.all(record1.per_atom["positions"].value[0] == positions1.value[0])
    assert np.all(record1.per_atom["positions"].value[1] == positions2.value[0])

    assert np.all(record1.per_system["energies"].value == np.array([[0.1], [0.5]]))
    assert record1.append_property == False

    # create a record with a different atomic_numbers
    record3 = Record(name="mol3")
    atomic_numbers3 = AtomicNumbers(value=np.array([[1], [8]]))
    record3.add_properties([positions1, energies1, atomic_numbers3, meta_data1])
    with pytest.raises(ValueError):
        record1.merge(record3)


def test_write_hdf5(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset8",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset8.sqlite",
    )
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.kilojoule_per_mole)
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
        assert data["name"] == "test_dataset8"
        assert data["total_records"] == new_dataset.total_records()
        assert data["total_configurations"] == new_dataset.total_configs()
        assert data["md5_checksum"] == checksum
        assert data["filename"] == "test_dataset.hdf5"

    # test that we can read the dataset back in

    from modelforge.curate.sourcedataset import create_dataset_from_hdf5

    new_dataset_read = create_dataset_from_hdf5(
        hdf5_filename=str(prep_temp_dir / "test_dataset.hdf5"),
        dataset_name="test_read_data",
        dataset_local_db_dir=str(prep_temp_dir),
    )

    assert new_dataset_read.total_records() == 3
    record_mo1 = new_dataset_read.get_record("mol1")
    assert record_mo1.n_configs == 1
    assert record_mo1.n_atoms == 2
    assert np.all(
        record_mo1.per_atom["positions"].value
        == np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
    )
    assert np.all(record_mo1.per_system["energies"].value == np.array([[0.1]]))
    assert np.all(record_mo1.atomic_numbers.value == np.array([[1], [6]]))
    assert record_mo1.meta_data["smiles"].value == "[CH]"


def test_dataset_validation(prep_temp_dir):
    new_dataset = SourceDataset(
        name="test_dataset9",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset9.sqlite",
    )

    assert new_dataset.local_db_dir == str(prep_temp_dir)
    assert new_dataset.local_db_name == "test_dataset9.sqlite"

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


def test_dataset_subsetting(prep_temp_dir):
    ds = SourceDataset(name="test_dataset10", local_db_dir=str(prep_temp_dir))

    assert ds.local_db_dir == str(prep_temp_dir)
    assert ds.local_db_name == "test_dataset10.sqlite"

    positions = Positions(
        value=[
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        ],
        units="nanometer",
    )
    energies = Energies(
        value=np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]), units=unit.hartree
    )
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    for i in range(10):
        ds.create_record(f"mol{i}")
        ds.add_properties(f"mol{i}", [positions, energies, atomic_numbers])

    assert ds.total_configs() == 50
    assert ds.total_records() == 10

    # check total records
    ds_subset = ds.subset_dataset(new_dataset_name="test_dataset_sub1", total_records=5)
    assert ds_subset.total_configs() == 25
    assert ds_subset.total_records() == 5

    assert ds_subset.name == "test_dataset_sub1"
    assert ds_subset.local_db_name == "test_dataset_sub1.sqlite"
    assert ds_subset.local_db_dir == ds.local_db_dir

    ds_subset = ds.subset_dataset(new_dataset_name="test_dataset_sub2", total_records=3)
    assert ds_subset.total_configs() == 15
    assert ds_subset.total_records() == 3

    # check total_records and max_configurations_per_record
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3",
        total_records=3,
        max_configurations_per_record=2,
    )
    assert ds_subset.total_configs() == 6
    assert ds_subset.total_records() == 3

    # check total_conformers
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub4", total_configurations=20
    )
    assert ds_subset.total_configs() == 20
    assert ds_subset.total_records() == 4

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub5",
        total_configurations=20,
        max_configurations_per_record=2,
    )
    assert ds_subset.total_configs() == 20
    assert ds_subset.total_records() == 10

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub6",
        total_configurations=20,
        max_configurations_per_record=6,
    )
    assert ds_subset.total_configs() == 20
    assert ds_subset.total_records() == 4

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub7",
        total_configurations=11,
        max_configurations_per_record=4,
    )
    assert ds_subset.total_configs() == 11
    assert ds_subset.total_records() == 3

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub8",
        total_configurations=11,
        max_configurations_per_record=5,
    )
    assert ds_subset.total_configs() == 11
    assert ds_subset.total_records() == 3

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub9",
        final_configuration_only=True,
    )
    assert ds_subset.total_configs() == 10
    assert ds_subset.total_records() == 10

    # check to ensure we fail when we should
    # this shoudl fail because the datasetname is the same
    with pytest.raises(ValueError):
        ds.subset_dataset(
            new_dataset_name="test_dataset10",
            total_records=5,
        )
    # this should fail if we set total_records and total_configurations
    with pytest.raises(ValueError):
        ds.subset_dataset(
            new_dataset_name="test_sub1",
            total_records=5,
            total_configurations=20,
        )
    # final_Configuration_only and max_configurations_per_record can't be set at the same time
    with pytest.raises(ValueError):
        ds.subset_dataset(
            new_dataset_name="test_sub2",
            total_records=5,
            final_configuration_only=True,
            max_configurations_per_record=5,
        )


def test_limit_atomic_numbers(prep_temp_dir):

    atomic_numbers = AtomicNumbers(
        value=np.array(
            [
                [8],
                [6],
                [6],
                [1],
            ]
        )
    )
    energies = Energies(
        value=np.array(
            [
                [-0.5],
            ]
        ),
        units=unit.kilojoule_per_mole,
    )
    positions = Positions(
        value=np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        units=unit.nanometer,
    )
    record = Record("mol1")
    record.add_properties([atomic_numbers, energies, positions])

    dataset = SourceDataset(name="test_dataset11", local_db_dir=str(prep_temp_dir))
    dataset.add_record(record)

    atomic_numbers_to_limit = np.array([8, 6, 1])

    assert record.contains_atomic_numbers(atomic_numbers_to_limit) == True
    new_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset11_sub",
        atomic_numbers_to_limit=atomic_numbers_to_limit,
    )

    assert new_dataset.total_records() == 1

    atomic_numbers_to_limit = np.array([8, 6])

    assert record.contains_atomic_numbers(atomic_numbers_to_limit) == False
    new_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset11_sub2",
        atomic_numbers_to_limit=atomic_numbers_to_limit,
    )
    assert new_dataset.total_records() == 0

    # test that we fail if we give the same name for a subset
    with pytest.raises(ValueError):
        dataset.subset_dataset(
            new_dataset_name="test_dataset11",
            atomic_numbers_to_limit=atomic_numbers_to_limit,
        )

    # create an empty record and try to limit the atomic numbers
    # it will fail
    with pytest.raises(ValueError):
        record_empty = Record("mol2")
        dataset.add_record(record_empty)
        dataset.subset_dataset(
            new_dataset_name="test_dataset11_sub3",
            atomic_numbers_to_limit=atomic_numbers_to_limit,
        )


def test_remove_high_force_configs(prep_temp_dir):

    atomic_numbers = AtomicNumbers(
        value=np.array(
            [
                [6],
                [1],
            ]
        )
    )
    energies = Energies(
        value=np.array(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
            ]
        ),
        units=unit.kilojoule_per_mole,
    )
    positions = Positions(
        value=np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ),
        units=unit.nanometer,
    )
    forces = Forces(
        value=np.array(
            [
                [
                    [10, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [20, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [30, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [40, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                [
                    [31, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ]
        ),
        units=unit.kilojoule_per_mole / unit.nanometer,
    )
    record = Record("mol1")
    record.add_properties([atomic_numbers, energies, positions, forces])

    # first check the record level removal of high energy configurations
    record_new = record.remove_high_force_configs(
        30 * unit.kilojoule_per_mole / unit.nanometer
    )
    assert record_new.n_configs == 3

    record_new = record.remove_high_force_configs(
        31 * unit.kilojoule_per_mole / unit.nanometer
    )
    assert record_new.n_configs == 4
    assert record_new.per_system["energies"].value.shape == (4, 1)
    assert np.all(
        record_new.per_system["energies"].value
        == np.array([[1.0], [2.0], [3.0], [5.0]])
    )

    # test filtering via the dataset

    dataset = SourceDataset(name="test_dataset12", local_db_dir=str(prep_temp_dir))
    dataset.add_record(record)

    # effectively the same tests as above, but done on the dataset level
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub1",
        total_configurations=5,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert np.all(
        trimmed_dataset.get_record("mol1").per_system["energies"].value
        == np.array([[1.0], [2.0], [3.0]])
    )

    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub2",
        total_configurations=5,
        max_force=31 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 4
    assert np.all(
        trimmed_dataset.get_record("mol1").per_system["energies"].value
        == np.array([[1.0], [2.0], [3.0], [5.0]])
    )

    # consider now including other restrictions on the number of configurations, records, etc.
    # limit the number of configurations in total
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub3",
        total_configurations=3,
        max_force=31 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert np.all(
        trimmed_dataset.get_record("mol1").per_system["energies"].value
        == np.array([[1.0], [2.0], [3.0]])
    )

    # total_configurations and max_configurations_per_record
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub4",
        total_configurations=5,
        max_configurations_per_record=3,
        max_force=31 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert np.all(
        trimmed_dataset.get_record("mol1").per_system["energies"].value
        == np.array([[1.0], [2.0], [3.0]])
    )

    # add a second record, but with different atomic numbers
    # to further test filtering

    atomic_numbers = AtomicNumbers(value=np.array([[8], [1]]))
    record2 = Record("mol2")
    record2.add_properties([atomic_numbers, energies, positions, forces])

    dataset.add_record(record2)

    #  limit total_configurations
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub5",
        total_configurations=5,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 2
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.get_record("mol2").n_configs == 2
    assert trimmed_dataset.total_configs() == 5

    # limit total_configurations and max_configurations_per_record
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub6",
        total_configurations=6,
        max_configurations_per_record=3,
        max_force=31 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 2
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.get_record("mol2").n_configs == 3
    assert trimmed_dataset.total_configs() == 6

    # Add in limiting of the atomic numbers
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub7",
        total_configurations=6,
        max_configurations_per_record=3,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
        atomic_numbers_to_limit=[6, 1],
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.total_configs() == 3
    assert "mol2" not in trimmed_dataset.records.keys()

    # same test but only restrictions on atomic_numbers and max_force
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub8",
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
        atomic_numbers_to_limit=[6, 1],
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.total_configs() == 3
    assert "mol2" not in trimmed_dataset.records.keys()

    # check toggling of total records
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub9",
        total_records=1,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.total_configs() == 3

    # check toggling of total records
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub10",
        total_records=2,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert trimmed_dataset.total_records() == 2
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.get_record("mol2").n_configs == 3
    assert trimmed_dataset.total_configs() == 6

    # make sure we can also exclude atomic numbers
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub11",
        total_records=2,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
        atomic_numbers_to_limit=[6, 1],
    )
    assert trimmed_dataset.total_records() == 1
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.total_configs() == 3

    # case where our atomic number filtering captures everything
    trimmed_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub12",
        total_records=2,
        max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
        atomic_numbers_to_limit=[6, 1, 8],
    )
    assert trimmed_dataset.total_records() == 2
    assert trimmed_dataset.get_record("mol1").n_configs == 3
    assert trimmed_dataset.get_record("mol2").n_configs == 3
    assert trimmed_dataset.total_configs() == 6

    # this will fail because we are going to try a key that doesn't exist
    with pytest.raises(ValueError):
        trimmed_dataset = dataset.subset_dataset(
            new_dataset_name="test_dataset12_sub13",
            total_records=2,
            max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
            atomic_numbers_to_limit=[6, 1, 8, 9],
            max_force_key="forces_that_do_not_exist",
        )
    # this should fail because we are giving the energy key
    with pytest.raises(ValueError):
        trimmed_dataset = dataset.subset_dataset(
            new_dataset_name="test_dataset12_sub14",
            total_records=2,
            max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
            atomic_numbers_to_limit=[6, 1, 8],
            max_force_key="energies",
        )
    # this should fail because we are giving a key that isn't force but is per atom
    with pytest.raises(ValueError):
        trimmed_dataset = dataset.subset_dataset(
            new_dataset_name="test_dataset12_sub15",
            total_records=2,
            max_force=30 * unit.kilojoule_per_mole / unit.nanometer,
            atomic_numbers_to_limit=[6, 1, 8],
            max_force_key="positions",
        )


def test_reading_from_db_file(prep_temp_dir):
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    smiles = MetaData(name="smiles", value="[CH]")

    record.add_properties([positions, energies, atomic_numbers, smiles])

    ds = SourceDataset(
        name="test_dataset14",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset14.sqlite",
    )
    ds.add_record(record)

    assert ds.total_records() == 1
    assert "mol1" in ds.records.keys()

    # now let us read from the file
    ds2 = SourceDataset(
        name="test_dataset15",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset14.sqlite",
        read_from_local_db=True,
    )
    assert ds2.total_records() == 1
    assert "mol1" in ds2.records.keys()
    record_from_db = ds2.get_record("mol1")
    assert record_from_db.validate() == True
    assert record_from_db.n_atoms == 2
    assert record_from_db.n_configs == 1
    assert np.all(record_from_db.per_system["energies"].value == energies.value)
    assert np.all(record_from_db.per_atom["positions"].value == positions.value)
    assert np.all(record_from_db.atomic_numbers.value == atomic_numbers.value)
    assert record_from_db.meta_data["smiles"].value == smiles.value

    # this will fail because the db file doesn't exist
    with pytest.raises(FileNotFoundError):
        ds3 = SourceDataset(
            name="test_dataset16",
            local_db_dir=str(prep_temp_dir),
            local_db_name="test_dataset_does_not_exist.sqlite",
            read_from_local_db=True,
        )
    # let us try to reinitialize a dataset that exists.  it will remove it
    ds = SourceDataset(
        name="test_dataset14",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset14.sqlite",
    )
    import os

    assert os.path.exists(str(prep_temp_dir / "test_dataset14.sqlite")) == False

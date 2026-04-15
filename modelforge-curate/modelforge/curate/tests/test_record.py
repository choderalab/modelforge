import pytest
import numpy as np
from openff.units import unit

from modelforge.curate import (
    Record,
    SourceDataset,
    AtomicNumbers,
    RecordGroup,
    Energies,
    PartialCharges,
)
from modelforge.curate.properties import *


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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

    record.add_properties([positions, energies, atomic_numbers, meta_data])

    rdkit_mol = record.to_rdkit()
    assert rdkit_mol.GetNumAtoms() == 2
    assert rdkit_mol.GetNumConformers() == 2
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
    meta_data = MetaData(name="smiles", value="[CH+3]")
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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    smiles = MetaData(name="smiles", value="[CH+3]")
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
        " name='smiles' value='[CH+3]' units=<Unit('dimensionless')> classification='meta_data' property_type='meta_data' n_configs=None n_atoms=None"
        in out
    )


def test_record_to_dict():
    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    smiles = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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

    meta_data = MetaData(name="positions", value="[CH+3]")
    with pytest.raises(ValueError):
        record.add_property(meta_data)
    meta_data = MetaData(name="energies", value="[CH+3]")
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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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


def test_record_reorder():
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

    meta_data1 = MetaData(name="smiles", value="[CH+3]")
    record1.add_properties([positions1, energies1, atomic_numbers1, meta_data1])
    assert record1.append_property == False

    record2 = Record(name="mol2")
    positions2 = Positions(
        value=[[[3.0, 1.0, 1.0], [4.0, 2.0, 2.0]]], units="nanometer"
    )
    energies2 = Energies(value=np.array([[0.5]]), units=unit.hartree)
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data2 = MetaData(name="smiles", value="[CH+3]")

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


def test_record_group_single_configurations(prep_temp_dir):
    # this will test grouping individual records into a RecordGroup instance

    # set up two records with equal number of atoms per system
    # keep it simple to start with only a single configuration in each record
    record1 = Record(name="mol1")

    positions1 = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="angstrom")
    energies1 = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))
    smiles1 = MetaData(name="smiles", value="[CH+3]")
    record1.add_properties([positions1, energies1, atomic_numbers1, smiles1])

    record2 = Record(name="mol2")

    positions2 = Positions(value=[[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]], units="angstrom")
    energies2 = Energies(value=np.array([[0.2]]), units=unit.hartree)
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [8]]))
    smiles2 = MetaData(name="smiles", value="[OH+]")
    record2.add_properties([positions2, energies2, atomic_numbers2, smiles2])

    record_group = RecordGroup(name="group1")

    # add one record and check that we have what we expect
    record_group.add_record(record1)

    # when we add a record, we set the atomic numbers
    assert record_group.atomic_numbers is not None
    # we do reshape it to be a 3d array
    assert len(record_group.atomic_numbers.value.shape) == 3
    assert record_group.atomic_numbers.value.shape[0] == 1
    assert record_group.atomic_numbers.value.shape[1] == 2
    assert record_group.atomic_numbers.value.shape[2] == 1

    # check the energies, i.e. a per_system property
    assert record_group.per_system is not None
    assert len(record_group.per_system["energies"].value.shape) == 2
    assert record_group.per_system["energies"].value.shape[0] == 1
    assert record_group.per_system["energies"].value.shape[1] == 1
    assert np.all(record_group.per_system["energies"].value[0][0] == energies1.value)

    # check the positions, i.e., a per_atom property
    assert record_group.per_atom is not None
    assert len(record_group.per_atom["positions"].value.shape) == 3
    assert record_group.per_atom["positions"].value.shape[0] == 1
    assert record_group.per_atom["positions"].value.shape[1] == 2
    assert record_group.per_atom["positions"].value.shape[2] == 3

    assert np.all(record_group.per_atom["positions"].value == positions1.value)

    # check the smiles metadata
    # the record_group instance stores the smile string in a list, even if only one entry
    # the original property is just the string
    assert len(record_group.meta_data["smiles"].value) == 1
    assert record_group.meta_data["smiles"].value[0] == smiles1.value

    # check the grouped_names and grouped_n_configs fields
    assert len(record_group.grouped_names) == 1
    assert record_group.grouped_names[0] == "mol1"
    assert len(record_group.grouped_n_configs) == 1
    assert record_group.grouped_n_configs[0] == 1

    # add the second group
    record_group.add_record(record2)

    # now check that we have updated the shape of atomic_numbers
    # we should now have 2 entries in the first column
    assert record_group.atomic_numbers.value.shape[0] == 2
    assert record_group.atomic_numbers.value.shape[1] == 2
    assert record_group.atomic_numbers.value.shape[2] == 1
    assert np.all(record_group.atomic_numbers.value[0] == atomic_numbers1.value)
    assert np.all(record_group.atomic_numbers.value[1] == atomic_numbers2.value)

    # check the per_system property of energies
    # now that we have a second entry, the first index should be 2 (since each record has a single configuration)
    assert record_group.per_system["energies"].value.shape[0] == 2
    assert record_group.per_system["energies"].value.shape[1] == 1
    assert np.all(record_group.per_system["energies"].value[0] == energies1.value[0])
    assert np.all(record_group.per_system["energies"].value[1] == energies2.value[0])

    # check the per_atom property of positions

    assert record_group.per_atom["positions"].value.shape[0] == 2
    assert record_group.per_atom["positions"].value.shape[1] == 2
    assert record_group.per_atom["positions"].value.shape[2] == 3

    assert np.all(record_group.per_atom["positions"].value[0] == positions1.value)
    assert np.all(record_group.per_atom["positions"].value[1] == positions2.value)

    # check the metadata; because metadata is most likely not a numpy array that can just be stacked
    # we just create a list when they are added
    assert len(record_group.meta_data["smiles"].value) == 2
    assert record_group.meta_data["smiles"].value[0] == smiles1.value
    assert record_group.meta_data["smiles"].value[1] == smiles2.value

    # check the grouped_names and grouped_n_configs fields
    assert len(record_group.grouped_names) == 2
    assert record_group.grouped_names[0] == "mol1"
    assert record_group.grouped_names[1] == "mol2"

    assert len(record_group.grouped_n_configs) == 2
    assert record_group.grouped_n_configs[0] == 1
    assert record_group.grouped_n_configs[1] == 1

    # same test, but we will add both records simultaneously to test the wrapper
    # add one record and check that we have what we expect

    record_group = RecordGroup(name="group1")

    record_group.add_records([record1, record2])

    # now check that we have updated the shape of atomic_numbers
    # we should now have 2 entries in the first column
    print(record_group.atomic_numbers)

    assert record_group.atomic_numbers.value.shape[0] == 2
    assert record_group.atomic_numbers.value.shape[1] == 2
    assert record_group.atomic_numbers.value.shape[2] == 1
    assert np.all(record_group.atomic_numbers.value[0] == atomic_numbers1.value)
    assert np.all(record_group.atomic_numbers.value[1] == atomic_numbers2.value)

    # check the per_system property of energies
    # now that we have a second entry, the first index should be 2 (since each record has a single configuration)
    assert record_group.per_system["energies"].value.shape[0] == 2
    assert record_group.per_system["energies"].value.shape[1] == 1
    assert np.all(record_group.per_system["energies"].value[0] == energies1.value[0])
    assert np.all(record_group.per_system["energies"].value[1] == energies2.value[0])

    # check the per_atom property of positions

    assert record_group.per_atom["positions"].value.shape[0] == 2
    assert record_group.per_atom["positions"].value.shape[1] == 2
    assert record_group.per_atom["positions"].value.shape[2] == 3

    assert np.all(record_group.per_atom["positions"].value[0] == positions1.value)
    assert np.all(record_group.per_atom["positions"].value[1] == positions2.value)

    # check the metadata; because metadata is most likely not a numpy array that can just be stacked
    # we just create a list when they are added
    assert len(record_group.meta_data["smiles"].value) == 2
    assert record_group.meta_data["smiles"].value[0] == smiles1.value
    assert record_group.meta_data["smiles"].value[1] == smiles2.value

    # check the grouped_names and grouped_n_configs fields
    assert len(record_group.grouped_names) == 2
    assert record_group.grouped_names[0] == "mol1"
    assert record_group.grouped_names[1] == "mol2"

    assert len(record_group.grouped_n_configs) == 2
    assert record_group.grouped_n_configs[0] == 1
    assert record_group.grouped_n_configs[1] == 1


def test_record_group_multiple_configurations(prep_temp_dir):
    # similar test as above, but we will have multiple configurations per record to ensure things are correct
    # Since this is just a child class of Record, which is already well tested, we don't need much

    record1 = Record(name="mol1")

    positions1 = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]]],
        units="angstrom",
    )
    energies1 = Energies(value=np.array([[0.1], [0.11]]), units=unit.hartree)
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))
    smiles1 = MetaData(name="smiles", value="[CH+3]")
    record1.add_properties([positions1, energies1, atomic_numbers1, smiles1])

    record2 = Record(name="mol2")

    positions2 = Positions(
        value=[[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], [[3.2, 3.2, 3.2], [4.2, 4.2, 4.2]]],
        units="angstrom",
    )
    energies2 = Energies(value=np.array([[0.2], [0.22]]), units=unit.hartree)
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [8]]))
    smiles2 = MetaData(name="smiles", value="[OH+]")
    record2.add_properties([positions2, energies2, atomic_numbers2, smiles2])

    record_group = RecordGroup(name="group1")

    record_group.add_records([record1, record2])

    # now check that we have updated the shape of atomic_numbers
    # we should now have 2 entries in the first column
    assert record_group.atomic_numbers.value.shape[0] == 2
    assert record_group.atomic_numbers.value.shape[1] == 2
    assert record_group.atomic_numbers.value.shape[2] == 1
    assert np.all(record_group.atomic_numbers.value[0] == atomic_numbers1.value)
    assert np.all(record_group.atomic_numbers.value[1] == atomic_numbers2.value)

    # check the per_system property of energies
    # now that we have a second entry, the first index should be 2 (since each record has a single configuration)
    assert record_group.per_system["energies"].value.shape[0] == 4
    assert record_group.per_system["energies"].value.shape[1] == 1
    assert np.all(record_group.per_system["energies"].value[0] == energies1.value[0])
    assert np.all(record_group.per_system["energies"].value[1] == energies1.value[1])

    assert np.all(record_group.per_system["energies"].value[2] == energies2.value[0])
    assert np.all(record_group.per_system["energies"].value[3] == energies2.value[1])

    # check the per_atom property of positions

    assert record_group.per_atom["positions"].value.shape[0] == 4
    assert record_group.per_atom["positions"].value.shape[1] == 2
    assert record_group.per_atom["positions"].value.shape[2] == 3

    assert np.all(record_group.per_atom["positions"].value[0] == positions1.value[0])
    assert np.all(record_group.per_atom["positions"].value[1] == positions1.value[1])

    assert np.all(record_group.per_atom["positions"].value[2] == positions2.value[0])
    assert np.all(record_group.per_atom["positions"].value[3] == positions2.value[1])

    # check the metadata; because metadata is most likely not a numpy array that can just be stacked
    # we just create a list when they are added
    assert len(record_group.meta_data["smiles"].value) == 2
    assert record_group.meta_data["smiles"].value[0] == smiles1.value
    assert record_group.meta_data["smiles"].value[1] == smiles2.value

    # check the grouped_names and grouped_n_configs fields
    assert len(record_group.grouped_names) == 2
    assert record_group.grouped_names[0] == "mol1"
    assert record_group.grouped_names[1] == "mol2"

    assert len(record_group.grouped_n_configs) == 2
    assert record_group.grouped_n_configs[0] == 2
    assert record_group.grouped_n_configs[1] == 2


def test_grouped_record_errors(prep_temp_dir):
    # we need to throw errors if the records we are trying to group do not have the same properties

    record1 = Record(name="mol1")
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))

    # per_atom properties
    positions1 = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]]],
        units="angstrom",
    )
    charges1 = PartialCharges(
        value=np.array([[[1], [2]], [[1], [2]]]), units=unit.elementary_charge
    )

    # per system properties
    energies1 = Energies(
        value=np.array([[0.1], [0.11]]), units=unit.kilojoule_per_mole
    )
    energies_b1 = Energies(
        name="dft_energy",
        value=np.array([[0.1], [0.11]]),
        units=unit.kilojoule_per_mole,
    )

    # meta data
    smiles1 = MetaData(name="smiles", value="[CH+3]")
    temperature1 = MetaData(name="temperature", value=298.0, units=unit.kelvin)
    record1.add_properties(
        [
            positions1,
            energies1,
            atomic_numbers1,
            smiles1,
            temperature1,
            charges1,
            energies_b1,
        ]
    )
    # First set of tests should fail simply because we do not have the same number of
    # properties defined

    # this will be the same except not have energies_b1
    record2 = Record(name="mol2")
    record2.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, temperature1, charges1]
    )

    group = RecordGroup(name="group1")
    with pytest.raises(ValueError):
        group.add_records([record1, record2])

    # this will be the same except not have charges1
    record2 = Record(name="mol2")
    record2.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, temperature1, energies_b1]
    )

    with pytest.raises(ValueError):
        group.add_records([record1, record2])

    # this will be the same but without temperatuer 1
    record2 = Record(name="mol2")
    record2.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, charges1, energies_b1]
    )
    with pytest.raises(ValueError):
        group.add_records([record1, record2])

    # now define with same number of properties but with different names, i.e., we actually ensure properties are same

    # swap energies1 for energies_b1
    record2 = Record(name="mol2")
    record1 = Record(name="mol1")
    record1.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, temperature1, charges1]
    )
    record2.add_properties(
        [positions1, energies_b1, atomic_numbers1, smiles1, temperature1, charges1]
    )

    group = RecordGroup(name="group1")
    with pytest.raises(ValueError):
        group.add_records([record1, record2])

    # swap smiles1 for temperature1
    record2 = Record(name="mol2")
    record1 = Record(name="mol1")
    record1.add_properties([positions1, energies1, atomic_numbers1, smiles1, charges1])
    record2.add_properties(
        [positions1, energies1, atomic_numbers1, temperature1, charges1]
    )

    group = RecordGroup(name="group1")
    with pytest.raises(ValueError):
        group.add_records([record1, record2])

    # swap positions2 for charges1
    record2 = Record(name="mol2")
    record1 = Record(name="mol1")
    record1.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, temperature1]
    )
    record2.add_properties(
        [energies1, atomic_numbers1, smiles1, temperature1, charges1]
    )

    group = RecordGroup(name="group1")
    with pytest.raises(ValueError):
        group.add_records([record1, record2])


def test_grouped_record_round_trip(prep_temp_dir):
    record1 = Record(name="mol1")

    positions1 = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]]],
        units="angstrom",
    )
    energies1 = Energies(value=np.array([[0.1], [0.11]]), units=unit.hartree)
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))
    smiles1 = MetaData(name="smiles", value="[CH+3]")
    temperature1 = MetaData(name="temperature", value=298.0, units=unit.kelvin)
    record1.add_properties(
        [positions1, energies1, atomic_numbers1, smiles1, temperature1]
    )

    record2 = Record(name="mol2")

    positions2 = Positions(
        value=[[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], [[3.2, 3.2, 3.2], [4.2, 4.2, 4.2]]],
        units="angstrom",
    )
    energies2 = Energies(value=np.array([[0.2], [0.22]]), units=unit.hartree)
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [8]]))
    smiles2 = MetaData(name="smiles", value="[OH+]")
    temperature2 = MetaData(name="temperature", value=350.0, units=unit.kelvin)
    record2.add_properties(
        [positions2, energies2, atomic_numbers2, smiles2, temperature2]
    )

    record_group = RecordGroup(name="group1")

    record_group.add_records([record1, record2])

    record_list = record_group.get_records()

    for record in record_list:
        print(record)
    assert len(record_list) == 2

    assert record_list[0].name == record1.name
    assert record_list[1].name == record2.name

    assert np.all(record_list[0].atomic_numbers.value == record1.atomic_numbers.value)
    assert np.all(record_list[1].atomic_numbers.value == record2.atomic_numbers.value)

    assert np.all(
        record_list[0].per_system["energies"].value
        == record1.per_system["energies"].value
    )
    assert np.all(
        record_list[1].per_system["energies"].value
        == record2.per_system["energies"].value
    )

    assert np.all(
        record_list[0].per_atom["positions"].value
        == record1.per_atom["positions"].value
    )
    assert np.all(
        record_list[1].per_atom["positions"].value
        == record2.per_atom["positions"].value
    )

    assert record_list[0].meta_data["smiles"].value == record1.meta_data["smiles"].value
    assert record_list[1].meta_data["smiles"].value == record2.meta_data["smiles"].value

    assert (
        record_list[0].meta_data["temperature"].value
        == record1.meta_data["temperature"].value
    )
    assert (
        record_list[1].meta_data["temperature"].value
        == record2.meta_data["temperature"].value
    )

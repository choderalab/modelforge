import pytest
import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset, Energies, AtomicNumbers
from modelforge.utils.units import GlobalUnitSystem
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


def test_add_properties_to_record_in_dataset(prep_temp_dir):
    new_dataset = SourceDataset(
        "test_dataset4",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset4.sqlite",
    )
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])
    # test unit conversion
    new_dataset.convert_to_global_unit_system()
    assert new_dataset.get_record("mol1").get_property("positions").units == "nanometer"
    assert (
        new_dataset.get_record("mol1").get_property("energies").units
        == "kilojoule_per_mole"
    )


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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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

    # try appending with the wrong shaped atomic numbers
    # note appending atomic numbers doesn't raise an error unless the shape doesn't match
    # we just don't append the atomic_numbers. This seems more consistent when append_property is set to True
    with pytest.raises(ValueError):
        atomic_numbers2 = AtomicNumbers(value=np.array([[1], [6], [8]]))
        new_dataset.add_property("mol1", atomic_numbers2)

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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")
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

        assert mol1["smiles"][()].decode("utf-8") == "[CH+3]"

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
        assert mol3["smiles"][()].decode("utf-8") == "[CH+3]"

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
    assert record_mo1.meta_data["smiles"].value == "[CH+3]"


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
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    # test breaking up a dataset into smaller datasets that apply some filtering
    # for example, total_records or total_configurations, max_configurations_per_record
    # including strategies for picking configurations
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

    # check max_configurations_per_record with different strategies
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3a",
        max_configurations_per_record=2,
        max_configurations_per_record_order="start",
    )
    assert ds_subset.total_configs() == 20
    # grab a record and check the energy values to ensure we took the first 2 configurations
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.2]]))

    # check max_configurations_per_record with different strategies for total_records
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3a2",
        total_records=2,
        max_configurations_per_record=2,
        max_configurations_per_record_order="start",
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    # grab a record and check the energy values to ensure we took the first 2 configurations
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.2]]))

    # check max_configurations_per_record with different strategies for total_configurations
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3a3",
        total_configurations=4,
        max_configurations_per_record=2,
        max_configurations_per_record_order="start",
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    # grab a record and check the energy values to ensure we took the first 2 configurations
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.2]]))

    # check max_configurations_per_record with different strategies for total_configurations

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3b",
        max_configurations_per_record=2,
        max_configurations_per_record_order="end",
    )
    assert ds_subset.total_configs() == 20
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.4], [0.5]]))

    # check max_configurations_per_record with different strategies for total_records
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3b2",
        total_records=2,
        max_configurations_per_record=2,
        max_configurations_per_record_order="end",
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.4], [0.5]]))

    # check max_configurations_per_record with different strategies for total_configurations
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3b3",
        total_configurations=4,
        max_configurations_per_record=2,
        max_configurations_per_record_order="end",
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    record_temp = ds_subset.get_record("mol0")
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.4], [0.5]]))

    # check for random subsetting

    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3c",
        max_configurations_per_record=2,
        max_configurations_per_record_order="random",
        seed=57,
    )
    assert ds_subset.total_configs() == 20
    # grab a record and check the energy values to ensure we took 2 random configurations
    record_temp = ds_subset.get_record("mol0")
    assert record_temp.n_configs == 2
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.4]]))

    # check max_configurations_per_record with different strategies for total_records
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3c2",
        total_records=2,
        max_configurations_per_record=2,
        max_configurations_per_record_order="random",
        seed=57,
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    record_temp = ds_subset.get_record("mol0")
    assert record_temp.n_configs == 2
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.4]]))

    # check max_configurations_per_record with different strategies for total_configurations
    ds_subset = ds.subset_dataset(
        new_dataset_name="test_dataset_sub3c3",
        total_configurations=4,
        max_configurations_per_record=2,
        max_configurations_per_record_order="random",
        seed=57,
    )
    assert ds_subset.total_configs() == 4
    assert ds_subset.total_records() == 2
    record_temp = ds_subset.get_record("mol0")
    assert record_temp.n_configs == 2
    assert np.all(record_temp.per_system["energies"].value == np.array([[0.1], [0.4]]))

    # check that this fails if we give a bad value to max_configurations_per_record_order
    with pytest.raises(ValueError):
        ds_subset = ds.subset_dataset(
            new_dataset_name="test_dataset_sub3d",
            max_configurations_per_record=2,
            max_configurations_per_record_order="totally_wrong",
            seed=57,
        )
    with pytest.raises(ValueError):
        ds_subset = ds.subset_dataset(
            new_dataset_name="test_dataset_sub3e",
            total_records=1,
            max_configurations_per_record=2,
            max_configurations_per_record_order="totally_wrong",
        )

    with pytest.raises(ValueError):
        ds_subset = ds.subset_dataset(
            new_dataset_name="test_dataset_sub3f",
            total_configurations=10,
            max_configurations_per_record=2,
            max_configurations_per_record_order="totally_wrong",
        )
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


def test_limit_to_spin_multiplicity(prep_temp_dir):

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
                    [0.0, 1.5, 1.5],
                ],
            ]
        ),
        units=unit.nanometer,
    )
    spin_multiplicity = SpinMultiplicitiesPerSystem(
        value=np.array([[1], [2], [2], [3], [2]])
    )

    record = Record("mol1")
    record.add_properties([atomic_numbers, energies, positions, spin_multiplicity])

    dataset = SourceDataset(name="test_dataset_spin", local_db_dir=str(prep_temp_dir))
    dataset.add_record(record)

    # test that we can limit to a specific spin multiplicity
    new_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub",
        spin_multiplicity_to_limit=2,
    )

    # this should produce 1 record with 3 configurations
    assert new_dataset.total_records() == 1
    assert new_dataset.get_record("mol1").n_configs == 3

    assert np.all(
        new_dataset.get_record("mol1").per_system["energies"].value
        == np.array([[2.0], [3.0], [5.0]])
    )
    assert np.all(
        new_dataset.get_record("mol1").per_atom["positions"].value
        == np.array(
            [
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0], [0.0, 1.5, 1.5]],
            ]
        )
    )
    assert np.all(
        new_dataset.get_record("mol1")
        .per_system["spin_multiplicities_per_system"]
        .value
        == np.array([[2], [2], [2]])
    )

    # test with a different spin multiplicity
    new_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub2",
        spin_multiplicity_to_limit=1,
    )
    # this should produce 1 record with 1 configuration
    assert new_dataset.total_records() == 1
    assert new_dataset.get_record("mol1").n_configs == 1
    assert np.all(
        new_dataset.get_record("mol1").per_system["energies"].value == np.array([[1.0]])
    )
    assert np.all(
        new_dataset.get_record("mol1").per_atom["positions"].value
        == np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    )
    # make sure we have the right spin multiplicity
    assert np.all(
        new_dataset.get_record("mol1")
        .per_system["spin_multiplicities_per_system"]
        .value
        == np.array([[1]])
    )

    # test where we don't get any configurations
    new_dataset = dataset.subset_dataset(
        new_dataset_name="test_dataset12_sub3",
        spin_multiplicity_to_limit=90,
    )
    # this should produce 0 records
    assert new_dataset.total_records() == 0


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
    smiles = MetaData(name="smiles", value="[CH+3]")

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


def test_read_hdf5(prep_temp_dir):

    new_dataset = SourceDataset(
        "test_dataset_for_reading",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset_for_reading.sqlite",
    )
    new_dataset.create_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.kilojoule_per_mole)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH+3]")

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
    meta_data = MetaData(name="smiles", value="[CH+3]")
    new_dataset.add_properties("mol3", [positions, energies, atomic_numbers, meta_data])

    checksum = new_dataset.to_hdf5(
        file_path=str(prep_temp_dir), file_name="test_dataset_for_reading.hdf5"
    )

    from modelforge.curate.sourcedataset import create_dataset_from_hdf5

    # first read in without a property map
    input_dataset = create_dataset_from_hdf5(
        hdf5_filename=f"{prep_temp_dir}/test_dataset_for_reading.hdf5",
        dataset_name="test_read_data",
        dataset_local_db_dir=str(prep_temp_dir),
    )

    assert input_dataset.total_records() == 3
    assert input_dataset.get_record("mol1").n_configs == 1
    assert input_dataset.get_record("mol1").n_atoms == 2
    assert input_dataset.get_record("mol2").n_configs == 2
    assert input_dataset.get_record("mol2").n_atoms == 3
    assert input_dataset.get_record("mol3").n_configs == 1
    assert input_dataset.get_record("mol3").n_atoms == 2

    record_mo1 = input_dataset.get_record("mol1")
    pos = record_mo1.get_property("positions")
    assert pos.value.shape == (1, 2, 3)
    assert np.all(pos.value == np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]))
    assert pos.units == "nanometer"
    assert pos.classification == "per_atom"
    assert pos.property_type == "length"

    energy = record_mo1.get_property("energies")
    assert energy.value.shape == (1, 1)
    assert np.all(energy.value == np.array([[0.1]]))
    assert energy.units == "kilojoule_per_mole"
    assert energy.classification == "per_system"
    assert energy.property_type == "energy"

    # read in with a property map
    property_map = {
        "energies": Energies,
        "positions": Positions,
        "atomic_numbers": AtomicNumbers,
        "smiles": MetaData,
    }

    input_dataset_with_map = create_dataset_from_hdf5(
        hdf5_filename=f"{prep_temp_dir}/test_dataset_for_reading.hdf5",
        dataset_name="test_read_data",
        dataset_local_db_dir=str(prep_temp_dir),
        property_map=property_map,
    )

    assert input_dataset_with_map.total_records() == 3
    assert input_dataset_with_map.get_record("mol1").n_configs == 1
    assert input_dataset_with_map.get_record("mol1").n_atoms == 2
    assert input_dataset_with_map.get_record("mol2").n_configs == 2
    assert input_dataset_with_map.get_record("mol2").n_atoms == 3
    assert input_dataset_with_map.get_record("mol3").n_configs == 1
    assert input_dataset_with_map.get_record("mol3").n_atoms == 2

    record_mo1 = input_dataset_with_map.get_record("mol1")

    pos = record_mo1.get_property("positions")
    assert isinstance(pos, Positions)
    assert pos.value.shape == (1, 2, 3)
    assert np.all(pos.value == np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]))
    assert pos.units == "nanometer"
    assert pos.classification == "per_atom"
    assert pos.property_type == "length"

    energy = record_mo1.get_property("energies")
    assert isinstance(energy, Energies)
    assert energy.value.shape == (1, 1)
    assert np.all(energy.value == np.array([[0.1]]))
    assert energy.units == "kilojoule_per_mole"
    assert energy.classification == "per_system"
    assert energy.property_type == "energy"

    atomic_numbers = record_mo1.get_property("atomic_numbers")
    assert isinstance(atomic_numbers, AtomicNumbers)
    assert atomic_numbers.value.shape == (2, 1)
    assert np.all(atomic_numbers.value == np.array([[1], [6]]))

    smiles = record_mo1.get_property("smiles")
    assert isinstance(smiles, MetaData)
    assert smiles.value == "[CH+3]"


def test_property_factory(prep_temp_dir):
    from modelforge.curate.properties import PropertyFactory

    # note the factory isn't used at the moment, but may be in the future

    # test that we can create a property factory
    factory = PropertyFactory()

    # test that we can create a property from a string
    positions = factory.create_property(
        class_name="positions",
        name="positions_test",
        value=[[[1.0, 1.0, 1.0]]],
        units="nanometer",
    )
    assert isinstance(positions, Positions)
    assert positions.value.shape == (1, 1, 3)
    assert positions.name == "positions_test"
    assert np.all(positions.value == np.array([[[1.0, 1.0, 1.0]]]))
    assert positions.units == "nanometer"
    assert positions.classification == "per_atom"
    assert positions.property_type == "length"

    energies = factory.create_property(
        class_name="energies",
        name="energies_test",
        value=np.array([[0.1]]),
        units=unit.hartree,
    )
    assert isinstance(energies, Energies)
    assert energies.value.shape == (1, 1)
    assert energies.name == "energies_test"
    assert np.all(energies.value == np.array([[0.1]]))
    assert energies.units == "hartree"

    forces = factory.create_property(
        class_name="forces",
        name="forces_test",
        value=np.array([[[0.1, 0.2, 0.3]]]),
        units=unit.kilojoule_per_mole / unit.nanometer,
    )
    assert isinstance(forces, Forces)
    assert forces.name == "forces_test"
    assert forces.value.shape == (1, 1, 3)
    assert np.all(forces.value == np.array([[[0.1, 0.2, 0.3]]]))
    assert forces.units == "kilojoule_per_mole / nanometer"

    atomic_numbers = factory.create_property(
        class_name="atomic_numbers",
        name="atomic_numbers_name",
        value=np.array([[1], [6]]),
        units=unit.dimensionless,
    )
    assert isinstance(atomic_numbers, AtomicNumbers)
    assert atomic_numbers.name == "atomic_numbers_name"
    assert atomic_numbers.value.shape == (2, 1)
    assert np.all(atomic_numbers.value == np.array([[1], [6]]))

    total_charge = factory.create_property(
        class_name="total_charge",
        name="total_charge_name",
        value=np.array([[0]]),
        units=unit.elementary_charge,
    )
    assert isinstance(total_charge, TotalCharge)
    assert total_charge.name == "total_charge_name"
    assert total_charge.value.shape == (1, 1)
    assert np.all(total_charge.value == np.array([[0]]))
    assert total_charge.property_type == "charge"
    assert total_charge.classification == "per_system"

    spin_multiplicities = factory.create_property(
        class_name="spin_multiplicities_per_system",
        name="spin_multiplicities_name",
        value=np.array([[1]]),
        units=unit.dimensionless,
    )
    assert isinstance(spin_multiplicities, SpinMultiplicitiesPerSystem)
    assert spin_multiplicities.name == "spin_multiplicities_name"

    spin_multiplicities_per_atom = factory.create_property(
        class_name="spin_multiplicities_per_atom",
        name="spin_multiplicities_per_atom_name",
        value=np.array([[[1], [2]]]),
        units=unit.dimensionless,
    )
    assert isinstance(spin_multiplicities_per_atom, SpinMultiplicitiesPerAtom)
    assert spin_multiplicities_per_atom.name == "spin_multiplicities_per_atom_name"

    dipole_moment_per_system = factory.create_property(
        class_name="dipole_moment_per_system",
        name="dipole_moment_per_system_name",
        value=np.array([[0.0, 0.0, 0.0]]),
        units=unit.debye,
    )
    assert isinstance(dipole_moment_per_system, DipoleMomentPerSystem)
    assert dipole_moment_per_system.name == "dipole_moment_per_system_name"

    dipole_moment_per_atom = factory.create_property(
        class_name="dipole_moment_per_atom",
        name="dipole_moment_per_atom_name",
        value=np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
        units=unit.debye,
    )
    assert isinstance(dipole_moment_per_atom, DipoleMomentPerAtom)
    assert dipole_moment_per_atom.name == "dipole_moment_per_atom_name"

    quadrupole_moment_per_system = factory.create_property(
        class_name="quadrupole_moment_per_system",
        name="quadrupole_moment_per_system_name",
        value=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        units=unit.debye * unit.nanometer,
    )
    assert isinstance(quadrupole_moment_per_system, QuadrupoleMomentPerSystem)
    assert quadrupole_moment_per_system.name == "quadrupole_moment_per_system_name"

    quadrupole_moment_per_atom = factory.create_property(
        class_name="quadrupole_moment_per_atom",
        name="quadrupole_moment_per_atom_name",
        value=np.array([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        units=unit.debye * unit.nanometer,
    )
    assert isinstance(quadrupole_moment_per_atom, QuadrupoleMomentPerAtom)

    polarizability = factory.create_property(
        class_name="polarizability",
        name="polarizability_name",
        value=np.array([[0.0]]),
        units=unit.angstrom**3,
    )
    assert isinstance(polarizability, Polarizability)
    assert polarizability.name == "polarizability_name"
    bond_orders_per_atom = factory.create_property(
        class_name="bond_orders",
        name="bond_orders_per_atom_name",
        value=np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        units=unit.dimensionless,
    )
    assert isinstance(bond_orders_per_atom, BondOrders)
    assert bond_orders_per_atom.name == "bond_orders_per_atom_name"

    dipole_moment_scalar_per_system = factory.create_property(
        class_name="dipole_moment_scalar_per_system",
        name="dipole_moment_scalar_per_system_name",
        value=np.array([[0.0]]),
        units=unit.debye,
    )
    assert isinstance(dipole_moment_scalar_per_system, DipoleMomentScalarPerSystem)


def test_dataset_grouped_records_to_hdf5(prep_temp_dir):
    # this will test writing grouped records to hdf5 files

    # we will create a few different RecordGroup instances to ensure we are getting the correct behavior
    # create a group that has n=2, with 3 records added to it of various number of configurations
    # create a group with n=3, with 2 records added to it
    # create a group with n=5, with 1 record added to it

    record1 = Record(name="mol1_n2")

    positions1 = Positions(
        value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]]],
        units="nanometers",
    )
    energies1 = Energies(value=np.array([[0.1], [0.11]]), units=unit.kilojoule_per_mole)
    atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))
    smiles1 = MetaData(name="smiles", value="[CH+3]")
    temperature1 = MetaData(name="temperature", value=298.0, units=unit.kelvin)
    vector_property1 = MetaData(name="vector_property", value=[[1.0, 1.0, 1.0]])
    record1.add_properties(
        [
            positions1,
            energies1,
            atomic_numbers1,
            smiles1,
            temperature1,
            vector_property1,
        ]
    )

    record2 = Record(name="mol2_n2")

    positions2 = Positions(
        value=[
            [[11.0, 11.0, 11.0], [12.0, 12.0, 12.0]],
            [[11.1, 11.1, 11.1], [12.1, 12.1, 12.1]],
            [[11.2, 11.2, 11.2], [12.2, 12.2, 12.2]],
        ],
        units="nanometers",
    )

    energies2 = Energies(
        value=np.array([[0.2], [0.21], [0.22]]), units=unit.kilojoule_per_mole
    )
    atomic_numbers2 = AtomicNumbers(value=np.array([[1], [6]]))
    smiles2 = MetaData(name="smiles", value="[CH+3]")
    temperature2 = MetaData(name="temperature", value=325.0, units=unit.kelvin)
    vector_property2 = MetaData(name="vector_property", value=[[2.0, 2.0, 2.0]])
    record2.add_properties(
        [
            positions2,
            energies2,
            atomic_numbers2,
            smiles2,
            temperature2,
            vector_property2,
        ]
    )

    record3 = Record(name="mol3_n2")

    positions3 = Positions(
        value=[
            [[111.0, 111.0, 111.0], [112.0, 112.0, 112.0]],
        ],
        units="nanometers",
    )
    energies3 = Energies(value=np.array([[0.3]]), units=unit.kilojoule_per_mole)
    atomic_numbers3 = AtomicNumbers(value=np.array([[1], [8]]))
    smiles3 = MetaData(name="smiles", value="[OH+]")
    temperature3 = MetaData(name="temperature", value=350.0, units=unit.kelvin)
    vector_property3 = MetaData(name="vector_property", value=[[3.0, 3.0, 3.0]])

    record3.add_properties(
        [
            positions3,
            energies3,
            atomic_numbers3,
            smiles3,
            temperature3,
            vector_property3,
        ]
    )

    record4 = Record(name="mol4_n3")
    atomic_numbers4 = AtomicNumbers(value=np.array([[1], [8], [1]]))
    positions4 = Positions(
        value=np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]]),
        units=unit.nanometers,
    )
    energies4 = Energies(value=np.array([[0.4]]), units=unit.kilojoule_per_mole)
    smiles4 = MetaData(name="smiles", value="O")
    temperature4 = MetaData(name="temperature", value=354.0, units=unit.kelvin)
    vector_property4 = MetaData(name="vector_property", value=[[4.0, 4.0, 4.0]])

    record4.add_properties(
        [
            positions4,
            energies4,
            atomic_numbers4,
            smiles4,
            temperature4,
            vector_property4,
        ]
    )

    record5 = Record(name="mol5_n3")
    atomic_numbers5 = AtomicNumbers(value=np.array([[1], [8], [1]]))
    positions5 = Positions(
        value=np.array(
            [
                [[11.0, 11.0, 11.0], [12.0, 12.0, 12.0], [13.0, 13.0, 13.0]],
                [[11.1, 11.1, 11.1], [12.1, 12.1, 12.1], [13.1, 13.1, 13.1]],
            ]
        ),
        units=unit.nanometers,
    )
    energies5 = Energies(value=np.array([[0.5], [0.51]]), units=unit.kilojoule_per_mole)
    smiles5 = MetaData(name="smiles", value="O")
    temperature5 = MetaData(name="temperature", value=354.0, units=unit.kelvin)
    vector_property5 = MetaData(name="vector_property", value=[[5.0, 5.0, 5.0]])

    record5.add_properties(
        [
            positions5,
            energies5,
            atomic_numbers5,
            smiles5,
            temperature5,
            vector_property5,
        ]
    )

    record6 = Record(name="mol6_n5")
    atomic_numbers6 = AtomicNumbers(value=np.array([[1], [1], [1], [1], [6]]))
    positions6 = Positions(
        value=np.array(
            [
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    [
                        4.0,
                        4.0,
                        4.0,
                    ],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],
                ]
            ],
        ),
        units=unit.nanometers,
    )
    energies6 = Energies(value=np.array([[0.6]]), units=unit.kilojoule_per_mole)
    smiles6 = MetaData(name="smiles", value="C")
    temperature6 = MetaData(name="temperature", value=356.0)
    vector_property6 = MetaData(name="vector_property", value=[[6.0, 6.0, 6.0]])

    record6.add_properties(
        [
            positions6,
            energies6,
            atomic_numbers6,
            smiles6,
            temperature6,
            vector_property6,
        ]
    )

    dataset = SourceDataset(
        name="dataset_to_group",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset_group1.sqlite",
    )

    dataset.add_records([record1, record2, record3, record4, record5, record6])

    # write out the dataset
    checksum = dataset.to_hdf5(
        file_path=str(prep_temp_dir),
        file_name="test_dataset_grouped.hdf5",
        group_records=True,
    )

    # read in the dataset directly
    import h5py

    full_path = f"{str(prep_temp_dir)}/test_dataset_grouped.hdf5"

    with h5py.File(full_path, "r") as f:
        # read in the group_keys and check them
        group_keys = list(f.keys())

        # records are grouped with the scheme group_{n_atoms}
        # let us ensure that we only have the expected groups from the records above
        assert len(group_keys) == 3
        assert "group_2" in group_keys
        assert "group_3" in group_keys
        assert "group_5" in group_keys

        # look at group_n2 to ensure we have 3 entries we expect
        group_n2 = f["group_2"]
        atomic_numbers = group_n2["atomic_numbers"][()]
        assert atomic_numbers.shape[0] == 3  # we have 3 molecules
        assert atomic_numbers.shape[1] == 2  # we have 2 atoms per molecule
        assert atomic_numbers.shape[2] == 1

        assert np.all(atomic_numbers == np.array([[[1], [6]], [[1], [6]], [[1], [8]]]))
        # we set the format attribute to be the 'atomic_numbers_grouped' when they are grouped
        assert group_n2["atomic_numbers"].attrs["format"] == "atomic_numbers_grouped"

        energies = group_n2["energies"][()]
        assert (
            energies.shape[0] == 6
        )  # we have 6 total configurations in the 3 molecules
        assert energies.shape[1] == 1
        assert np.all(
            energies == np.array([[0.1], [0.11], [0.2], [0.21], [0.22], [0.3]])
        )
        grouped_indices = group_n2["grouped_indices"][()]
        assert np.all(
            grouped_indices
            == np.array(
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                ]
            )
        )
        assert group_n2["n_configs"][()] == 6
        assert np.all(group_n2["grouped_n_configs"][()] == np.array([2, 3, 1]))

        grouped_names = group_n2["grouped_names"][()]
        # need to convert this from bytes to strings
        grouped_names = [val.decode("utf-8") for val in grouped_names]

        assert "mol1_n2" in grouped_names
        assert "mol2_n2" in grouped_names
        assert "mol3_n2" in grouped_names
        assert len(grouped_names) == 3

        smiles = group_n2["smiles"][()]
        # convert from byte to string
        smiles = [val.decode("utf-8") for val in smiles]
        assert smiles[0] == "[CH+3]"
        assert smiles[1] == "[CH+3]"
        assert smiles[2] == "[OH+]"

        temperature = group_n2["temperature"][()]
        assert np.all(temperature == np.array([298.0, 325.0, 350.0]))

        vector_property = group_n2["vector_property"][()]
        assert vector_property.shape[0] == 3
        assert vector_property.shape[1] == 3

        assert np.all(
            vector_property
            == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        )

        group_names = ["group_2", "group_3", "group_5"]
        records_in_groups = [[record1, record2, record3], [record4, record5], [record6]]

        for i in range(len(group_names)):
            group_name = group_names[i]
            n_atoms = group_name.split("_")[-1]
            group = f[group_name]
            records_in_group = records_in_groups[i]

            atomic_numbers = group["atomic_numbers"][()]
            temperature = group["temperature"][()]
            smiles = group["smiles"][()]
            smiles = [val.decode("utf-8") for val in smiles]
            vector_property = group["vector_property"][()]
            grouped_names_hdf5 = group["grouped_names"][()]
            grouped_names_hdf5 = [val.decode("utf-8") for val in grouped_names_hdf5]
            # figure out the start and end points to split the numpy arrays

            grouped_indices = group["grouped_indices"][()]
            grouped_n_configs = group["grouped_n_configs"][()]

            start_id = 0
            for j in range(len(records_in_group)):

                # for atomic numbers and metadata, we can just index into array
                # as none of these support variablility as a function of configurations
                assert np.all(
                    atomic_numbers[j] == records_in_group[j].atomic_numbers.value
                )
                assert np.all(
                    temperature[j] == records_in_group[j].meta_data["temperature"].value
                )
                assert np.all(
                    smiles[j] == records_in_group[j].meta_data["smiles"].value
                )
                assert np.all(
                    vector_property[j]
                    == records_in_group[j].meta_data["vector_property"].value
                )
                assert records_in_group[j].name == grouped_names_hdf5[j]

                # for per_atom and per_system properties, we just need to figure out the indices for slicing
                end_id = grouped_n_configs[j] + start_id
                positions = group["positions"][()][start_id:end_id]
                assert np.all(
                    positions == records_in_group[j].per_atom["positions"].value
                )
                energies = group["energies"][()][start_id:end_id]
                assert np.all(
                    energies == records_in_group[j].per_system["energies"].value
                )

                start_id = end_id

        # for key in group_n2.keys():
        #     print(key, group_n2[key][()])

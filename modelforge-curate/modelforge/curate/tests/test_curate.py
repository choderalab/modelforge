import pytest
import numpy as np
from openff.units import unit

from modelforge.curate.curate import *


def test_source_dataset_init():
    new_dataset = SourceDataset("test_dataset")
    assert new_dataset.dataset_name == "test_dataset"

    new_dataset.add_record("mol1")
    assert "mol1" in new_dataset.records

    new_dataset.add_record("mol2")

    assert len(new_dataset.records) == 2


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

    # various tests that should fail based on wrong dimensions
    with pytest.raises(ValueError):
        positions = Positions(
            value=[[[1.0, 1.0, 1.0, 2.0], [2.0, 2.0, 2.0, 3.0]]], units="nanometer"
        )
    with pytest.raises(ValueError):
        positions = Positions(value=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    with pytest.raises(ValueError):
        energies = Energies(value=np.array([0.1]), units=unit.hartree)

    with pytest.raises(ValueError):
        energies = Energies(value=np.array([[0.1, 0.3]]), units=unit.hartree)

    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6]]))
    with pytest.raises(ValueError):
        atomic_numbers = AtomicNumbers(value=np.array([[1, 6], [1, 6]]))


def test_add_properties():
    new_dataset = SourceDataset("test_dataset")
    new_dataset.add_record("mol1")
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


def test_append_properties():
    new_dataset = SourceDataset("test_dataset", append_property=True)

    new_dataset.add_record("mol1")
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
        positions = Positions(value=[[[3.0, 1.0, 1.0]]], units="nanometer")

        new_dataset.add_properties("mol1", [positions])


def test_write_hdf5(prep_temp_dir):
    new_dataset = SourceDataset("test_dataset")
    new_dataset.add_record("mol1")
    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    new_dataset.add_properties("mol1", [positions, energies, atomic_numbers, meta_data])

    new_dataset.add_record("mol2")
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

    new_dataset.add_record("mol3")
    positions = Positions(value=[[[3.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.3]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")
    new_dataset.add_properties("mol3", [positions, energies, atomic_numbers, meta_data])

    new_dataset.to_hdf5(file_path=str(prep_temp_dir), file_name="test_dataset.hdf5")

    import os

    assert os.path.exists(str(prep_temp_dir / "test_dataset.hdf5")) == True

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
        assert mol1["energies"].shape == (1, 1)
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

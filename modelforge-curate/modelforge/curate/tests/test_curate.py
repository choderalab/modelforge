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

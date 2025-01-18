import pytest
import numpy as np
from openff.units import unit

from modelforge.curate.curate import *
from schnetpack.properties import dipole_moment, polarizability


def test_source_dataset_init():
    new_dataset = SourceDataset("test_dataset")
    assert new_dataset.dataset_name == "test_dataset"

    new_dataset.create_record("mol1")
    assert "mol1" in new_dataset.records

    new_dataset.create_record("mol2")

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

    dipole_moment = DipoleMoment(value=np.array([[0.1, 0.1, 0.1]]), units=unit.debye)
    assert dipole_moment.value.shape == (1, 3)
    assert dipole_moment.units == unit.debye
    assert dipole_moment.classification == "per_system"
    assert dipole_moment.name == "dipole_moment"
    assert dipole_moment.property_type == "dipole_moment"

    quadrupole_moment = QuadrupoleMoment(
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
        units=unit.debye * unit.nanometer,
    )
    assert quadrupole_moment.value.shape == (1, 3, 3)
    assert quadrupole_moment.units == unit.debye * unit.nanometer
    assert quadrupole_moment.classification == "per_system"
    assert quadrupole_moment.name == "quadrupole_moment"
    assert quadrupole_moment.property_type == "quadrupole_moment"

    polarizability = Polarizability(
        value=np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]),
        units=unit.bohr**3,
    )
    assert polarizability.value.shape == (1, 3, 3)
    assert polarizability.units == unit.bohr**3
    assert polarizability.classification == "per_system"
    assert polarizability.name == "polarizability"
    assert polarizability.property_type == "polarizability"

    base_prop = RecordProperty(
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
    # not units! we don't assume, must specify
    with pytest.raises(ValueError):
        positions = Positions(value=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([0.1]), units=unit.hartree)

    # wrong shape
    with pytest.raises(ValueError):
        energies = Energies(value=np.array([[0.1, 0.3]]), units=unit.hartree)

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

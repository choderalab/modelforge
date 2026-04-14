import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset, Energies, AtomicNumbers, MetaData

from build.lib.modelforge.tests.temp_download import prep_temp_dir
from modelforge.utils.units import GlobalUnitSystem
from modelforge.curate.properties import *

prep_temp_dir = "./data/local_dataset/"
record1 = Record(name="mol1_n2")

positions1 = Positions(
    value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[1.1, 1.1, 1.1], [2.1, 2.1, 2.1]]],
    units="nanometers",
)
energies1 = Energies(value=np.array([[0.1], [0.11]]), units=unit.kilojoule_per_mole)
atomic_numbers1 = AtomicNumbers(value=np.array([[1], [6]]))
smiles1 = MetaData(name="smiles", value="[CH+3]")
temperature1 = MetaData(name="temperature", value=298.0, units=unit.kelvin)
vector_property1 = MetaData(name="vector_property", value=np.array([[1.0, 1.0, 1.0]]))
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
vector_property2 = MetaData(name="vector_property", value=np.array([[2.0, 2.0, 2.0]]))
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
vector_property3 = MetaData(name="vector_property", value=np.array([[3.0, 3.0, 3.0]]))

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
vector_property4 = MetaData(name="vector_property", value=np.array([[4.0, 4.0, 4.0]]))

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
vector_property5 = MetaData(name="vector_property", value=np.array([[5.0, 5.0, 5.0]]))

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
vector_property6 = MetaData(name="vector_property", value=np.array([[6.0, 6.0, 6.0]]))

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

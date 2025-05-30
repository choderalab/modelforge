modelforge.curate
==================
The curate module provides an API for creating modelforge compatible hdf5 datasets from raw data.
The curate class works on a hierarchy of classes that are used to define the structure of the dataset.


Basic Usage
**************
At the top level we have `SourceDataset`. Each instance of `SourceDataset` contains records that are instances of the `Record` class.
Each instance of the `Record` class contains properties.  Properties are defined using pydanitic models that inherit from the `PropertyBaseClass`.

The use of pydantic models allows for considerable validation of the input properties.  The entire API is designed to put an emphasis on validation at the time of dataset construction, rather than at the time of dataset loading. This includes checking for compatibility of units and ensuring consistent number of atoms and configurations with a record.

The following example demonstrates how to use the curate API to create a dataset, using fictitious data.


.. code-block:: python

    from modelforge.curate import SourceDataset, Record
    from modelforge.curate.properties import AtomicNumbers, Positions, Energies, Forces, MetaData

    from openff.units import unit
    import numpy as np

    new_dataset = SourceDataset(name="test_dataset")

    record_mol1 = Record(name='mol1')

    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))

    positions = Positions(
        value=np.array([[[1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0]]]),
        units="nanometer"
    )

    energies = Energies(
        name="total_energies",
        value=np.array([[0.1]]),
        units=unit.hartree
    )

    smiles = MetaData(name='smiles', value='[CH]')

    record_mol1.add_properties([atomic_numbers, positions,energies, smiles])

    new_dataset.validate_records()

    checksum = new_dataset.to_hdf5(file_path="./", file_name="test_dataset.hdf5")



The above code creates a dataset with a single record.  The record contains the atomic numbers, positions, energies, and a metadata field for the SMILES string.  The dataset is then validated and written to an hdf5 file.  The checksum of the file is returned.  Jupyter notebooks are provided in the examples folder to  demonstrate more comprehensive usage.


Note, this module also includes classes for curating individual datasets that inherit from a `DatasetCuration` baseclass in curation_baseclass.py file.  This baseclass provides a common interface for curating the datasets, specifically providing a wrapper to make it easier to gather subsets of a dataset and impose other restrictions on the dataset at the time of writing to an hdf5 file.

Examples
**************
.. toctree::
    :maxdepth: 2
    :caption: Contents:

    _collections/notebooks/basic_usage.ipynb
    _collections/notebooks/properties
    _collections/notebooks/record_and_sourcedataset

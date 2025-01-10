from modelforge.curate.curate import *
import numpy as np

from modelforge.curate.utils import (
    _convert_unit_str_to_unit_unit,
    chem_context,
    NdArray,
)
from openff.units import unit


# To start, we will create a new instance of SourceDataset class.
new_dataset = SourceDataset("test_dataset")

# Add a new record to the dataset, giving it a unique name (as a string)
new_dataset.add_record("mol1")

# Next we can define the atomic numbers of the record using the AtomicNumbers pydantic model.
# This model can accept either a List or numpy array; if a list is provided it will automatically convert
# it to a numpy array internally.
# The array should be of (n_atoms, 1); the model will raise an error if shape[1] !=1 or len(shape) != 2
atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))


# Define positions; again, this accepts either a List or a numpy array.
# Since positions have units associated with them, we also need to specify the units
# Units can be passed as an openff.units Unit or a string that can be understood by openff.units.
# Note positions are a per_atom property, and thus must be a 3d array with shape (n_configs, n_atoms, 3).
# if shape[2] !=3 and len(shape) != 3, this will raise an error.

positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")

# Energies are a per_system property with shape (n_configs, 1)
# An error will be raised if shape[1] !=1 or len(shape) != 2
energies = Energies(value=np.array([[0.1]]), units=unit.hartree)

# Forces are a per_atom property with shape (n_configs, n_atoms, 3)
# An error will be raised if shape[2] !=4 or len(shape) != 4
forces = Forces(
    value=np.array([[[1.0, 1.0, 1.0], [0, 0.1, 0.3]]]),
    units="kilojoule_per_mole/nanometer",
)

# We can also add meta_data to describe the record.
# Note, one could specify units if they are applicable, otherwise, these are marked as dimnensionless automatically
# Here we must provide a unique name to the MetaData model.
smiles = MetaData(name="smiles", value="[CH]")

# Once defined, we can add a  property to the record.
# Note properties can be added in any order to a record
new_dataset.add_property(record_name="mol1", property=atomic_numbers)

# We can also pass a list of properties to be added rather than individual function calls
new_dataset.add_properties(
    record_name="mol1", properties=[positions, energies, forces, smiles]
)

# we can view an individual record by using get_record method
mol1_record = new_dataset.get_record(record_name="mol1")

# Printing this will provide a formatted output of the record data:
print(mol1_record)

# The record can also be exported to a dictionary

mol1_record.to_dict()

# Note, within the Record class, accessing n_atoms and n_configs, triggers validation to ensure
# the number of atoms and number of configurations defined are consistent.
# If values are inconsistent detailed warning messages will be provided in the log.
# Validation can also be triggered manually:

mol1_record.validate()

# More complete validation can be performed at the dataset level. This validation includes:
# 1- consistent number of atoms
# 2- consistent number of configurations
# 3 -validation of units (e.g., that the unit provided for Positions is a length),
# 4- ensuring that atomic_numbers, positions, and energies have been defined in the dataset

# This can be done on an individual record
print("validation status of record mol1: ", new_dataset.validate_record("mol1"))

# or over all records in a dataset
print("validation status of all records: ", new_dataset.validate_records())

# To save ths to an hdf5 file, we call the to_hdf5 function, passing the output path and filename
# This will automatically validation between we write to the file

new_dataset.to_hdf5(file_path="./", file_name="test_dataset.hdf5")


# Above, we simply used the default names for positions, energies, and forces.
# These can also accept a name parameter, similar to meta_data, to provide a unique name
# This may be necessary if, e.g., energy was calculated using different levels of theory or dispersion corrections
# are logged separately.

energies_0K = Energies(name="energies_0K", value=np.array([[0.2]]), units=unit.hartree)
new_dataset.add_property(record_name="mol1", property=energies_0K)

# In this case, the record will contain entries for both "energies" and "energies_0K"
# Note validation to ensure that energies are included, only looks at the type, ensuring that an instance of the
# Energies class has been included; it does not look for a specific string in the name field.
# Note the name given to a property must be unique and not used by any other property, including meta_data;
# atomic_numbers can also not be used as a name, as this is reserved.


# By default when instantiating a new SourceDataset instance, append_property = False.
# If append_property == False, an error will be raised if you try to add a property with the same name more than once
# to the same record. This ensures we do not accidentally overwrite data.

try:
    new_dataset.add_property(record_name="mol1", property=energies_0K)
except:
    print(
        "A value error will be raised since we have already added this and append_property is False"
    )

# If append_property is set to True, then we will merge the numpy arrays within the two records with the same name.
# This is intended for use for datasets where different configurations may exist in different files or different
# database records.
# The function will check to ensure that the arrays have compatible shapes and also perform any necessary unit conversion,
# using the first instance of the property as the reference unit.

appendable_dataset = SourceDataset(dataset_name="appendable", append_property=True)

appendable_dataset.add_record("mol2")

appendable_dataset.add_properties("mol2", [energies, atomic_numbers, positions])

# if we print this out, we will see we have n_configs = 1
print(appendable_dataset.get_record("mol2"))

# if we add the  energy and positions a second time, this adds a second configuration.
appendable_dataset.add_properties("mol2", [energies, positions])

print(appendable_dataset.get_record("mol2"))

# if we add a third time, but this time reinitializating position and energy and with different units, we can observe the
# appropriate unit conversion that occurs:

energies = Energies(value=np.array([[0.1]]), units=unit.kilojoules_per_mole)
positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="angstrom")
appendable_dataset.add_properties("mol2", [energies, positions])

print(appendable_dataset.get_record("mol2"))

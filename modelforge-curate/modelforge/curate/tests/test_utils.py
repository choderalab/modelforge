import pytest
import numpy as np


def test_convert_list_to_ndarray():
    """
    Test the _convert_list_to_ndarray function.
    """
    from modelforge.curate.utils import _convert_list_to_ndarray

    # Test with a list
    input_value = [1, 2, 3]
    result = _convert_list_to_ndarray(input_value)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array(input_value))

    # Test with a numpy array
    input_value = np.array([1, 2, 3])
    result = _convert_list_to_ndarray(input_value)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, input_value)


def test_gzip_file(prep_temp_dir):
    #  let us create a file then make sure we can gzip it
    from modelforge.curate.utils import gzip_file
    import os

    # Create a temporary file
    file_name = "test_file.txt"
    temp_file = f"{prep_temp_dir}/{file_name}"
    with open(temp_file, "w") as f:
        f.write("This is a test file.")

    # assert that the file exists
    assert os.path.exists(temp_file)

    # Gzip the file, keeping the original
    result = gzip_file(
        input_file_name=file_name, input_file_dir=prep_temp_dir, keep_original=True
    )

    # check the returned info
    assert result[0] > 0
    assert result[1] == f"{file_name}.gz"

    # Check if the gzipped file exists
    assert os.path.exists(f"{prep_temp_dir}/{result[1]}")
    # ensure the original file is still there
    assert os.path.exists(temp_file)
    # remove the gzipped file
    os.remove(f"{prep_temp_dir}/{result[1]}")

    result2 = gzip_file(
        input_file_name=file_name, input_file_dir=prep_temp_dir, keep_original=False
    )
    # check the returned info
    assert result2[0] > 0
    assert result2[0] == result[0]
    assert result2[1] == f"{file_name}.gz"
    # Check if the gzipped file exists
    assert os.path.exists(f"{prep_temp_dir}/{result2[1]}")
    # ensure the original file is not there
    assert not os.path.exists(temp_file)

    # ensure that by default we will overwrite the gzipped file
    # to check this, let us just add some more data that will change the length of the gzip file

    with open(temp_file, "w") as f:
        f.write("This is some more text for the file.")
        f.write("This will alter the length so we can ensure it is different.")
        for i in range(100):
            f.write(f"{i*29+i*i} ")

    result3 = gzip_file(
        input_file_name=file_name,
        input_file_dir=prep_temp_dir,
        keep_original=False,
    )

    assert result3[0] > 0
    assert result3[0] != result2[0]
    assert result3[1] == f"{file_name}.gz"
    # Check if the gzipped file exists
    assert os.path.exists(f"{prep_temp_dir}/{result3[1]}")


def test_VersionMetadata(prep_temp_dir):
    # this will test the yaml file metadata helper class VersionMetadata
    # first we need to create a SourceDataset instance so we can write an HDF5 file

    from modelforge.curate import SourceDataset, Record
    from modelforge.curate.properties import (
        Positions,
        Energies,
        AtomicNumbers,
        MetaData,
    )
    from openff.units import unit

    record = Record(name="mol1")

    positions = Positions(value=[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], units="nanometer")
    energies = Energies(value=np.array([[0.1]]), units=unit.hartree)
    atomic_numbers = AtomicNumbers(value=np.array([[1], [6]]))
    meta_data = MetaData(name="smiles", value="[CH]")

    record.add_property(property=atomic_numbers)
    record.add_properties([positions, energies, meta_data])

    new_dataset = SourceDataset(
        name="test_dataset3",
        local_db_dir=str(prep_temp_dir),
        local_db_name="test_dataset3.sqlite",
    )
    new_dataset.add_record(record)

    checksum = new_dataset.to_hdf5(
        file_path=str(prep_temp_dir), file_name="test_dataset.hdf5"
    )

    from modelforge.curate.utils import VersionMetadata

    version_metadata = VersionMetadata(
        version_name="test1",
        about="test dataset version test1",
        hdf5_file_name="test_dataset.hdf5",
        hdf5_file_dir=str(prep_temp_dir),
        available_properties=[
            "atomic_numbers",
            "energies",
            "positions",
        ],
    )
    version_metadata.compress_hdf5()
    # check the gzipped file exists
    import os

    os.path.exists(f"{prep_temp_dir}/{version_metadata.gzipped_file_name}")
    # generate the dictionary if this is a remote dataset
    data_dict = version_metadata.remote_dataset_to_dict()
    # check the dictionary is correct
    assert data_dict["test1"]["hdf5_schema"] == 2
    assert data_dict["test1"]["available_properties"] == [
        "atomic_numbers",
        "energies",
        "positions",
    ]
    assert data_dict["test1"]["about"] == "test dataset version test1"
    assert data_dict["test1"]["remote_dataset"]["doi"] == " "
    assert data_dict["test1"]["remote_dataset"]["url"] == " "
    assert (
        data_dict["test1"]["remote_dataset"]["gz_data_file"]["length"]
        == version_metadata.gzipped_length
    )
    assert (
        data_dict["test1"]["remote_dataset"]["gz_data_file"]["md5"]
        == version_metadata.gzipped_checksum
    )
    assert (
        data_dict["test1"]["remote_dataset"]["gz_data_file"]["file_name"]
        == version_metadata.gzipped_file_name
    )
    assert (
        data_dict["test1"]["remote_dataset"]["hdf5_data_file"]["md5"]
        == version_metadata.hdf5_checksum
    )
    assert (
        data_dict["test1"]["remote_dataset"]["hdf5_data_file"]["file_name"]
        == version_metadata.hdf5_file_name
    )
    # check the local dataset dictionary
    data_dict = version_metadata.local_dataset_to_dict()
    # check the dictionary is correct
    assert data_dict["test1"]["hdf5_schema"] == 2
    assert data_dict["test1"]["available_properties"] == [
        "atomic_numbers",
        "energies",
        "positions",
    ]
    assert data_dict["test1"]["about"] == "test dataset version test1"
    assert (
        data_dict["test1"]["local_dataset"]["hdf5_data_file"]["md5"]
        == version_metadata.hdf5_checksum
    )
    assert (
        data_dict["test1"]["local_dataset"]["hdf5_data_file"]["file_name"]
        == f"{version_metadata.hdf5_file_dir}/{version_metadata.hdf5_file_name}"
    )

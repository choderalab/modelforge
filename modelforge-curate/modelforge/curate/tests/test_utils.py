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

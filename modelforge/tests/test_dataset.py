import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modelforge.dataset import QM9Dataset


@pytest.fixture(
    autouse=True,
)
def cleanup_files():
    """Fixture to clean up temporary files before and after test execution."""

    def _cleanup():
        files = [
            "tmp.hdf5_cache.hdf5",
            "tmp.hdf5_cache.hdf5.gz",
            "tmp.hdf5_processed.npz",
            "tmp.hdf5_subset_cache.hdf5",
            "tmp.hdf5_subset_cache.hdf5.gz",
            "tmp.hdf5_subset_processed.npz",
        ]
        for f in files:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except FileNotFoundError:
                print(f"{f} not found")

    _cleanup()
    yield  # This is where the test will execute
    _cleanup()


def test_dataset_imported():
    """Sample test, will always pass so long as import statement worked."""
    import modelforge.dataset

    assert "modelforge.dataset" in sys.modules


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_file_existence_after_initialization(dataset):
    """Test if files are created after dataset initialization."""
    d = dataset("tmp.hdf5", test_data=True)
    assert os.path.exists(d.raw_dataset_file)
    assert os.path.exists(d.processed_dataset_file)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_different_scenarios_of_file_availability(dataset):
    """Test the behavior when raw and processed dataset files are removed."""
    d = dataset("tmp.hdf5", test_data=True)

    os.remove(d.raw_dataset_file)
    d = dataset("tmp.hdf5", test_data=True)

    os.remove(d.processed_dataset_file)
    d = dataset("tmp.hdf5", test_data=True)
    assert os.path.exists(d.raw_dataset_file)
    assert os.path.exists(d.processed_dataset_file)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_data_item_format(dataset):
    """Test the format of individual data items in the dataset."""
    d = dataset("tmp.hdf5", test_data=True)
    data_item = d[0]
    assert isinstance(data_item, tuple)
    assert len(data_item) == 3
    assert isinstance(data_item[0], torch.Tensor)
    assert isinstance(data_item[1], torch.Tensor)
    assert isinstance(data_item[2], torch.Tensor)


def test_padding():
    """Test the padding function to ensure correct behavior on dummy data."""
    dummy_data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(
        [[0, 0, 0], [0, 0, 0]]
    )
    max_len = max(len(arr) for arr in dummy_data)
    padded_data = QM9Dataset.pad_molecules(dummy_data)

    for data in padded_data:
        assert data.shape[0] == max_len

    assert np.array_equal(padded_data[-1][-1], np.array([-1, -1, -1]))


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_download_dataset(dataset):
    """Test the downloading and batching of the dataset."""
    d = dataset(dataset_name="tmp.hdf5", test_data=True)
    print(d.dataset_name)
    train_dataloader = DataLoader(d, batch_size=64, shuffle=False)
    for b in train_dataloader:
        print("batching")


def test_dataset_length():
    """Test the length method of the dataset."""
    d = QM9Dataset("tmp.hdf5", test_data=True)
    assert len(d) == len(d.dataset["atomic_numbers"])


def test_getitem_type_and_shape():
    """Test the __getitem__ method for type and shape consistency."""
    d = QM9Dataset("tmp.hdf5", test_data=True)
    data_item = d[0]
    assert isinstance(data_item, tuple)
    assert data_item[0].shape[1] == 3  # Assuming 3D coordinates
    assert data_item[1].ndim == 1  # Atomic numbers should be 1D

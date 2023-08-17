import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modelforge.dataset import QM9Dataset
from modelforge.dataset.dataset import GenericDataset


@pytest.fixture(
    autouse=True,
)
def cleanup_files():
    """Fixture to clean up temporary files before and after test execution."""

    def _cleanup():
        files = [
            "tmp_cache.hdf5",
            "tmp_cache.hdf5.gz",
            "tmp_processed.npz",
            "tmp_subset_cache.hdf5",
            "tmp_subset_cache.hdf5.gz",
            "tmp_subset_processed.npz",
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
    d = dataset("tmp", test_data=True)
    assert not os.path.exists(d.raw_dataset_file)
    assert not os.path.exists(d.processed_dataset_file)

    d.load_or_process_data()  # call explicitly load_or_process_data to load data
    assert os.path.exists(d.raw_dataset_file)
    assert os.path.exists(d.processed_dataset_file)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_different_scenarios_of_file_availability(dataset):
    """Test the behavior when raw and processed dataset files are removed."""
    d = dataset("tmp", test_data=True)
    d.load_or_process_data()  # call explicitly load_or_process_data to load data
    os.remove(d.raw_dataset_file)
    d = dataset("tmp", test_data=True)

    os.remove(d.processed_dataset_file)
    d = dataset("tmp", test_data=True)
    d.load_or_process_data()  # call explicitly load_or_process_data to load data
    assert os.path.exists(d.raw_dataset_file)
    assert os.path.exists(d.processed_dataset_file)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_data_item_format(dataset):
    """Test the format of individual data items in the dataset."""
    d = dataset("tmp", test_data=True)
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
    d = dataset(dataset_name="tmp", test_data=True)
    assert d.dataset_name == "tmp_subset"
    train_dataloader = DataLoader(d, batch_size=64, shuffle=True)
    assert len([b for b in train_dataloader]) == 16  # (1_000 / 64 = 15.625)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_dataset_length(dataset):
    """Test the length method of the dataset."""
    d = dataset("tmp", test_data=True)
    assert len(d) == len(d.dataset["atomic_numbers"])


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_dataset_splitting(dataset):
    """Test random_split on the the dataset."""
    d = dataset("tmp", test_data=True)
    training_d, validation_d, test_d = torch.utils.data.random_split(d, [800, 100, 100])
    assert len(training_d) == 800
    train_dataloader = DataLoader(training_d, batch_size=10, shuffle=True)
    assert len([b for b in train_dataloader]) == 80  # (800 / 10 = 80)


@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_getitem_type_and_shape(dataset):
    """Test the __getitem__ method for type and shape consistency."""
    d = dataset("tmp", test_data=True)
    data_item = d[0]
    assert isinstance(data_item, tuple)
    assert data_item[0].shape[1] == 3  # Assuming 3D coordinates
    assert data_item[1].ndim == 1  # Atomic numbers should be 1D


def test_GenericDataset():
    # generate the npz file
    QM9Dataset("tmp", test_data=True).load_or_process_data()

    g = GenericDataset("generic")
    dataset = np.load("tmp_subset_processed.npz")
    g.dataset = dataset
    train_dataloader = DataLoader(g, batch_size=64, shuffle=True)
    assert len([b for b in train_dataloader]) == 16  # (1_000 / 64 = 15.625)

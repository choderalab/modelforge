import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from modelforge.dataset.dataset import DatasetFactory
from modelforge.dataset.qm9 import QM9Dataset

DATASETS = [QM9Dataset]


@pytest.fixture(
    autouse=True,
)
def cleanup_files():
    """Fixture to clean up temporary files before and after test execution."""

    def _cleanup():
        for dataset_prop in DATASETS:
            dataset_name = dataset_prop().dataset_name

            files = [
                f"{dataset_name}_cache.hdf5",
                f"{dataset_name}_cache.hdf5.gz",
                f"{dataset_name}_processed.npz",
                f"{dataset_name}_subset_cache.hdf5",
                f"{dataset_name}_subset_cache.hdf5.gz",
                f"{dataset_name}_subset_processed.npz",
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


@pytest.mark.parametrize("dataset_properties", DATASETS)
def test_file_existence_after_initialization(dataset_properties):
    """Test if files are created after dataset initialization."""
    factory = DatasetFactory()
    prop = dataset_properties(for_testing=True)

    assert not os.path.exists(prop.raw_dataset_file)
    assert not os.path.exists(prop.processed_dataset_file)

    dataset = factory.create_dataset(prop)
    assert os.path.exists(prop.raw_dataset_file)
    assert os.path.exists(prop.processed_dataset_file)


@pytest.mark.parametrize("dataset_properties", DATASETS)
def test_different_scenarios_of_file_availability(dataset_properties):
    """Test the behavior when raw and processed dataset files are removed."""
    factory = DatasetFactory()
    prop = dataset_properties(for_testing=True)

    factory.create_dataset(prop)

    os.remove(prop.raw_dataset_file)
    factory.create_dataset(prop)

    os.remove(prop.processed_dataset_file)
    factory.create_dataset(prop)
    assert os.path.exists(prop.raw_dataset_file)
    assert os.path.exists(prop.processed_dataset_file)


@pytest.mark.parametrize("dataset_properties", DATASETS)
def test_data_item_format(dataset_properties):
    """Test the format of individual data items in the dataset."""
    factory = DatasetFactory()
    prop = dataset_properties(for_testing=True)
    dataset = factory.create_dataset(prop)

    raw_data_item = dataset.dataset[0]
    assert isinstance(raw_data_item, tuple)
    assert len(raw_data_item) == 3
    assert isinstance(raw_data_item[0], torch.Tensor)
    assert isinstance(raw_data_item[1], torch.Tensor)
    assert isinstance(raw_data_item[2], torch.Tensor)


def test_padding():
    """Test the padding function to ensure correct behavior on dummy data."""
    from modelforge.dataset.utils import PadTensors

    dummy_data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(
        [[0, 0, 0], [0, 0, 0]]
    )
    max_len = max(len(arr) for arr in dummy_data)
    padded_data = PadTensors.pad_molecules(dummy_data)

    for data in padded_data:
        assert data.shape[0] == max_len

    assert np.array_equal(padded_data[-1][-1], np.array([-1, -1, -1]))


@pytest.mark.parametrize("dataset_properties", DATASETS)
def test_dataset_generation(dataset_properties):
    """Test the splitting of the dataset."""
    factory = DatasetFactory()

    prop = dataset_properties(for_testing=True)
    dataset = factory.create_dataset(prop)

    assert len(dataset.train_dataset) == 800
    assert len(dataset.test_dataset) == 100
    assert len(dataset.val_dataset) == 100


@pytest.mark.parametrize("dataset_properties", DATASETS)
def test_dataset_splitting(dataset_properties):
    """Test random_split on the the dataset."""
    from modelforge.dataset.utils import RandomSplittingStrategy

    factory = DatasetFactory()

    prop = dataset_properties(for_testing=True)
    dataset = factory.create_dataset(prop)

    energy = dataset.train_dataset[0][2].item()
    assert np.isclose(energy, -236.91345796962494)
    print(energy)

    try:
        splitting = RandomSplittingStrategy(split=[0.2, 0.1, 0.1])
    except AssertionError:
        print("AssertionError raised")
        pass

    splitting = RandomSplittingStrategy(split=[0.6, 0.3, 0.1]).split
    dataset = factory.create_dataset(prop, splitting=splitting)

    assert len(dataset.train_dataset) == 600


# @pytest.mark.parametrize("dataset", [QM9Dataset])
# def test_getitem_type_and_shape(dataset):
#     """Test the __getitem__ method for type and shape consistency."""
#     d = dataset("tmp", test_data=True)
#     data_item = d[0]
#     assert isinstance(data_item, tuple)
#     assert data_item[0].shape[1] == 3  # Assuming 3D coordinates
#     assert data_item[1].ndim == 1  # Atomic numbers should be 1D


# def test_GenericDataset():
#     # generate the npz file
#     QM9Dataset("tmp", test_data=True).load_or_process_data()

#     g = GenericDataset("generic")
#     dataset = np.load("tmp_subset_processed.npz")
#     g.dataset = dataset
#     train_dataloader = DataLoader(g, batch_size=64, shuffle=True)
#     assert len([b for b in train_dataloader]) == 16  # (1_000 / 64 = 15.625)

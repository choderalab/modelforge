import os
import sys

import numpy as np
import pytest
import torch

from modelforge.dataset.dataset import DatasetFactory, TorchDataset

from .helper_functions import initialize_dataset, DATASETS


@pytest.fixture(
    autouse=True,
)
def cleanup_files():
    """Fixture to clean up temporary files before and after test execution."""

    def _cleanup():
        for dataset in DATASETS:
            dataset_name = dataset().dataset_name

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


def generate_torch_dataset(dataset) -> TorchDataset:
    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)
    return factory.create_dataset(data)


def test_dataset_imported():
    """Sample test, will always pass so long as import statement worked."""

    assert "modelforge.dataset" in sys.modules


@pytest.mark.parametrize("dataset", DATASETS)
def test_different_properties_of_interest(dataset):
    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)
    assert data.properties_of_interest == [
        "geometry",
        "atomic_numbers",
        "internal_energy_at_0K",
    ]

    dataset = factory.create_dataset(data)
    raw_data_item = dataset[0]
    assert isinstance(raw_data_item, dict)
    assert len(raw_data_item) == 5

    data.properties_of_interest = ["internal_energy_at_0K", "geometry"]
    assert data.properties_of_interest == [
        "internal_energy_at_0K",
        "geometry",
    ]

    dataset = factory.create_dataset(data)
    raw_data_item = dataset[0]
    print(raw_data_item)
    assert isinstance(raw_data_item, dict)
    assert len(raw_data_item) != 3


@pytest.mark.parametrize("dataset", DATASETS)
def test_file_existence_after_initialization(dataset):
    """Test if files are created after dataset initialization."""
    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)

    assert not os.path.exists(data.raw_data_file)
    assert not os.path.exists(data.processed_data_file)

    dataset = factory.create_dataset(data)
    assert os.path.exists(data.raw_data_file)
    assert os.path.exists(data.processed_data_file)


@pytest.mark.parametrize("dataset", DATASETS)
def test_different_scenarios_of_file_availability(dataset):
    """Test the behavior when raw and processed dataset files are removed."""
    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)

    factory.create_dataset(data)

    os.remove(data.raw_data_file)
    factory.create_dataset(data)

    os.remove(data.processed_data_file)
    factory.create_dataset(data)
    assert os.path.exists(data.raw_data_file)
    assert os.path.exists(data.processed_data_file)


@pytest.mark.parametrize("dataset", DATASETS)
def test_data_item_format(dataset):
    """Test the format of individual data items in the dataset."""
    from typing import Dict

    dataset = initialize_dataset(
        dataset, mode="fit", split_file="modelforge/tests/qm9tut/split.npz"
    )

    raw_data_item = dataset.dataset[0]
    assert isinstance(raw_data_item, Dict)
    assert isinstance(raw_data_item["atomic_numbers"], torch.Tensor)
    assert isinstance(raw_data_item["positions"], torch.Tensor)
    assert isinstance(raw_data_item["E_label"], torch.Tensor)
    print(raw_data_item)

    assert (
        raw_data_item["atomic_numbers"].shape[0] == raw_data_item["positions"].shape[0]
    )


def test_padding():
    """Test the padding function to ensure correct behavior on dummy data."""
    from modelforge.dataset.utils import pad_molecules

    dummy_data = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array(
        [[0, 0, 0], [0, 0, 0]]
    )
    max_len = max(len(arr) for arr in dummy_data)
    padded_data = pad_molecules(dummy_data)

    for data in padded_data:
        assert data.shape[0] == max_len

    assert np.array_equal(padded_data[-1][-1], np.array([-1, -1, -1]))


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_generation(dataset):
    """Test the splitting of the dataset."""

    dataset = initialize_dataset(dataset, mode="fit")
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()

    try:
        dataset.test_dataloader()
    except AttributeError:
        # this isn't set when dataset is in 'fit' mode
        pass

    # the dataloader automatically splits and batches the dataset
    # for the training set it batches the 80 datapoints in
    # a batch of 64 and a batch of 16 samples
    assert len(train_dataloader) == 2  # nr of batches
    v = [v_ for v_ in train_dataloader]
    assert len(v[0]["atomic_subsystem_counts"]) == 64
    assert len(v[1]["atomic_subsystem_counts"]) == 16


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_splitting(dataset):
    """Test random_split on the the dataset."""
    from modelforge.dataset.utils import RandomSplittingStrategy

    dataset = generate_torch_dataset(dataset)
    train_dataset, val_dataset, test_dataset = RandomSplittingStrategy().split(dataset)

    energy = train_dataset[0]["E_label"].item()
    assert np.isclose(energy, -412509.9375)
    print(energy)

    try:
        RandomSplittingStrategy(split=[0.2, 0.1, 0.1])
    except AssertionError:
        print("AssertionError raised")
        pass

    train_dataset, val_dataset, test_dataset = RandomSplittingStrategy(
        split=[0.6, 0.3, 0.1]
    ).split(dataset)

    assert len(train_dataset) == 60


@pytest.mark.parametrize("dataset", DATASETS)
def test_file_cache_methods(dataset):
    """
    Test the FileCache methods to ensure data is cached and loaded correctly.
    """

    # generate files to test _from_hdf5()

    _ = initialize_dataset(dataset, mode="str")

    data = dataset(for_unit_testing=True)

    data._from_hdf5()

    data._to_file_cache()
    data._from_file_cache()
    assert len(data.numpy_data["atomic_subsystem_counts"]) == 100


@pytest.mark.parametrize("dataset", DATASETS)
def test_dataset_downloader(dataset):
    """
    Test the DatasetDownloader functionality.
    """
    data = dataset(for_unit_testing=True)
    data._download()
    assert os.path.exists(data.raw_data_file)


@pytest.mark.parametrize("dataset", DATASETS)
def test_numpy_dataset_assignment(dataset):
    """
    Test if the numpy_dataset attribute is correctly assigned after processing or loading.
    """
    from modelforge.dataset.transformation import default_transformation

    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)
    factory._load_or_process_data(data, None, default_transformation)

    assert hasattr(data, "numpy_data")
    assert isinstance(data.numpy_data, np.lib.npyio.NpzFile)

import os
import sys

import numpy as np
import pytest
import torch

from modelforge.dataset.dataset import DatasetFactory, TorchDataset
from modelforge.dataset import QM9Dataset

DATASETS = [QM9Dataset]
from ..utils import PropertyNames


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


def test_dataset_imported():
    """Sample test, will always pass so long as import statement worked."""

    assert "modelforge.dataset" in sys.modules


def test_dataset_basic_operations():
    atomic_subsystem_counts = np.array([3, 4])
    n_confs = np.array([2, 1])
    total_atoms_series = (atomic_subsystem_counts * n_confs).sum()
    total_atoms_single = atomic_subsystem_counts.sum()
    total_confs = n_confs.sum()
    total_records = len(atomic_subsystem_counts)
    geom_shape = (total_atoms_series, 3)
    atomic_numbers_shape = (total_atoms_single, 1)
    internal_energy_shape = (total_confs, 1)
    input_data = {
        "geometry": np.arange(geom_shape[0] * geom_shape[1]).reshape(geom_shape),
        "atomic_numbers": np.arange(atomic_numbers_shape[0]).reshape(
            atomic_numbers_shape
        ),
        "internal_energy_at_0K": np.arange(internal_energy_shape[0]).reshape(
            internal_energy_shape
        ),
        "atomic_subsystem_counts": atomic_subsystem_counts,
        "n_confs": n_confs,
        "charges": torch.randint(-1, 2, torch.Size([total_confs])).numpy(),
    }

    property_names = PropertyNames(
        "atomic_numbers", "geometry", "internal_energy_at_0K", "charges"
    )
    dataset = TorchDataset(input_data, property_names)
    assert len(dataset) == total_confs
    assert dataset.record_len() == total_records

    atomic_numbers_true = []
    geom_true = []
    energy_true = []
    series_mol_idxs = []
    atom_start_idx_series = 0
    atom_start_idx_single = 0
    conf_idx = 0
    for rec in range(total_records):
        series_mol_idxs_for_rec = []
        atom_end_idx_single = atom_start_idx_single + atomic_subsystem_counts[rec]
        for conf in range(n_confs[rec]):
            atom_end_idx_series = atom_start_idx_series + atomic_subsystem_counts[rec]
            energy_true.append(input_data["internal_energy_at_0K"][conf_idx])
            geom_true.append(
                input_data["geometry"][atom_start_idx_series:atom_end_idx_series]
            )
            atomic_numbers_true.append(
                input_data["atomic_numbers"][
                    atom_start_idx_single:atom_end_idx_single
                ].flatten()
            )
            series_mol_idxs_for_rec.append(conf_idx)
            atom_start_idx_series = atom_end_idx_series
            conf_idx += 1
        atom_start_idx_single = atom_end_idx_single
        series_mol_idxs.append(series_mol_idxs_for_rec)

    for conf_idx in range(len(dataset)):
        conf_data = dataset[conf_idx]
        assert np.array_equal(conf_data["positions"], geom_true[conf_idx])
        assert np.array_equal(
            conf_data["atomic_numbers"], atomic_numbers_true[conf_idx]
        )
        assert np.array_equal(conf_data["E"], energy_true[conf_idx])

    for rec_idx in range(dataset.record_len()):
        assert np.array_equal(
            dataset.get_series_mol_idxs(rec_idx), series_mol_idxs[rec_idx]
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_different_properties_of_interest(dataset):
    factory = DatasetFactory()
    data = dataset(for_unit_testing=True)
    assert data.properties_of_interest == [
        "geometry",
        "atomic_numbers",
        "internal_energy_at_0K",
        "charges",
    ]

    dataset = factory.create_dataset(data)
    raw_data_item = dataset[0]
    assert isinstance(raw_data_item, dict)
    assert len(raw_data_item) == 7

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


def test_data_item_format(initialized_dataset):
    """Test the format of individual data items in the dataset."""
    from typing import Dict

    raw_data_item = initialized_dataset.torch_dataset[0]
    assert isinstance(raw_data_item, Dict)
    assert isinstance(raw_data_item["atomic_numbers"], torch.Tensor)
    assert isinstance(raw_data_item["positions"], torch.Tensor)
    assert isinstance(raw_data_item["E"], torch.Tensor)
    print(raw_data_item)

    assert (
        raw_data_item["atomic_numbers"].shape[0] == raw_data_item["positions"].shape[0]
    )


def test_dataset_generation(initialized_dataset):
    """Test the splitting of the dataset."""

    dataset = initialized_dataset
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
    batch_data = [v_ for v_ in train_dataloader]
    assert len(batch_data[0].metadata.atomic_subsystem_counts) == 64
    assert len(batch_data[1].metadata.atomic_subsystem_counts) == 16


from modelforge.dataset.utils import (
    RandomRecordSplittingStrategy,
    FirstComeFirstServeSplittingStrategy,
)


@pytest.mark.parametrize(
    "splitting_strategy",
    [RandomRecordSplittingStrategy, FirstComeFirstServeSplittingStrategy],
)
def test_dataset_splitting(splitting_strategy, datasets_to_test):
    """Test random_split on the the dataset."""
    from modelforge.dataset import DatasetFactory

    dataset = DatasetFactory.create_dataset(datasets_to_test)
    train_dataset, val_dataset, test_dataset = splitting_strategy().split(dataset)

    energy = train_dataset[0]["E"].item()

    assert np.isclose(energy, -412509.9375) or np.isclose(energy, -106277.4161215308)

    try:
        splitting_strategy(split=[0.2, 0.1, 0.1])
    except AssertionError as excinfo:
        print(f"AssertionError raised: {excinfo}")

    train_dataset, val_dataset, test_dataset = splitting_strategy(
        split=[0.6, 0.3, 0.1]
    ).split(dataset)

    assert len(train_dataset) == 60
    assert len(val_dataset) == 30
    assert len(test_dataset) == 10


@pytest.mark.parametrize("dataset", DATASETS)
def test_file_cache_methods(dataset):
    """
    Test the FileCache methods to ensure data is cached and loaded correctly.
    """

    # generate files to test _from_hdf5()

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


def test_numpy_dataset_assignment(initialized_dataset):
    """
    Test if the numpy_dataset attribute is correctly assigned after processing or loading.
    """

    factory = DatasetFactory()
    data = initialized_dataset(for_unit_testing=True)
    factory._load_or_process_data(data)

    assert hasattr(data, "numpy_data")
    assert isinstance(data.numpy_data, np.lib.npyio.NpzFile)


def test_self_energy():

    from modelforge.dataset.dataset import TorchDataModule

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=32, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )
    dataset.prepare_data(
        remove_self_energies=False, normalize=False, regression_ase=False
    )
    methane_energy_reference = float(dataset.torch_dataset[0]["E"])
    assert np.isclose(methane_energy_reference, -106277.4161)

    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=32, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    # Scenario 1: dataset contains self energies
    # self energy is obtained in prepare_data if `remove_self_energies` is True
    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # it is saved in the dataset statistics
    assert dataset.dataset_statistics
    self_energies = dataset.dataset_statistics.atomic_self_energies
    # 5 elements present in the QM9 dataset
    assert len(self_energies) == 5
    # H: -1313.4668615546
    assert np.isclose(self_energies[1], -1313.4668615546)
    # C: -99366.70745535441
    assert np.isclose(self_energies[6], -99366.70745535441)
    # N: -143309.9379722722
    assert np.isclose(self_energies[7], -143309.9379722722)
    # O: -197082.0671774158
    assert np.isclose(self_energies[8], -197082.0671774158)

    # Scenario 2: dataset may or may not contain self energies
    # but user wants to use least square regression to calculate the energies
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=32, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )
    dataset.prepare_data(
        remove_self_energies=True, normalize=False, regression_ase=True
    )
    # it is saved in the dataset statistics
    assert dataset.dataset_statistics
    self_energies = dataset.dataset_statistics.atomic_self_energies
    # 5 elements present in the total QM9 dataset
    # but only 4 in the reduced QM9 dataset
    assert len(self_energies) == 4
    # H: -1313.4668615546
    assert np.isclose(self_energies[1], -1584.5087457646314)
    # C: -99366.70745535441
    assert np.isclose(self_energies[6], -99960.8894178209)
    # N: -143309.9379722722
    assert np.isclose(self_energies[7], -143754.02638655982)
    # O: -197082.0671774158
    assert np.isclose(self_energies[8], -197495.00132926635)

    dataset.prepare_data(
        remove_self_energies=True, normalize=False, regression_ase=True
    )
    # it is saved in the dataset statistics
    assert dataset.dataset_statistics
    self_energies = dataset.dataset_statistics.atomic_self_energies

    # Test that self energies are correctly removed
    for regression in [True, False]:
        data = QM9Dataset(for_unit_testing=True)
        dataset = TorchDataModule(
            data,
            batch_size=32,
            splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        )
        dataset.prepare_data(
            remove_self_energies=True, normalize=False, regression_ase=regression
        )
        # Extract the first molecule (methane)
        # double check that it is methane
        k = dataset.torch_dataset[0]
        methane_atomic_indices = dataset.torch_dataset[0]["atomic_numbers"]
        # extract energy
        methane_energy_offset = dataset.torch_dataset[0]["E"]
        if regression is False:
            # checking that the offset energy is actually correct for methane
            assert torch.isclose(
                methane_energy_offset, torch.tensor([-1656.8412], dtype=torch.float64)
            )
        # extract the ase offset
        self_energies = dataset.dataset_statistics.atomic_self_energies
        methane_ase = sum(
            [
                self_energies[int(index)]
                for index in list(methane_atomic_indices.numpy())
            ]
        )
        # compare this to the energy without postprocessing
        assert np.isclose(methane_energy_reference, methane_energy_offset + methane_ase)

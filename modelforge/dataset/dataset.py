import os
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple, Callable

import h5py
import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from .utils import RandomSplittingStrategy, SplittingStrategy


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: np.ndarray):
        self.dataset = dataset

    def __len__(self) -> int:
        """
        Return the number of datapoints in the dataset.

        Returns:
        --------
        int
            Total number of datapoints available in the dataset.
        """
        return len(self.dataset["atomic_numbers"])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch a tuple of geometry, atomic numbers, and energy for a given molecule index.

        Parameters:
        -----------
        idx : int
            Index of the molecule to fetch data for.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing tensors for geometry, atomic numbers, and energy of the molecule.
        """
        return (
            torch.tensor(self.dataset["coordinates"][idx]),
            torch.tensor(self.dataset["atomic_numbers"][idx]),
            torch.tensor(self.dataset["return_energy"][idx]),
        )


class Dataset:
    """
    A class representing a dataset ready for training, testing, and validation.
    Provides data loaders for each split of the data.
    """

    def __init__(
        self,
        dataset: np.ndarray,
        splitting: Callable,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initializes the Dataset class.

        Args:
            dataset (np.ndarray): The underlying dataset.
            splitting (Callable): The splitting strategy for train/test/validation datasets.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of worker threads for data loading.
            pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory before returning them.
        """

        self.dataset = TorchDataset(dataset)
        self.split = splitting
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._pin_memory = pin_memory
        self._train_dataloader = None
        self._test_dataloader = None
        self._val_dataloader = None
        self._train_dataset, self._val_dataset, self._test_dataset = self.split(
            self.dataset
        )

    def _create_dataloader(self, dataset_subset, batch_size):
        """
        Internal method to create a DataLoader instance for a given dataset subset.

        Args:
            dataset_subset (torch.utils.data.Dataset): Subset of the data to create DataLoader for.
            batch_size (int): Batch size for the DataLoader.

        Returns:
            DataLoader: DataLoader instance for the given dataset subset.
        """

        return DataLoader(
            dataset_subset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self._pin_memory,
        )

    @property
    def train_dataset(self) -> TorchDataset:
        return self._train_dataset

    @property
    def val_dataset(self) -> TorchDataset:
        return self._val_dataset

    @property
    def test_dataset(self) -> TorchDataset:
        return self._test_dataset

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None:
            self._train_dataloader = self._create_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
            )
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        """
        Provides the DataLoader for the validation data.

        Returns:
            DataLoader: DataLoader instance for the validation data.
        """
        if self._val_dataloader is None:
            self._val_dataloader = self._create_dataloader(
                self.val_dataset,
                batch_size=self.batch_size,
            )
        return self._val_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        """
        Provides the DataLoader for the test data.

        Returns:
            DataLoader: DataLoader instance for the test data.
        """
        if self._test_dataloader is None:
            self._test_dataloader = self._create_dataloader(
                self.test_dataset,
                batch_size=self.batch_size,
            )
        return self._test_dataloader


class FileCache:
    """
    Utility class for caching datasets into files and loading them from cache.
    """

    @staticmethod
    def from_file_cache(processed_dataset_file) -> np.ndarray:
        """
        Loads the dataset from a cached file.

        Args:
            processed_dataset_file (str): Path to the cached dataset file.

        Returns:
            np.ndarray: The loaded dataset.
        """
        logger.info(f"Loading cached dataset_file {processed_dataset_file}")
        return np.load(processed_dataset_file)

    @staticmethod
    def to_file_cache(hdf_dataset, data):
        """
        Caches the processed dataset into a file.

        Args:
            hdf_dataset :HDFDataset.
            processed_dataset_file (str): Path to the file to cache the dataset.
        """
        logger.info("Caching hdf5 file ...")
        # This assumes that there exists a 'to_npz' method which saves the data
        hdf_dataset.to_npz(data)


class HDF5Dataset:
    """
    class for datasets stored in HDF5 format. Provides methods
    for processing and interacting with the data.
    """

    def __init__(self, raw_dataset_file, processed_dataset_file):
        self.raw_dataset_file = raw_dataset_file
        self.processed_dataset_file = processed_dataset_file

    def from_hdf5(self) -> Dict[str, List]:
        """
        Processes and extracts data from an hdf5 file.

        Returns:
            Dict[str, List]: Processed data from the hdf5 file.
        """
        logger.debug("Reading in and processing hdf5 file ...")
        data = defaultdict(list)
        logger.debug(f"Processing and extracting data from {self.raw_dataset_file}")
        with h5py.File(self.raw_dataset_file, "r") as hf:
            logger.debug(f"n_entries: {len(hf.keys())}")
            for mol in tqdm.tqdm(list(hf.keys())):
                for value in self.keywords_for_hdf5_dataset:
                    data[value].append(hf[mol][value][()])
        return data

    def _load(self) -> np.ndarray:
        """
        Loads and returns the dataset from the npz file.

        Returns:
            np.ndarray: Loaded dataset.
        """
        return np.load(self.processed_dataset_file)


class DatasetFactory:
    """
    Factory class for creating Dataset instances. Provides utilities for
    processing and caching datasets.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes the Dataset class.

        Args:
            load_in_memory (bool): Whether to load the entire dataset into memory.
        """
        pass

    @staticmethod
    def _load_or_process_data(dataset: HDF5Dataset) -> None:
        """
        Loads the dataset from cache if available, otherwise processes and caches the data.
        """

        if not os.path.exists(dataset.processed_dataset_file):
            if not os.path.exists(dataset.raw_dataset_file):
                dataset.download_hdf_file()
            data = dataset.from_hdf5()
            FileCache.to_file_cache(dataset, data)

        dataset.numpy_dataset = FileCache.from_file_cache(
            dataset.processed_dataset_file
        )

    @staticmethod
    def create_dataset(
        dataset: HDF5Dataset,
        splitting: Callable = RandomSplittingStrategy().split,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> Dataset:
        """
        Creates a Dataset instance given an HDF5Dataset.

        Args:
            dataset (HDF5Dataset): The HDF5 dataset to use.
            splitting (Callable): The splitting strategy for train/test/validation datasets.
            batch_size (int): Batch size for data loading.
            num_workers (int): Number of worker threads for data loading.
            pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory before returning them.

        Returns:
            Dataset: Dataset instance with data loaders ready for training/testing/validation.
        """

        logger.info(f"Creating {dataset.dataset_name} dataset")
        DatasetFactory._load_or_process_data(dataset)
        return Dataset(
            dataset.numpy_dataset,
            splitting=splitting,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

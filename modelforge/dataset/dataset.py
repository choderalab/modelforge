import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable

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
    def __init__(
        self,
        dataset: np.ndarray,
        splitting: Callable,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
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
            self._train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self._pin_memory,
            )
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_val_workers,
                pin_memory=self._pin_memory,
            )
        return self._val_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        if self._test_dataloader is None:
            self._test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                pin_memory=self._pin_memory,
            )
        return self._test_dataloader


class HDF5Dataset(ABC):
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

    def from_file_cache(self) -> None:
        """
        Loads data from the cached dataset file.
        """
        logger.info(f"Loading cached dataset_file {self.processed_dataset_file}")
        self.numpy_dataset = self._load()

    def _load(self) -> np.ndarray:
        """
        Loads and returns the dataset from the npz file.

        Returns:
            np.ndarray: Loaded dataset.
        """
        return np.load(self.processed_dataset_file)

    def to_file_cache(self, data):
        """
        Saves the processed data to a file cache in npz format.

        Args:
            data (Dict[str, List]): Data to be saved to cache.
        """
        logger.info("Caching hdf5 file ...")
        self.to_npz(data)


class DatasetFactory:
    """Abstract base class for a dataset. This class is intended to be extended by
    specific datasets, providing common functionality to load and cache data.

    Attributes:
        raw_dataset_file (str): Path to the raw dataset file (hdf5 format).
        processed_dataset_file (str): Path to the processed dataset file (npz format).
        dataset (np.ndarray or None): Loaded dataset.
    """

    def __init__(
        self,
        # splitter: Optional[BaseSplittingStrategy] = None,
    ) -> None:
        """
        Initializes the Dataset class.

        Args:
            load_in_memory (bool): Whether to load the entire dataset into memory.
        """
        pass

    def load_or_process_data(self, dataset) -> None:
        """
        Loads the dataset from cache if available, otherwise processes and caches the data.
        """

        if not os.path.exists(dataset.processed_dataset_file):
            if not os.path.exists(dataset.raw_dataset_file):
                dataset.download_hdf_file()
            data = dataset.from_hdf5()
            dataset.to_file_cache(data)

        dataset.from_file_cache()

    def create_dataset(
        self,
        dataset: HDF5Dataset,
        splitting=RandomSplittingStrategy().split,
        batch_size=64,
        num_workers=4,
        pin_memory=False,
    ) -> Dataset:
        logger.info(f"Creating {dataset.dataset_name} dataset")
        self.load_or_process_data(dataset)
        return Dataset(
            dataset.numpy_dataset,
            splitting=splitting,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

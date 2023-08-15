import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import requests
import torch
from loguru import logger
import h5py
import numpy as np
from torch.utils.data import DataLoader


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Abstract Base Class representing a dataset."""

    def __init__(self, load_in_memory: bool) -> None:
        """Initialize the Dataset class."""
        self.raw_dataset_file = f"{self.dataset_name}_cache.hdf5"
        self.processed_dataset_file = f"{self.dataset_name}_processed.npz"
        with h5py.File(self.raw_dataset_file, "r") as hdf_fh:
            self.molecules = list(hdf_fh.keys())
        self.dataset = None
        self.load_or_process_data()

    def load_or_process_data(self) -> None:
        """Load dataset from cache, or process and cache if not available."""
        if os.path.exists(self.processed_dataset_file):
            self._load_cached_data()
        else:
            if not os.path.exists(self.raw_dataset_file):
                self.download_hdf_file()
            logger.info("Processing hdf5 file ...")
            self.process()
            logger.info("Caching hdf5 file ...")
            self._load_cached_data()

    def _load_cached_data(self) -> None:
        """Load data from the cached dataset file."""
        logger.info(f"Loading cached dataset_file {self.processed_dataset_file}")
        self.dataset = self._load()

    def _load(self) -> np.ndarray:
        """Load a numpy file."""
        return np.load(self.processed_dataset_file, allow_pickle=True)

    @abstractmethod
    def process(self):
        """Process the downloaded hdf5 file."""
        pass

    @abstractmethod
    def download_hdf_file(self):
        """Download the hdf5 file."""
        pass

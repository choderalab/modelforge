import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for a dataset. This class is intended to be extended by
    specific datasets, providing common functionality to load and cache data.

    Attributes:
        raw_dataset_file (str): Path to the raw dataset file (hdf5 format).
        processed_dataset_file (str): Path to the processed dataset file (npz format).
        dataset (np.ndarray or None): Loaded dataset.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes the Dataset class.

        Args:
            load_in_memory (bool): Whether to load the entire dataset into memory.
        """
        self.raw_dataset_file = f"{self.dataset_name}_cache.hdf5"
        self.processed_dataset_file = f"{self.dataset_name}_processed.npz"
        self.dataset = None

    @abstractmethod
    def load_or_process_data():
        raise NotImplementedError

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

    @classmethod
    def pad_to_max_length(
        cls, data: List[np.ndarray], max_length: int
    ) -> List[np.ndarray]:
        """
        Pad each array in the data list to a specified maximum length.

        Parameters:
        -----------
        data : List[np.ndarray]
            List of arrays to be padded.
        max_length : int
            Desired length for each array after padding.

        Returns:
        --------
        List[np.ndarray]
            List of padded arrays.
        """
        return [
            np.pad(arr, (0, max_length - len(arr)), "constant", constant_values=-1)
            for arr in data
        ]

    @classmethod
    def pad_molecules(cls, molecules: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pad molecules to ensure each has a consistent number of atoms.

        Parameters:
        -----------
        molecules : List[np.ndarray]
            List of molecules to be padded.

        Returns:
        --------
        List[np.ndarray]
            List of padded molecules.
        """
        max_atoms = max(mol.shape[0] for mol in molecules)
        return [
            np.pad(
                mol,
                ((0, max_atoms - mol.shape[0]), (0, 0)),
                mode="constant",
                constant_values=(-1, -1),
            )
            for mol in molecules
        ]

    def from_file_cache(self) -> None:
        """
        Loads data from the cached dataset file.
        """
        logger.info(f"Loading cached dataset_file {self.processed_dataset_file}")
        self.dataset = self._load()

    def _load(self) -> np.ndarray:
        """
        Loads and returns the dataset from the npz file.

        Returns:
            np.ndarray: Loaded dataset.
        """
        return np.load(self.processed_dataset_file)

    @abstractmethod
    def download_hdf_file(self):
        """
        Abstract method to download the hdf5 file. This method should be implemented by
        the child classes.

        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError

    def to_file_cache(self, data):
        """
        Saves the processed data to a file cache in npz format.

        Args:
            data (Dict[str, List]): Data to be saved to cache.
        """
        logger.info("Caching hdf5 file ...")
        self.to_npz(data)

    @staticmethod
    def is_gzipped(filename) -> bool:
        with open(filename, "rb") as f:
            # Read the first two bytes of the file
            file_start = f.read(2)

        return True if file_start == b"\x1f\x8b" else False

    def decompress_gziped_file(
        self, compressed_file: str, uncompressed_file: str
    ) -> None:
        import gzip
        import shutil

        with gzip.open(f"{compressed_file}", "rb") as f_in:
            with open(f"{uncompressed_file}", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

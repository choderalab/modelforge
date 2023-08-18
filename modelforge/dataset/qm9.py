import os
from typing import Any, Dict, List, Tuple

import gdown
import h5py
import numpy as np
import torch
from loguru import logger
from .utils import PadTensors, is_gzipped, decompress_gziped_file
from .dataset import HDF5Dataset


class QM9Dataset(HDF5Dataset):
    """
    Dataset class for handling QM9 data.

    Provides utilities for processing and interacting with QM9 data stored in hdf5 format.
    Also allows for lazy loading of data or caching in memory for faster operations.
    """

    def __init__(
        self,
        dataset_name: str = "QM9",
        for_testing: bool = False,
    ) -> None:
        """
        Initialize the QM9Dataset class.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset, default is "QM9".
        load_in_memory : bool
            Flag to determine if the dataset should be loaded into memory, default is True.
        test_data : bool
            If set to true it will only load a small fraction of the QM9 dataset.
        """

        if for_testing:
            dataset_name = f"{dataset_name}_subset"
        self.dataset_name = dataset_name
        self.keywords_for_hdf5_dataset = ["geometry", "atomic_numbers", "return_energy"]
        self.for_testing = for_testing
        self.raw_dataset_file = f"{dataset_name}_cache.hdf5"
        self.processed_dataset_file = f"{dataset_name}_processed.npz"

    def to_npz(self, data: Dict[str, Any]) -> None:
        """
        Save processed data to a numpy (.npz) file.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary containing processed data to be saved.
        """
        max_len_species = max(len(arr) for arr in data["atomic_numbers"])

        padded_coordinates = PadTensors.pad_molecules(data["geometry"])
        padded_atomic_numbers = PadTensors.pad_to_max_length(
            data["atomic_numbers"], max_len_species
        )
        logger.debug(f"Writing data cache to {self.processed_dataset_file}")

        np.savez(
            self.processed_dataset_file,
            coordinates=padded_coordinates,
            atomic_numbers=padded_atomic_numbers,
            return_energy=np.array(data["return_energy"]),
        )

    def _download_from_gdrive(self):
        """Internal method to download the dataset from Google Drive."""

        test_id = "13ott0kVaCGnlv858q1WQdOwOpL7IX5Q9"
        full_id = "1_bSdQjEvI67Tk_LKYbW0j8nmggnb5MoU"
        if self.for_testing:
            logger.debug("Downloading test data")
            id = test_id
        else:
            logger.debug("Downloading full dataset")

            id = full_id
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, self.raw_dataset_file, quiet=False)

        if is_gzipped(self.raw_dataset_file):
            logger.debug("Decompressing gzipped file")
            os.rename(f"{self.raw_dataset_file}", f"{self.raw_dataset_file}.gz")
            decompress_gziped_file(f"{self.raw_dataset_file}.gz", self.raw_dataset_file)

    def download_hdf_file(self):
        """
        Download the hdf5 file containing the dataset.

        Fetches the dataset from the specified source (Google Drive in this case)
        and saves it in hdf5 format.
        """
        self._download_from_gdrive()

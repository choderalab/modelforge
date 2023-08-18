import os
from typing import Any, Dict, List, Tuple

import gdown
import numpy as np
from loguru import logger
from .utils import PadTensors, is_gzipped, decompress_gziped_file
from .dataset import HDF5Dataset


class DatasetDownloader:
    """
    Utility class for downloading datasets.
    """

    @staticmethod
    def download_from_gdrive(for_testing: bool, raw_dataset_file: str):
        """
        Downloads a dataset from Google Drive.

        Args:
            for_testing (bool): If True, downloads a test subset of the dataset. Otherwise, downloads the full dataset.
            raw_dataset_file (str): Path to save the downloaded dataset.
        """

        test_id = "13ott0kVaCGnlv858q1WQdOwOpL7IX5Q9"
        full_id = "1_bSdQjEvI67Tk_LKYbW0j8nmggnb5MoU"
        id = test_id if for_testing else full_id
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, raw_dataset_file, quiet=False)

        if is_gzipped(raw_dataset_file):
            logger.debug("Decompressing gzipped file")
            os.rename(f"{raw_dataset_file}", f"{raw_dataset_file}.gz")
            decompress_gziped_file(f"{raw_dataset_file}.gz", raw_dataset_file)


class QM9Dataset(HDF5Dataset):
    """
    Dataset class for handling QM9 data. Provides utilities for processing and
    interacting with QM9 data stored in hdf5 format.
    """

    def __init__(
        self,
        dataset_name: str = "QM9",
        for_testing: bool = False,
    ) -> None:
        """
        Initialize the QM9Dataset class.

        Args:
            dataset_name (str): Name of the dataset, default is "QM9".
            for_testing (bool): If set to True, a subset of the dataset is used for testing purposes.
        """

        if for_testing:
            dataset_name = f"{dataset_name}_subset"

        super().__init__(f"{dataset_name}_cache.hdf5", f"{dataset_name}_processed.npz")
        self.dataset_name = dataset_name
        self.keywords_for_hdf5_dataset = ["geometry", "atomic_numbers", "return_energy"]
        self.for_testing = for_testing

    def download_hdf_file(self):
        """
        Download the hdf5 file containing the dataset from Google Drive.
        """

        DatasetDownloader.download_from_gdrive(self.for_testing, self.raw_dataset_file)

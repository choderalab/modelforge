import os
from typing import Any, Dict, List, Tuple

import numpy as np
from loguru import logger
from .utils import DataDownloader
from .dataset import HDF5Dataset


class QM9Dataset(HDF5Dataset, DataDownloader):
    """
    Data class for handling QM9 data.

    This class provides utilities for processing and interacting with QM9 data
    stored in HDF5 format.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "QM9".
    for_unit_testing : bool
        If set to True, a subset of the dataset is used for unit testing purposes; by default False.

    Examples
    --------
    >>> data = QM9Dataset()
    >>> data._download()
    """

    def __init__(
        self,
        dataset_name: str = "QM9",
        for_unit_testing: bool = False,
        local_cache_dir: str = ".",
    ) -> None:
        """
        Initialize the QM9Data class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "QM9".
        for_unit_testing : bool, optional
            If set to True, a subset of the dataset is used for unit testing purposes; by default False.
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".

        Examples
        --------
        >>> data = QM9Dataset()  # Default dataset
        >>> test_data = QM9Dataset(for_unit_testing=True)  # Testing subset
        """

        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        super().__init__(
            f"{local_cache_dir}/{dataset_name}_cache.hdf5",
            f"{local_cache_dir}/{dataset_name}_processed.npz",
        )
        self.dataset_name = dataset_name
        self.keywords_for_hdf5_data = ["geometry", "atomic_numbers", "return_energy"]
        self.for_unit_testing = for_unit_testing
        self.test_id = "17oZ07UOxv2fkEmu-d5mLk6aGIuhV0mJ7"
        self.full_id = "1_bSdQjEvI67Tk_LKYbW0j8nmggnb5MoU"

    def download(self) -> None:
        """
        Download the hdf5 file containing the data from Google Drive.

        Examples
        --------
        >>> data = QM9Dataset()
        >>> data.download()  # Downloads the dataset from Google Drive

        """
        id = self.test_id if self.for_unit_testing else self.full_id
        self._download_from_gdrive(id, self.raw_data_file)

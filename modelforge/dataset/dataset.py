import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger


class TorchDataset(torch.utils.data.Dataset):
    """
    A custom dataset class to wrap numpy datasets for PyTorch.

    Parameters
    ----------
    dataset : np.ndarray
        The underlying numpy dataset.
    prop : List[str]
        List of property names to extract from the dataset.
    preloaded : bool, optional
        If True, preconverts the properties to PyTorch tensors to save time during item fetching.
        Default is False.

    Examples
    --------
    >>> numpy_data = np.load("data_file.npz")
    >>> properties = ["geometry", "atomic_numbers"]
    >>> torch_dataset = TorchDataset(numpy_data, properties)
    >>> data_point = torch_dataset[0]
    """

    def __init__(self, dataset: np.ndarray, prop: List[str], preloaded: bool = False):
        self.properties_of_interest = [dataset[p] for p in prop]
        self.length = len(dataset[prop[0]])
        self.preloaded = preloaded

        if preloaded:
            self.properties_of_interest = [
                torch.tensor(p) for p in self.properties_of_interest
            ]

    def __len__(self) -> int:
        """
        Return the number of datapoints in the dataset.

        Returns:
        --------
        int
            Total number of datapoints available in the dataset.
        """
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Fetch a tuple of the values for the properties of interest for a given molecule index.

        Parameters
        ----------
        idx : int
            Index of the molecule to fetch data for.

        Returns
        -------
        Tuple[torch.Tensor]
            Tuple containing tensors for properties of interest of the molecule.

        Examples
        --------
        >>> data_point = torch_dataset[5]
        >>> geometry, atomic_numbers = data_point
        """
        if self.preloaded:
            return tuple(prop[idx] for prop in self.properties_of_interest)
        else:
            return tuple(
                torch.tensor(prop[idx]) for prop in self.properties_of_interest
            )


class HDF5Dataset(ABC):
    """
    Base class for data stored in HDF5 format.

    Provides methods for processing and interacting with the data stored in HDF5 format.

    Attributes
    ----------
    raw_data_file : str
        Path to the raw data file.
    processed_data_file : str
        Path to the processed data file.
    """

    def __init__(self, raw_data_file: str, processed_data_file: str):
        self.raw_data_file = raw_data_file
        self.processed_data_file = processed_data_file

    def _from_hdf5(self) -> Dict[str, List]:
        """
        Processes and extracts data from an hdf5 file.

        Returns
        -------
        Dict[str, List]
            Processed data from the hdf5 file.

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> processed_data = hdf5_data._from_hdf5()

        """
        import h5py
        import tqdm
        import gzip

        logger.debug("Reading in and processing hdf5 file ...")
        data = defaultdict(list)
        logger.debug(f"Processing and extracting data from {self.raw_data_file}")
        with gzip.open(self.raw_data_file, "rb") as gz_file, h5py.File(
            gz_file, "r"
        ) as hf:
            logger.debug(f"n_entries: {len(hf.keys())}")
            for mol in tqdm.tqdm(list(hf.keys())):
                for value in self.properties_of_interest:
                    data[value].append(hf[mol][value][()])
        return data


class DatasetFactory:
    """
    Factory class for creating Dataset instances.

    Provides utilities for processing and caching data.

    Examples
    --------
    >>> factory = DatasetFactory()
    >>> qm9_data = QM9Data()
    >>> torch_dataset = factory.create_dataset(qm9_data)
    """

    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def _load_or_process_data(data: HDF5Dataset) -> None:
        """
        Loads the dataset from cache if available, otherwise processes and caches the data.

        Parameters
        ----------
        dataset : HDF5Dataset
            The HDF5 dataset instance to use.
        """
        from .utils import _to_file_cache, _from_file_cache

        # if not cached, download and process
        if not os.path.exists(data.processed_data_file):
            if not os.path.exists(data.raw_data_file):
                data.download()
            # load from hdf5 and process
            numpy_data = data._from_hdf5()
            # save to cache
            _to_file_cache(numpy_data, data.processed_data_file)
        # load from cache
        data.numpy_data = _from_file_cache(data.processed_data_file)

    @staticmethod
    def create_dataset(
        data: HDF5Dataset,
    ) -> TorchDataset:
        """
        Creates a Dataset instance given an HDF5Dataset.

        Parameters
        ----------
        data : HDF5Dataset
            The HDF5 data to use.

        Returns
        -------
        TorchDataset
            Dataset instance wrapped for PyTorch.
        """

        logger.info(f"Creating {data.dataset_name} dataset")
        DatasetFactory._load_or_process_data(data)
        return TorchDataset(data.numpy_data, data.properties_of_interest)

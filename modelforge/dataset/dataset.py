import os
from abc import ABC, abstractmethod
from typing import List

import torch


"""
The Dataset class defines the interface for datasets as a ABC.


"""

class Dataset(torch.utils.data.Dataset, ABC):
    """
    Abstract Base Class representing a dataset.
    """

    def __init__(self):
        pass
    # Using qcarchive won't necessarily require this
    # but if we are trying to create a general class
    # that would load datasets from other formats
    # it might be useful.
    def set_cache_dir(self, cache_dir: str) -> None:
        """
        Sets directory to cache files.

        Parameters
        ----------
        cache_dir: str
             path to the directory

        """
        self._cache_dir = cache_dir

    @property
    def name(self) -> str:
        """
        Returns the name of the dataset.

        Returns:
            str: name of the dataset
        """
        return self.name

    # I this this should be an abstract class
    # because we should include extension checking here
    # also so that we can update the docstrings to
    # reflect the expected inputs
    @abstractmethod
    def set_raw_dataset_file(self, raw_input_file: str):
        """
        Defines raw dataset input.

        Parameters
        ----------
            raw_input : str
                file path to the input raw dataset
        """
        self._raw_input_file = raw_input_file


    @property
    def raw_dataset_file(self) -> str:
        """
        Returns the path and name of the raw dataset file.

        Returns:
            str: path dataset
        """
        return self._raw_input_file

    # I think calling this cache would avoid confusing
    # it would the idea of saving the file torch composed file.
    # For qcarchive, this will just be saving as an hdf5 file
    # that we could load in via set_raw_dataset_file instead of downloading again.
    # this is distinct from setting up a cache dir
    @abstractmethod
    def save_raw_dataset(self, raw_output_file: str):
        """
        Defines the hdf5 file for saving save the raw dataset.

        Loading this file via set_raw_dataset_file will avoid
        the need to re-download the data.

        Parameters
        ----------
            raw_output_file : str
                file path to save raw dataset
        """
        pass
    @abstractmethod
    def load(self):
        """
        Loads the dataset from the cache.
        """
        pass

    @abstractmethod
    def prepare_dataset(self):
        pass

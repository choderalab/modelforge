from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Subset, random_split

if TYPE_CHECKING:
    from modelforge.dataset.dataset import TorchDataset


class SplittingStrategy(ABC):
    """
    Base class for dataset splitting strategies.

    Attributes
    ----------
    seed : int, optional
        Random seed for reproducibility.
    generator : torch.Generator, optional
        Torch random number generator.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        self.seed = seed
        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)

    @abstractmethod
    def split():
        """
        Split the dataset.

        Returns
        -------
        List[List[int]]
            List of indices for each split.
        """

        raise NotImplementedError


class RandomSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset randomly.

    Examples
    --------
    >>> dataset = [1, 2, 3, 4, 5]
    >>> strategy = RandomSplittingStrategy(seed=42)
    >>> train_idx, val_idx, test_idx = strategy.split(dataset)
    """

    def __init__(self, seed: int = 42, split: List[float] = [0.8, 0.1, 0.1]):
        """
        Initializes the RandomSplittingStrategy with a specified seed and split ratios.

        This strategy splits a dataset randomly based on provided ratios for training, validation,
        and testing subsets. The sum of split ratios should be 1.0.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.
        split : List[float], optional
            List containing three float values representing the ratio of data for
            training, validation, and testing respectively, by default [0.8, 0.1, 0.1].

        Raises
        ------
        AssertionError
            If the sum of split ratios is not close to 1.0.

        Examples
        --------
        >>> random_strategy_default = RandomSplittingStrategy()
        >>> random_strategy_custom = RandomSplittingStrategy(seed=123, split=[0.7, 0.2, 0.1])
        """

        super().__init__(seed)
        self.train_size, self.val_size, self.test_size = split[0], split[1], split[2]
        assert np.isclose(sum(split), 1.0), "Splits must sum to 1.0"

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Splits the provided dataset into training, validation, and testing subsets based on the predefined ratios.

        This method uses the ratios defined during initialization to randomly partition the dataset.
        The result is a tuple of indices for each subset.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset to be split.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            A tuple containing three Subsets for training, validation, and testing subsets, respectively.

        Examples
        --------
        >>> dataset = TorchDataset(numpy_data)
        >>> random_strategy = RandomSplittingStrategy(seed=42, split=[0.7, 0.2, 0.1])
        >>> train_dataset, val_dataset, test_dataset = random_strategy.split(dataset)
        """

        logger.debug(f"Using random splitting strategy with seed {self.seed} ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        train_d, val_d, test_d = random_split(
            dataset,
            lengths=[self.train_size, self.val_size, self.test_size],
            generator=self.generator,
        )

        return (train_d, val_d, test_d)


class FirstComeFirstServeSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset based on idx.

    Examples
    --------
    >>> dataset = [1, 2, 3, 4, 5]
    >>> strategy = FirstComeFirstServeSplittingStrategy()
    >>> train_idx, val_idx, test_idx = strategy.split(dataset)
    """

    def __init__(self, split: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(42)
        self.train_size, self.val_size, self.test_size = split[0], split[1], split[2]
        assert np.isclose(sum(split), 1.0), "Splits must sum to 1.0"

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        logger.debug(f"Using first come/first serve splitting strategy ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        len_dataset = len(dataset)
        first_split_on = int(len_dataset * self.train_size)
        second_split_on = int(len_dataset * self.val_size)
        indices = np.arange(len_dataset, dtype=int)
        train_d, val_d, test_d = (
            Subset(dataset, indices[0:first_split_on]),
            Subset(dataset, indices[first_split_on:second_split_on]),
            Subset(dataset, indices[second_split_on:]),
        )

        return (train_d, val_d, test_d)


def _download_from_gdrive(id: str, raw_dataset_file: str):
    """
    Downloads a dataset from Google Drive.

    Parameters
    ----------
    id : str
        Google Drive ID for the dataset.
    raw_dataset_file : str
        Path to save the downloaded dataset.

    Examples
    --------
    >>> _download_from_gdrive("1v2gV3sG9JhMZ5QZn3gFB9j5ZIs0Xjxz8", "data_file.hdf5.gz")
    """
    import gdown

    url = f"https://drive.google.com/uc?id={id}"
    gdown.download(url, raw_dataset_file, quiet=False)


def _to_file_cache(
    data: OrderedDict[str, List[np.ndarray]], processed_dataset_file: str
) -> None:
    """
    Save processed data to a numpy (.npz) file.

    Parameters
    ----------
    data : OrderedDict[str, List[np.ndarray]]
        Dictionary containing processed data to be saved.
    processed_dataset_file : str
        Path to save the processed dataset.

    Examples
    --------
    >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    >>> _to_file_cache(data, "data_file.npz")
    """

    for prop_key in data:
        if prop_key not in data:
            raise ValueError(f"Property {prop_key} not found in data")
        if prop_key == "geometry":  # NOTE: here a 2d tensor is padded
            logger.debug(prop_key)
            data[prop_key] = pad_molecules(data[prop_key])
        else:
            logger.debug(data[prop_key])
            logger.debug(prop_key)
            try:
                max_len_species = max(len(arr) for arr in data[prop_key])
            except TypeError:
                continue
            data[prop_key] = pad_to_max_length(data[prop_key], max_len_species)

    logger.debug(f"Writing data cache to {processed_dataset_file}")

    np.savez(
        processed_dataset_file,
        **data,
    )


def pad_to_max_length(data: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pad each array in the data list to a specified maximum length.

    Parameters
    ----------
    data : List[np.ndarray]
        List of arrays to be padded.
    Returns
    -------
    List[np.ndarray]
        List of padded arrays.
    """
    max_length = max(arr.shape[0] for arr in data)

    return [
        np.pad(arr, (0, max_length - arr.shape[0]), "constant", constant_values=0)
        for arr in data
    ]


def pad_molecules(molecules: List[np.ndarray]) -> List[np.ndarray]:
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
    # I don't think this is quite right.  As this is going to create a 2D array. why are we padding
    return [
        np.pad(
            mol,
            ((0, max_atoms - mol.shape[0]), (0, 0)),
            mode="constant",
            constant_values=(-1, -1),
        )
        for mol in molecules
    ]



translate_property_names = {"return_energy": "U0"}

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import random_split, Subset

from .dataset import TorchDataset


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

    def split(self, dataset: TorchDataset) -> Tuple[Subset, Subset, Subset]:
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


class DataDownloader(ABC):
    """
    Utility class for downloading datasets.
    """

    @staticmethod
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
        >>> _download_from_gdrive("1v2gV3sG9JhMZ5QZn3gFB9j5ZIs0Xjxz8", "data_file.hdf5")
        """
        import gdown

        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, raw_dataset_file, quiet=False)

    @abstractmethod
    def download():
        raise NotImplementedError


def _from_file_cache(processed_dataset_file: str) -> np.ndarray:
    """
    Loads the dataset from a cached file.

    Parameters
    ----------
    processed_dataset_file : str
        Path to the cached dataset file.

    Returns
    -------
    np.ndarray
        The loaded dataset.

    Examples
    --------
    >>> data = _from_file_cache("data_file.npz")
    """
    logger.info(f"Loading cached dataset_file {processed_dataset_file}")
    return np.load(processed_dataset_file)


def _to_file_cache(
    data: Dict[str, List[np.ndarray]], processed_dataset_file: str
) -> None:
    """
    Save processed data to a numpy (.npz) file.

    Parameters
    ----------
    data : Dict[str, List[np.ndarray]]
        Dictionary containing processed data to be saved.
    processed_dataset_file : str
        Path to save the processed dataset.

    Examples
    --------
    >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    >>> _to_file_cache(data, "data_file.npz")
    """
    max_len_species = max(len(arr) for arr in data["atomic_numbers"])

    padded_coordinates = pad_molecules(data["geometry"])
    padded_atomic_numbers = pad_to_max_length(data["atomic_numbers"], max_len_species)
    logger.debug(f"Writing data cache to {processed_dataset_file}")

    np.savez(
        processed_dataset_file,
        geometry=padded_coordinates,
        atomic_numbers=padded_atomic_numbers,
        return_energy=np.array(data["return_energy"]),
    )


def pad_to_max_length(data: List[np.ndarray], max_length: int) -> List[np.ndarray]:
    """
    Pad each array in the data list to a specified maximum length.

    Parameters
    ----------
    data : List[np.ndarray]
        List of arrays to be padded.
    max_length : int
        Desired length for each array after padding.

    Returns
    -------
    List[np.ndarray]
        List of padded arrays.
    """
    return [
        np.pad(arr, (0, max_length - len(arr)), "constant", constant_values=-1)
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
    return [
        np.pad(
            mol,
            ((0, max_atoms - mol.shape[0]), (0, 0)),
            mode="constant",
            constant_values=(-1, -1),
        )
        for mol in molecules
    ]

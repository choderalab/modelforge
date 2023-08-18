# generates and retrieves hdf5 files from zenodo, defines the interaction with the hdf5 file
from torch.utils.data import random_split
import torch
from typing import List
import numpy as np
from loguru import logger


class SplittingStrategy:
    def __init__(
        self,
        seed: int,
    ):
        self.seed = seed
        torch.manual_seed(self.seed)

    def split():
        raise NotImplementedError


class RandomSplittingStrategy(SplittingStrategy):
    def __init__(self, seed: int = 42, split: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(seed)
        self.train_size = split[0]
        self.val_size = split[1]
        self.test_size = split[2]
        assert np.isclose(sum(split), 1.0)

    def split(self, dataset) -> List[List[int]]:
        torch.Generator().manual_seed(self.seed)

        logger.debug(f"Using random splitting strategy with seed {self.seed} ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        train_d, val_d, test_d = random_split(
            dataset,
            lengths=[self.train_size, self.val_size, self.test_size],
        )

        return (train_d, val_d, test_d)


class PadTensors:
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


def is_gzipped(filename) -> bool:
    with open(filename, "rb") as f:
        # Read the first two bytes of the file
        file_start = f.read(2)

    return True if file_start == b"\x1f\x8b" else False


def decompress_gziped_file(compressed_file: str, uncompressed_file: str) -> None:
    import gzip
    import shutil

    logger.debug(f"Unzipping {compressed_file} to {uncompressed_file} ")
    with gzip.open(f"{compressed_file}", "rb") as f_in:
        with open(f"{uncompressed_file}", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

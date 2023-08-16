from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple

import gdown
import h5py
import numpy as np
import torch
from loguru import logger

from .dataset import BaseDataset


class PlainQM9Dataset(torch.utils.data.Dataset):
    """
    Abstract Base Class representing a dataset.
    load_data: If True, loads all the data immediately into RAM. Use this if
    the dataset is fits into memory. Otherwise, leave this at false and
    the data will load lazily.

    """

    def __init__(self, dataset_name: str = "QM9", load_in_memory: bool = False) -> None:
        """initialize the Dataset class."""
        self.dataset_name = dataset_name
        self.raw_dataset_file: str = f"{self.dataset_name}_cache.hdf5"
        self.processed_dataset_file: str = f"{self.dataset_name}_processed.hdf5"
        self.chunk_size: int = 500
        self.hdf_fh = h5py.File(f"{self.raw_dataset_file}", "r")
        self.molecules = [mol for mol in self.hdf_fh.keys()]
        self.molecule_cache = {}

        logger.debug(f"Nr of mols: {self.__len__()}")
        print(f"Nr of mols: {self.__len__()}")

        if load_in_memory:
            self.cache_in_memory()

    def cache_in_memory(self):
        """load the dataset into memory."""

        for mol_idx in range(len(self.molecules)):
            self.molecule_cache[mol_idx] = (
                self.hdf_fh[self.molecules[mol_idx]]["geometry"][()],
                self.hdf_fh[self.molecules[mol_idx]]["return_energy"][()],
                self.hdf_fh[self.molecules[mol_idx]]["atomic_numbers"][()],
            )
        self.hdf_fh.close()
        self.hdf_fh = None

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Tuple[0] with dim=3, Tuple[1] with dim=1
        """pytorch dataset getitem method to return a tuple of geometry and energy.
        if a .

        Args:
            idx (int): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns a tuple of geometry and energy
        """
        if self.load_in_memory:
            return self.molecule_cache[idx]

        geometry = torch.tensor(self.hdf_fh[self.molecules[idx]]["geometry"][()])
        energy = torch.tensor(self.hdf_fh[self.molecules[idx]]["return_energy"][()])
        species = torch.tensor(self.hdf_fh[self.molecules[idx]]["atomic_numbers"][()])
        return (species, geometry, energy)


class QM9Dataset(BaseDataset):
    """
    Dataset class for handling QM9 data.

    Provides utilities for processing and interacting with QM9 data stored in hdf5 format.
    Also allows for lazy loading of data or caching in memory for faster operations.
    """

    def __init__(self, dataset_name: str = "QM9", load_in_memory: bool = True) -> None:
        """
        Initialize the QM9Dataset class.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset, default is "QM9".
        load_in_memory : bool
            Flag to determine if the dataset should be loaded into memory, default is True.
        """
        self.dataset_name = dataset_name
        self.keywords_for_hdf5_dataset = ["geometry", "atomic_numbers", "return_energy"]
        super().__init__(load_in_memory)

    def to_npz(self, data: Dict[str, Any]) -> None:
        """
        Save processed data to a numpy (.npz) file.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary containing processed data to be saved.
        """
        max_len_species = max(len(arr) for arr in data["atomic_numbers"])

        padded_coordinates = self._pad_molecules(data["geometry"])
        padded_atomic_numbers = self._pad_to_max_length(
            data["atomic_numbers"], max_len_species
        )
        logger.debug(f"Writing data cache to {self.processed_dataset_file}")

        np.savez(
            self.processed_dataset_file,
            coordinates=padded_coordinates,
            atomic_numbers=padded_atomic_numbers,
            return_energy=np.array(data["return_energy"]),
        )

    @staticmethod
    def _pad_molecules(molecules: List[np.ndarray]) -> List[np.ndarray]:
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

    def _download_from_gdrive(self):
        """Internal method to download the dataset from Google Drive."""

        id = "1h3eh-79wQy69_I7Fr-BoYNvHW6wYisPc"
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, self.raw_dataset_file, quiet=False)

    def download_hdf_file(self):
        """
        Download the hdf5 file containing the dataset.

        Fetches the dataset from the specified source (Google Drive in this case)
        and saves it in hdf5 format.
        """
        self._download_from_gdrive()

    @staticmethod
    def _pad_to_max_length(data: List[np.ndarray], max_length: int) -> List[np.ndarray]:
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

    def __len__(self) -> int:
        """
        Return the number of datapoints in the dataset.

        Returns:
        --------
        int
            Total number of datapoints available in the dataset.
        """
        return len(self.dataset["atomic_numbers"])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch a tuple of geometry, atomic numbers, and energy for a given molecule index.

        Parameters:
        -----------
        idx : int
            Index of the molecule to fetch data for.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing tensors for geometry, atomic numbers, and energy of the molecule.
        """
        return (
            torch.tensor(self.dataset["coordinates"][idx]),
            torch.tensor(self.dataset["atomic_numbers"][idx]),
            torch.tensor(self.dataset["return_energy"][idx]),
        )

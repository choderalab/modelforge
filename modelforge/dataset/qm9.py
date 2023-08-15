from collections import defaultdict
from typing import Any, Dict, Generator, List, Tuple

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
    """Dataset class for QM9."""

    def __init__(self, dataset_name: str = "QM9", load_in_memory: bool = True) -> None:
        """Initialize the QM9Dataset class."""
        self.dataset_name = dataset_name
        self.keywords_for_hdf5_dataset = ["geometry", "atomic_numbers", "return_energy"]
        super().__init__(load_in_memory)

    def process(self) -> None:
        """Process the downloaded hdf5 file."""
        data = defaultdict(list)
        with h5py.File(self.raw_dataset_file, "r") as hf:
            for mol in list(hf.keys()):
                for value in self.keywords_for_hdf5_dataset:
                    data[value].append(hf[mol][value][()])
        self._save_npz(data)

    def _save_npz(self, data: Dict[str, Any]) -> None:
        """Save data to a numpy file in chunks."""
        max_len_species = max(len(arr) for arr in data["atomic_numbers"])

        padded_coordinates = self._pad_molecules(data["geometry"])
        padded_atomic_numbers = self._pad_to_max_length(
            data["atomic_numbers"], max_len_species
        )

        np.savez(
            self.processed_dataset_file,
            coordinates=padded_coordinates,
            atomic_numbers=padded_atomic_numbers,
            return_energy=np.array(data["return_energy"]),
        )

    @staticmethod
    def _pad_molecules(molecules: List[np.ndarray]) -> List[np.ndarray]:
        """Pad molecules to have the same number of atoms."""
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
        import gdown

        id = "1h3eh-79wQy69_I7Fr-BoYNvHW6wYisPc"
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, self.raw_dataset_file, quiet=False)

    def download_hdf_file(self):
        """
        Download the hdf5 file.
        """
        # Send a GET request to the URL
        self._download_from_gdrive()

    @staticmethod
    def _pad_to_max_length(data: List[np.ndarray], max_length: int) -> List[np.ndarray]:
        """Pad each array in the data list to the specified max length with -1."""
        return [
            np.pad(arr, (0, max_length - len(arr)), "constant", constant_values=-1)
            for arr in data
        ]

    def __len__(self) -> int:
        """Return the number of molecules."""
        return len(self.dataset["atomic_numbers"])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of geometry, atomic numbers, and energy for a given index."""
        return (
            torch.tensor(self.dataset["coordinates"][idx]),
            torch.tensor(self.dataset["atomic_numbers"][idx]),
            torch.tensor(self.dataset["return_energy"][idx]),
        )

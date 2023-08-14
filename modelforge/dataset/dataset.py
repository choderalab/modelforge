import os
from abc import ABC, abstractmethod
from typing import Tuple
import requests
import qcportal as ptl
import torch
from loguru import logger
import h5py
import numpy as np
from typing import Tuple, List, Dict, Any, Generator


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract Base Class representing a dataset.
    """

    def __init__(
        self,
    ) -> None:
        """initialize the Dataset class."""
        self.raw_dataset_file: str = f"{self.dataset_name}_cache.hdf5"
        self.processed_dataset_file: str = self.dataset_name + "_processed.npy"
        self.chunk_size: int = 500
        self._load_or_process_data()

    def _load_cached_data(self) -> None:
        """Load data from the cached dataset file."""
        logger.info(f"Loading cached dataset_file {self.processed_dataset_file}")
        self.dataset = self.load(self.processed_dataset_file)

    def _load_or_process_data(self) -> None:
        """Load dataset from cache, or process and cache if not available."""
        if os.path.exists(self.processed_dataset_file):
            self._load_cached_data()
        else:
            if not os.path.exists(self.raw_dataset_file):
                self.download_hdf_file()
            logger.info(f"Processing hdf5 file ...")
            self.process()
            logger.info(f"Caching hdf5 file ...")
            self.cache()
            self._load_cached_data()

        # records_properties = self._dataset_records
        # (
        #     self.molecules,
        #     self.records,
        # ) = self.dataset.get_molecules(), self.dataset.get_records(
        #     **records_properties,
        # )

    def _save_npz(self, r: Dict[str, Any], idx: int) -> None:
        """Save data to a numpy file in chunks."""
        np.savez(
            f"dataset_{idx}.npy",
            coordinates=np.array(r["coordinates"], dtype=object),
            atomic_numbers=np.array(r["atomic_numbers"], dtype=object),
            return_energy=np.array(r["E0"], dtype=object),
        )

    def _process_hdf_file(self):
        """Process the raw HDF5 file and convert data into chunks."""
        from collections import defaultdict

        def generate_an_iterator_on_a_junk(buf: List[Any], chunk: int):
            """Generate chunks from a buffer."""
            for start in range(0, len(buf), chunk):
                yield buf[start : start + chunk]

        with h5py.File(f"{self.raw_dataset_file}", "r") as hf:
            print("n_entries ", len(hf.keys()))

            mols = [mol for mol in hf.keys()]
            iterator = generate_an_iterator_on_a_junk(mols, self.chunk_size)
            r = defaultdict(list)
            count = 0

            for idx, mols in enumerate(iterator):
                logger.debug(f"Processing chunk {idx} with {len(mols)} mols")
                for mol in mols:
                    count += 1
                    for key, value in self.keywords_for_hdf5_dataset.items():
                        r[key].append(hf[mol][value][()])
                self._save_npz(r, idx)
            logger.debug(f"Nr of mols: {count}")

    def process(self):
        """
        Process the downloaded hdf5 file.
        """
        self._process_hdf_file()

    def download_hdf_file(self):
        """
        Download the hdf5 file.
        """
        # Send a GET request to the URL
        logger.info(f"Downloading hdf5 file from {self.url}")

        try:
            response = requests.get(self.url, stream=True)

            # Check if the request was successful
            response.raise_for_status()

            # Save the content to the output file
            logger.debug(f"Saving hdf5 file to {self.raw_dataset_file}")
            with open(self.raw_dataset_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
        except requests.RequestException as e:
            print(f"Error downloading the file: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    @property
    @abstractmethod
    def _dataset_records(self)->Dict[str, Any]::
        """
        Defines the dataset to be downloaded.
        Multiple keys can be defined and provided, e.g.: 'method', 'basis', 'program'.
        For an exhaustive list see https://docs.qcarchive.molssi.org/projects/QCPortal/en/stable/collection-dataset.html#qcportal.collections.Dataset.get_records
        """
        pass

    @abstractmethod
    def transform_y(self, y: torch.Tensor) -> torch.Tensor:

        """
        Abstract method to transform the y values. This is necessary if e.g. multiple values are returned by the dataset query.
        """
        pass

    @abstractmethod
    def transform_x(self, x: torch.Tensor) -> torch.Tensor:

        """
        Abstract method to transform the x values.
        """
        pass

    @abstractmethod
    def load(self, x: torch.Tensor) -> torch.Tensor:

        """
        Abstract method to load the dataset from a hdf5 file.
        """
        pass

    @property
    def cache_file_name(self)->str:
        return f"{self.dataset_name}_cache.np"

    def to_npy_cache(self):
        """
        Save the dataset to a npy file.
        """
        np.save(self.cache_file_name, self.dataset)

    def from_cache(self):
        self._parse_hdf()

    def __len__(self)->int:
        return self.molecules.shape[0]

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
        geometry = torch.tensor(self.records.iloc[idx].record["geometry"])
        energy = torch.tensor(self.records.iloc[idx].record["energy"])

        if self.transform_x:
            geometry = self.transform_x(geometry)
        if self.transform_y:
            energy = self.transform_y(energy)

        return geometry, energy

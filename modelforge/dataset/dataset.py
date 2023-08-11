import os
from abc import ABC, abstractmethod
from typing import Tuple

import qcportal as ptl
import torch
from loguru import logger


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract Base Class representing a dataset.
    """

    def __init__(
        self,
        dataset_file: str = "",
    ):
        """initialize the Dataset class.

        Args:
            dataset_file (str, optional): The file location where the dataset is cached. Defaults to "".
        """
        self.dataset_file = dataset_file
        self.dataset = self.load(dataset_file)
        records_properties = self._dataset_records
        (
            self.molecules,
            self.records,
        ) = self.dataset.get_molecules(), self.dataset.get_records(
            **records_properties,
        )

    @property
    @abstractmethod
    def _dataset_records(self):
        """
        Defines the dataset to be downloaded.
        Multiple keys can be defined and provided, e.g.: 'method', 'basis', 'program'.
        For an exhaustive list see https://docs.qcarchive.molssi.org/projects/QCPortal/en/stable/collection-dataset.html#qcportal.collections.Dataset.get_records
        """
        pass

    @abstractmethod
    def transform_y():
        """
        Abstract method to transform the y values. This is necessary if e.g. multiple values are returned by the dataset query.
        """
        pass

    @abstractmethod
    def transform_x():
        """
        Abstract method to transform the x values.
        """
        pass

    @abstractmethod
    def load():
        """
        Abstract method to load the dataset from a hdf5 file.
        """
        pass

    @abstractmethod
    def to_cache():
        pass

    @abstractmethod
    def from_cache():
        pass

    def __len__(self):
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

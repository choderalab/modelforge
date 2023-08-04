import os
from abc import ABC, abstractmethod

import qcportal as ptl
import torch
from loguru import logger
from typing import Tuple


class Dataset(torch.utils.data.Dataset, ABC):
    """
    Abstract Base Class representing a dataset.
    """

    def __init__(self, dataset_file: str = ""):
        self.dataset_file = dataset_file
        self.dataset = self.load(dataset_file)
        (
            self.molecules,
            self.records,
        ) = self.dataset.get_molecules(), self.dataset.get_records(method="b3lyp")

    @abstractmethod
    def 

    @property
    @abstractmethod
    def qcportal_data(self):
        """
        Defines the qcportal data to be downloaded.
        """
        pass

    def load(self, dataset_file):
        """
        Loads the raw dataset from qcarchive.

        If a valid qcarchive generated hdf5 file is not passed to the
        to the init function, the code will download the
        raw dataset from qcarchive.
        """
        qcp_client = ptl.FractalClient()
        qcportal_data = self.qcportal_data

        try:
            dataset = qcp_client.get_collection(
                qcportal_data["collection"], qcportal_data["dataset"]
            )
        except Exception:
            print(
                f"Dataset {qcportal_data['dataset']} is not available in collection {qcportal_data['collection']}."
            )

        if dataset_file and os.path.isfile(dataset_file):
            if not dataset_file.endswith(".hdf5"):
                raise ValueError("Input file must be an .hdf5 file.")
            logger.debug(f"Loading from {dataset_file}")
            dataset.set_view(dataset_file)  
        else:
            logger.debug(f"Downloading from qcportal")
            dataset.download(dataset_file)

            # If dataset_file was specified, but does not exist, the file will be downloaded from qcarchive and saved to the file/path specified by dataset_file
            if dataset_file is None:
                dataset.download()
            else:
                if not dataset_file.endswith(".hdf5"):
                      raise ValueError("File must be an .hdf5 file.")
                dataset.download(dataset_file)
                dataset.to_file(path=dataset_file, encoding="hdf5")
        return dataset

    def __len__(self):
        return self.molecules.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]: # Tuple[0] with dim=3, Tuple[1] with dim=1
        """ pytorch dataset getitem method to return a tuple of geometry and energy

        Args:
            idx (int): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: returns a tuple of geometry and energy
        """
        geometry = torch.tensor(self.records.iloc[idx].record["geometry"])
        energy = torch.tensor(self.records.iloc[idx].record["energy"])

        if self.transform:
            geometry = self.transform(geometry)
        if self.target_transform:
            energy = self.target_transform(energy)

        return geometry, energy


class QM9Dataset(Dataset):
    """
    QM9 dataset as curated by qcarchive.
    """

    @property
    def qcportal_data(self):
        return {"collection": "Dataset", "dataset": "QM9"}

    @property
    def name(self):
        return "QM9"

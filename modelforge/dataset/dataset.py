import os
from abc import ABC, abstractmethod

import qcportal as ptl
import torch
from loguru import logger


class Dataset(torch.utils.data.Dataset, ABC):
    """
    Abstract Base Class representing a dataset.
    """

    def __init__(self, dataset_file: str = None, name: str = ""):
        self.dataset_file = dataset_file
        self.name = name
        self.dataset = self.load(dataset_file)
        (
            self.molecules,
            self.records,
        ) = self.dataset.get_molecules(), self.dataset.get_records(method="b3lyp")

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
            dataset.set_view(dataset_file)  # NOTE: this doesn't work yet
        else:
            logger.debug(f"Downloading from qcportal")

            # to get QM9 from qcportal, we need to define which collection and QM9
            if dataset_file is None:
                dataset.download()
            else:
                dataset.download(dataset_file)
                dataset.to_file(path=dataset_file, encoding="hdf5")
        return dataset

    def __len__(self):
        return self.molecules.shape[0]

    def __getitem__(self, idx):
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

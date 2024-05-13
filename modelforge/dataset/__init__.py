# defines the interaction with public datasets
from .qm9 import QM9Dataset
from .ani1x import ANI1xDataset
from .ani2x import ANI2xDataset
from .spice114 import SPICE114Dataset
from .spice2 import SPICE2Dataset
from .spice114openff import SPICE114OpenFFDataset
from .dataset import DatasetFactory, DataModule
from enum import Enum


class _ImplementedDatasets(Enum):
    QM9 = QM9Dataset
    ANI1X = ANI1xDataset
    ANI2X = ANI2xDataset
    SPICE114 = SPICE114Dataset
    SPICE2 = SPICE2Dataset
    SPICE114_OPENFF = SPICE114OpenFFDataset

    @classmethod
    def get_dataset_class(cls, dataset_name: str):
        try:
            # Normalize the input and get the class directly from the Enum
            return cls[dataset_name.upper()].value
        except KeyError:
            available_datasets = ", ".join([d.name for d in cls])
            raise ValueError(
                f"Dataset {dataset_name} is not implemented. Available datasets are: {available_datasets}"
            )

    @staticmethod
    def get_all_dataset_names():
        return [dataset.name for dataset in _ImplementedDatasets]

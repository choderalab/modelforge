"""Module that contains classes and function for loading and processing of datasets."""

from .dataset import HDF5Dataset

from .dataset import DataModule  # DatasetFactory,
from enum import Enum


# class _ImplementedDatasets(Enum):
#     QM9 = QM9Dataset
#     ANI1X = ANI1xDataset
#     ANI2X = ANI2xDataset
#     SPICE1 = SPICE1Dataset
#     SPICE2 = SPICE2Dataset
#     SPICE1_OPENFF = SPICE1OpenFFDataset
#     SPICE2_OPENFF = SPICE2OpenFFDataset
#     PHALKETHOH = PhAlkEthOHDataset
#     TMQM = tmQMDataset
#     TMQM_XTB = tmQMXTBDataset
#     FE_II = FeIIDataset
#
#     @classmethod
#     def get_dataset_class(cls, dataset_name: str):
#         try:
#             # Normalize the input and get the class directly from the Enum
#             return cls[dataset_name.upper()].value
#         except KeyError:
#             available_datasets = ", ".join([d.name for d in cls])
#             raise ValueError(
#                 f"Dataset {dataset_name} is not implemented. Available datasets are: {available_datasets}"
#             )
#
#     @staticmethod
#     def get_all_dataset_names():
#         return [dataset.name for dataset in _ImplementedDatasets]


class _ImplementedDatasets(Enum):
    QM9 = "QM9"
    ANI1X = "ANI1X"
    ANI2X = "ANI2X"
    SPICE1 = "SPICE1"
    SPICE2 = "SPICE2"
    SPICE1_OPENFF = "SPICE1_OPENFF"
    SPICE2_OPENFF = "SPICE2_OPENFF"
    PHALKETHOH = "PHALKETHOH"
    TMQM = "TMQM"
    TMQM_XTB = "TMQM_XTB"
    FE_II = "FE_II"

    @staticmethod
    def get_all_dataset_names():
        return [dataset.name for dataset in _ImplementedDatasets]

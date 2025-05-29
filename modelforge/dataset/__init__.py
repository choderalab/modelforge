"""Module that contains classes and function for loading and processing of datasets."""

from .dataset import HDF5Dataset

from .dataset import DataModule  # DatasetFactory,
from enum import Enum


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

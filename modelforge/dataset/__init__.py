# defines the interaction with public datasets
from .qm9 import QM9Dataset
from .dataset import DatasetFactory, TorchDataModule

_IMPLEMENTED_DATASETS = [
    "QM9",
    "ANI1X",
    # "ANI2X",
    # "SPICE114",
    # "SPICE2",
    # "SPICE114_OPENFF",
]

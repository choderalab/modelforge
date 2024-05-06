# defines the interaction with public datasets
from .qm9 import QM9Dataset
from .ani1x import ANI1xDataset
from .ani2x import ANI2xDataset
from .spice114 import SPICE114Dataset
from .spice2 import SPICE2Dataset
from .spice114openff import SPICE114OpenFFDataset
from .dataset import DatasetFactory, TorchDataModule

_IMPLEMENTED_DATASETS = {
    "QM9": QM9Dataset,
    "ANI1X": ANI1xDataset,
    "ANI2X": ANI2xDataset,
    "SPICE114": SPICE114Dataset,
    "SPICE2": SPICE2Dataset,
    "SPICE114_OPENFF": SPICE114OpenFFDataset,
}

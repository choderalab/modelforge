#

from typing import Literal

ModelType = Literal[
    "ANI2x", "PhysNet", "SchNet", "PaiNN", "SAKE", "TensorNet", "AimNet2"
]
DatasetType = Literal[
    "QM9",
    "ANI1X",
    "ANI2X",
    "SPICE1",
    "SPICE2",
    "SPICE1_OPENFF",
    "PhAlkEthOH",
]

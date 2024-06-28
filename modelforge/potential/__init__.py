from .schnet import SchNet
from .physnet import PhysNet
from .painn import PaiNN
from .ani import ANI2x
from .sake import SAKE
from .utils import (
    CosineCutoff,
    RadialBasisFunction,
    AngularSymmetryFunction,
)
from .processing import FromAtomToMoleculeReduction
from .models import NeuralNetworkPotentialFactory
from enum import Enum


class _Implemented_NNPs(Enum):
    ANI2X = ANI2x
    SCHNET = SchNet
    PAINN = PaiNN
    PHYSNET = PhysNet
    SAKE = SAKE

    @classmethod
    def get_neural_network_class(cls, neural_network_name: str):
        try:
            # Normalize the input and get the class directly from the Enum
            return cls[neural_network_name.upper()].value
        except KeyError:
            available_datasets = ", ".join([d.name for d in cls])
            raise ValueError(
                f"Dataset {neural_network_name} is not implemented. Available datasets are: {available_datasets}"
            )

    @staticmethod
    def get_all_neural_network_names():
        return [neural_network.name for neural_network in _Implemented_NNPs]

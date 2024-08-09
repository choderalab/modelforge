"""
This module contains the implemented neural network potentials and their parameters.
"""

from .schnet import SchNet
from .physnet import PhysNet
from .painn import PaiNN
from .ani import ANI2x
from .sake import SAKE
from .tensornet import TensorNet
from .parameters import (
    ANI2xParameters,
    SchNetParameters,
    PhysNetParameters,
    PaiNNParameters,
    SAKEParameters,
    TensorNetParameters,
)
from .utils import (
    CosineAttenuationFunction,
    RadialBasisFunction,
    AngularSymmetryFunction,
    FeaturizeInput,
)
from .processing import FromAtomToMoleculeReduction
from .models import NeuralNetworkPotentialFactory
from enum import Enum


class _Implemented_NNP_Parameters(Enum):
    ANI2X_PARAMETERS = ANI2xParameters
    SCHNET_PARAMETERS = SchNetParameters
    TENSORNET_PARAMETERS = TensorNetParameters
    PAINN_PARAMETERS = PaiNNParameters
    PHYSNET_PARAMETERS = PhysNetParameters
    SAKE_PARAMETERS = SAKEParameters

    @classmethod
    def get_neural_network_parameter_class(cls, neural_network_name: str):
        try:
            # Normalize the input and get the class directly from the Enum
            name = neural_network_name.upper() + "_PARAMETERS"
            return cls[name.upper()].value
        except KeyError:
            available_potentials = ", ".join([d.name for d in cls])
            raise ValueError(
                f"Parameters for {neural_network_name} are not implemented. Available parameters: {available_potentials}"
            )


class _Implemented_NNPs(Enum):
    ANI2X = ANI2x
    SCHNET = SchNet
    PAINN = PaiNN
    PHYSNET = PhysNet
    SAKE = SAKE
    TENSORNET = TensorNet

    @classmethod
    def get_neural_network_class(cls, neural_network_name: str):
        try:
            # Normalize the input and get the class directly from the Enum
            return cls[neural_network_name.upper()].value
        except KeyError:
            available_potentials = ", ".join([d.name for d in cls])
            raise ValueError(
                f"Potential {neural_network_name} is not implemented. Available potentials are: {available_potentials}"
            )

    @staticmethod
    def get_all_neural_network_names():
        return [neural_network.name for neural_network in _Implemented_NNPs]

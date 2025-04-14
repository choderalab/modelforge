"""
This module contains the implemented neural network potentials and their parameters.
"""

from enum import Enum

from .potential import NeuralNetworkPotentialFactory
from .painn import PaiNNCore
from .parameters import (
    AimNet2Parameters,
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
    SpookyNetParameters,
)
from .processing import FromAtomToMoleculeReduction
from .representation import (
    CosineAttenuationFunction,
    PhysNetRadialBasisFunction,
    TensorNetRadialBasisFunction,
    AniRadialBasisFunction,
    SchnetRadialBasisFunction,
)
from .featurization import FeaturizeInput

from .physnet import PhysNetCore
from .sake import SAKECore
from .schnet import SchNetCore
from .tensornet import TensorNetCore
from .aimnet2 import AimNet2Core
from .ani import ANI2xCore


class _Implemented_NNP_Parameters(Enum):
    ANI2X_PARAMETERS = ANI2xParameters
    SCHNET_PARAMETERS = SchNetParameters
    TENSORNET_PARAMETERS = TensorNetParameters
    PAINN_PARAMETERS = PaiNNParameters
    PHYSNET_PARAMETERS = PhysNetParameters
    SAKE_PARAMETERS = SAKEParameters
    AIMNET2_PARAMETERS = AimNet2Parameters

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
    SCHNET = SchNetCore
    ANI2X = ANI2xCore
    PHYSNET = PhysNetCore
    TENSORNET = TensorNetCore
    PAINN = PaiNNCore
    SAKE = SAKECore
    AIMNET2 = AimNet2Core
    SPOOKYNET = SpookyNet

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

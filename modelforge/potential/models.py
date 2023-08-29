import torch.nn as nn
import torch
from typing import Dict, List, Optional

from modelforge.utils import Properties, Inputs, SpeciesEnergies

import torch.nn as nn
import torch
from typing import Dict, List, Optional


class BaseNNP(nn.Module):
    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        input_dtype: str = "float32",
    ):
        """
        Initialize the NeuralNetworkPotential class.

        Parameters
        ----------
        representation : nn.Module
            The module that builds representation from inputs.
        input_modules : List[nn.Module], optional
            Modules applied before representation, default is None.
        output_modules : List[nn.Module], optional
            Modules that predict output properties from the representation, default is None.
        input_dtype: str, optional
            The dtype of real inputs, default is "float32".
        """
        super().__init__(
            input_dtype=input_dtype,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules or [])
        self.output_modules = nn.ModuleList(output_modules or [])

    def forward(
        self,
        inputs: Inputs,
    ) -> SpeciesEnergies:
        E = self.calculate_energies_and_forces(inputs)
        return SpeciesEnergies(inputs.Z, E)

    def calculate_energies_and_forces():
        raise NotImplementedError

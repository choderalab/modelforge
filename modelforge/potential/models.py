import torch.nn as nn
import torch
from typing import Dict, List, Optional

from modelforge.utils import Properties, Inputs, SpeciesEnergies

import torch.nn as nn
import torch
from typing import Dict, List, Optional
import torch.nn as nn
from torch import dtype
from torch import DeviceObjType


class BaseNNP(nn.Module):
    def __init__(self, dtype: torch.dtype, device: torch.device):
        """
        Initialize the NeuralNetworkPotential class.

        input_dtype: str, optional
            The dtype of real inputs, default is "float32".
        """
        super().__init__()
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        inputs: Inputs,
    ) -> SpeciesEnergies:
        E = self.calculate_energies_and_forces(inputs)
        return SpeciesEnergies(inputs.Z, E)

    def calculate_energies_and_forces():
        raise NotImplementedError


class SchNetRepresentation(nn.Module):
    def __init__(self, n_atom_basis, n_filters, n_gaussians, n_interactions):
        super().__init__()
        # Define the SchNet layers and operations here
        # e.g., continuous-filter convolutional layers, interaction blocks, etc.

    def forward(self, inputs):
        # Implement the forward pass for the SchNet representation
        # Apply the defined layers and operations to the inputs
        return outputs


class SchNetInputModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define any input preprocessing layers or operations here

    def forward(self, inputs):
        # Implement input preprocessing here
        return processed_inputs


class SchNetOutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define any output postprocessing layers or operations here

    def forward(self, inputs):
        # Implement output postprocessing here
        return processed_outputs


class SchNetPotential(BaseNNP):
    def __init__(self, n_atom_basis, n_filters, n_gaussians, n_interactions):
        representation = SchNetRepresentation(
            n_atom_basis, n_filters, n_gaussians, n_interactions
        )
        input_modules = [SchNetInputModule()]  # Optional
        output_modules = [SchNetOutputModule()]  # Optional
        super().__init__(representation, input_modules, output_modules)

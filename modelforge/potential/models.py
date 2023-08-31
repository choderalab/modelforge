from typing import Dict, List, Optional

import torch.nn as nn
import lightning as pl
from modelforge.utils import Inputs, PropertyNames, SpeciesEnergies


class BaseNNP(pl.LightningModule):
    """
    Abstract base class for neural network potentials.
    This class defines the overall structure and ensures that subclasses
    implement the `calculate_energies_and_forces` method.
    """

    def __init__(self):
        """
        Initialize the NeuralNetworkPotential class.

        Parameters
        ----------
        dtype : torch.dtype
            Data type for the PyTorch tensors.
        device : torch.device
            Device ("cpu" or "cuda") on which computations will be performed.

        """
        super().__init__()

    def forward(
        self,
        inputs: Inputs,
    ) -> SpeciesEnergies:
        """
        Forward pass for the neural network potential.

        Parameters
        ----------
        inputs : Inputs
            An instance of the Inputs data class containing atomic numbers, positions, etc.

        Returns
        -------
        SpeciesEnergies
            An instance of the SpeciesEnergies data class containing species and calculated energies.

        """

        E = self.calculate_energies_and_forces(inputs)
        return SpeciesEnergies(inputs.Z, E)

    def calculate_energies_and_forces(self, inputs: Optional[Inputs] = None):
        """
        Placeholder for the method that should calculate energies and forces.
        This method should be implemented in subclasses.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in the subclass.

        """
        raise NotImplementedError("Subclasses must implement this method.")

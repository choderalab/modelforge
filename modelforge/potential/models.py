from typing import Dict, List, Optional

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from modelforge.utils import Inputs, SpeciesEnergies


class BaseNNP(pl.LightningModule):
    """
    Abstract base class for neural network potentials.
    This class defines the overall structure and ensures that subclasses
    implement the `calculate_energies_and_forces` method.
    """

    def __init__(self):
        """
        Initialize the NNP class.
        """
        super().__init__()

    def forward(
        self,
        inputs: dict,
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
        assert isinstance(inputs, dict)  #
        E = self.calculate_energy(inputs)
        return SpeciesEnergies(inputs["Z"], E)

    def calculate_energy(self, inputs: dict) -> torch.Tensor:
        """
        Placeholder for the method that should calculate energies and forces.
        This method should be implemented in subclasses.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in the subclass.

        """
        raise NotImplementedError("Subclasses must implement this method.")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        E_hat = self.forward(batch)  # wrap_vals_from_dataloader(batch))
        loss = nn.functional.mse_loss(E_hat.energies, batch["E"])
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer


def wrap_vals_from_dataloader(vals):
    Z, R, E = vals["Z"], vals["R"], vals["E"]
    return Inputs(Z, R, E)

from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from loguru import logger as log

from abc import ABC, abstractmethod
import torch


class Postprocess(nn.Module, ABC):
    def __init__(self, stats: dict):
        """
        Initializes the postprocessing operation with dataset statistics.

        Parameters
        ----------
        stats : dict
            A dictionary containing dataset statistics such as self energies, mean, and stddev.
        """
        super().__init__()
        self.stats = stats

    @abstractmethod
    def forward(
        self, postprocessing_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Abstract method to be implemented by subclasses for applying specific postprocessing operations.

        Parameters
        ----------
        postprocessing_data : Dict[str, torch.Tensor]
            energy_readout: torch.Tensor, shape (nr_of_molecuels_in_batch)
            atomic_numbers: torch.Tensor, shape (nr_of_atoms_in_batch)
            atomic_subsystem_indices: torch.Tensor, shape (nr_of_atoms_in_batch)

        Returns
        -------
        Dict[str, torch.Tensor]
            The postprocessed data with updated energy_readout
        """
        pass


class NoPostprocess(Postprocess):

    def __init__(self, stats: Dict = {}):
        super().__init__(stats)

    def forward(
        self,
        postprocessing_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Simply returns the input energies without any modification.
        """
        return postprocessing_data


class UndoNormalization(Postprocess):
    def forward(
        self,
        energies: torch.Tensor,
        atomic_numbers: torch.Tensor,
        molecule_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undoes the normalization of the energies using the mean and stddev from the statistics.
        """
        mean = self.stats.get("mean", 0.0)
        stddev = self.stats.get("stddev", 1.0)
        return (energies * stddev) + mean


class AddSelfEnergies(Postprocess):
    def forward(
        self,
        energies: torch.Tensor,
        atomic_numbers: torch.Tensor,
        molecule_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adds self energies to each molecule based on its constituent atomic numbers in a vectorized manner.
        """
        # Convert self_energies to a tensor for efficient lookup
        max_atomic_number = max(self.stats["self_energies"].keys())
        self_energies_tensor = torch.zeros(max_atomic_number + 1)
        for atomic_number, energy in self.stats["self_energies"].items():
            self_energies_tensor[atomic_number] = energy

        # Calculate self energies for each atom and then aggregate them per molecule
        atom_self_energies = self_energies_tensor[atomic_numbers]
        molecule_self_energies = torch.zeros_like(energies)

        for i, energy in enumerate(energies):
            # Find atoms belonging to the current molecule
            mask = molecule_indices == i
            # Sum self energies for these atoms
            molecule_self_energies[i] = atom_self_energies[mask].sum()

        # Adjust energies by adding aggregated self energies for each molecule
        adjusted_energies = energies + molecule_self_energies

        return adjusted_energies


import torch.nn as nn


class PostprocessingPipeline(nn.Module):
    def __init__(self, modules):
        """
        Initializes the postprocessing pipeline with a list of postprocessing modules.

        Parameters
        ----------
        modules : list
            A list of postprocessing modules (instances of classes inheriting from Postprocess).
        """
        super().__init__()
        self.postprocess = nn.ModuleList(modules)

    def forward(self, postprocessing_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses for applying specific postprocessing operations.

        Parameters
        ----------
        postprocessing_data : Dict[str, torch.Tensor]
            energy_readout: torch.Tensor, shape (nr_of_molecuels_in_batch)
            atomic_numbers: torch.Tensor, shape (nr_of_atoms_in_batch)
            atomic_subsystem_indices: torch.Tensor, shape (nr_of_atoms_in_batch)


        Returns
        -------
        torch.Tensor
            The postprocessed energies.
        """
        for module in self.postprocess:
            postprocessing_data = module.forward(postprocessing_data)
        return postprocessing_data

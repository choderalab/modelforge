from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW

from loguru import logger as log

from abc import ABC, abstractmethod
import torch


class Postprocess(nn.Module, ABC):
    def __init__(self, dataset_statistics: dict):
        """
        Initializes the postprocessing operation with dataset statistics.

        Parameters
        ----------
        stats : dict
            A dictionary containing dataset statistics such as self energies, mean, and stddev.
        """
        super().__init__()
        self.dataset_statistics = dataset_statistics

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

    def __init__(self, dataset_statistics: dict = {}):
        super().__init__(dataset_statistics)

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
        postprocessing_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Undoes the normalization of the energies using the mean and stddev from the statistics.
        """
        mean = self.dataset_statistics.get("mean", 0.0)
        stddev = self.dataset_statistics.get("stddev", 1.0)
        postprocessing_data["energy_readout"] = (
            postprocessing_data["energy_readout"] * stddev
        ) + mean
        return postprocessing_data["energy_readout"]


class AddSelfEnergies(Postprocess):

    def __init__(self, dataset_statistics: Dict):
        super().__init__(dataset_statistics)

        # calculate maximum atomic number from provided ase
        atomic_numbers = []
        for atomic_number, ase in dataset_statistics["self_energies"]:
            atomic_numbers.append(atomic_number)

        self.max_atomic_number: int = max(atomic_numbers) + 1

        # fill ase in self_energies tensor
        self.self_energies_tensor = torch.zeros(self.max_atomic_number)
        for atomic_number, ase in self.dataset_statistics["self_energies"]:
            self.self_energies_tensor[atomic_number] = ase

    def forward(
        self,
        postprocessing_data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Adds self energies to each molecule based on its constituent atomic numbers in a vectorized manner.
        """
        # Convert self_energies to a tensor for efficient lookup
        energies = postprocessing_data["energy_readout"]
        atomic_numbers = postprocessing_data["atomic_numbers"]
        molecule_indices = postprocessing_data["atomic_subsystem_indices"]

        # Calculate self energies for each atom and then aggregate them per molecule
        atom_self_energies = self.self_energies_tensor[atomic_numbers]
        molecule_self_energies = torch.zeros_like(energies)

        for i in range(len(energies)):
            # Find atoms belonging to the current molecule
            mask = molecule_indices == i
            # Sum self energies for these atoms
            molecule_self_energies[i] = atom_self_energies[mask].sum()

        # Adjust energies by adding aggregated self energies for each molecule
        adjusted_energies = energies + molecule_self_energies

        return adjusted_energies


import torch.nn as nn

from typing import List


class PostprocessingPipeline(nn.Module):
    def __init__(self, modules: List[nn.Module]):
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

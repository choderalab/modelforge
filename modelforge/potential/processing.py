"""
This module contains utility functions and classes for processing the output of the potential model.
"""

import torch
from typing import Dict
from openff.units import unit


def load_atomic_self_energies(path: str) -> Dict[str, unit.Quantity]:
    """
    Load the atomic self energies from the dataset statistics toml.

    Parameters
    ----------
    path : str
        path to the dataset statistics toml file.

    Returns
    -------
    Dict[str, unit.Quantity]
        returns the atomic self energies from the dataset statistics toml with units attached.
    """

    import toml

    energy_statistic = toml.load(open(path, "r"))

    # attach units
    atomic_self_energies = {
        key: unit.Quantity(value)
        for key, value in energy_statistic["atomic_self_energies"].items()
    }

    return atomic_self_energies


def load_dataset_energy_statistics(path: str) -> Dict[str, unit.Quantity]:
    """
    Load the per-atom energy distribution statistics (mean and stddev of atomic energies) from the dataset statistics toml.

    Parameters
    ----------
    path : str
        path to the dataset statistics toml file.

    Returns
    -------
    Dict[str, unit.Quantity]
        returns the per-atom energy distribution statistics from the dataset statistics toml with units attached.
    """
    import toml

    energy_statistic = toml.load(open(path, "r"))
    # attach units
    training_dataset_statistics = {
        key: unit.Quantity(value)
        for key, value in energy_statistic["training_dataset_statistics"].items()
    }

    return training_dataset_statistics


class FromAtomToMoleculeReduction(torch.nn.Module):
    """
    Reducing per-atom property to per-molecule property.
    """

    def __init__(
        self,
        reduction_mode: str = "sum",
    ):
        """
        Initializes the per-atom property readout_operation module.

        Parameters
        ----------
        per_atom_property_name : str
            The name of the per-atom property that will be reduced.
        index_name : str
            The name of the index used to identify the molecules.
        output_name : str
            The name of the output property.
        reduction_mode : str, optional
            The reduction mode. Default is "sum".
        keep_per_atom_property : bool, optional
            Whether to keep the per-atom property. Default is False.
        """
        super().__init__()
        self.reduction_mode = reduction_mode

    def forward(
        self, indices: torch.Tensor, per_atom_property: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data dictionary containing the per-atom property and index.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output data dictionary containing the per-molecule property.
        """

        # Perform scatter add operation for atoms belonging to the same molecule
        nr_of_molecules = torch.unique(indices)
        nr_of_molecules = nr_of_molecules.size(0)
        property_per_molecule_zeros = torch.zeros(
            nr_of_molecules,
            dtype=per_atom_property.dtype,
            device=per_atom_property.device,
        )

        return property_per_molecule_zeros.scatter_reduce(
            0, indices.long(), per_atom_property, reduce=self.reduction_mode
        )


from dataclasses import dataclass, field
from typing import Dict, Iterator

from openff.units import unit
from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT


@dataclass
class AtomicSelfEnergies:
    """
    AtomicSelfEnergies stores a mapping of atomic elements to their self energies.

    Provides lookup by atomic number or symbol, iteration over the mapping,
    and utilities to convert between atomic number and symbol.

    Intended as a base class to be extended with specific element-energy values.
    """

    # We provide a dictionary with {str:float} of element name to atomic self-energy,
    # which can then be accessed by atomic index or element name
    energies: Dict[str, unit.Quantity] = field(default_factory=dict)
    # Example mapping, replace or extend as necessary
    atomic_number_to_element: Dict[int, str] = field(
        default_factory=lambda: _ATOMIC_NUMBER_TO_ELEMENT
    )
    _ase_tensor_for_indexing = None

    def __getitem__(self, key):
        from modelforge.utils.units import chem_context

        if isinstance(key, int):
            # Convert atomic number to element symbol
            element = self.atomic_number_to_element.get(key)
            if element is None:
                raise KeyError(f"Atomic number {key} not found.")
            if self.energies.get(element) is None:
                return None
            return self.energies.get(element).to(unit.kilojoule_per_mole, "chem").m
        elif isinstance(key, str):
            # Directly access by element symbol
            if key not in self.energies:
                raise KeyError(f"Element {key} not found.")
            if self.energies[key] is None:
                return None

            return self.energies[key].to(unit.kilojoule_per_mole, "chem").m
        else:
            raise TypeError(
                "Key must be an integer (atomic number) or string (element name)."
            )

    def __iter__(self) -> Iterator[Dict[str, float]]:
        """Iterate over the energies dictionary."""
        from modelforge.utils.units import chem_context

        for element, energy in self.energies.items():
            atomic_number = self.element_to_atomic_number(element)
            yield (atomic_number, energy.to(unit.kilojoule_per_mole, "chem").m)

    def __len__(self) -> int:
        """Return the number of element-energy pairs."""
        return len(self.energies)

    def element_to_atomic_number(self, element: str) -> int:
        """Return the atomic number for a given element symbol."""
        for atomic_number, elem_symbol in self.atomic_number_to_element.items():
            if elem_symbol == element:
                return atomic_number
        raise ValueError(f"Element symbol '{element}' not found in the mapping.")

    @property
    def atomic_number_to_energy(self) -> Dict[int, float]:
        """Return a dictionary mapping atomic numbers to their energies."""
        return {
            atomic_number: self[atomic_number]
            for atomic_number in self.atomic_number_to_element.keys()
            if self[atomic_number] is not None
        }

    @property
    def ase_tensor_for_indexing(self) -> torch.Tensor:
        if self._ase_tensor_for_indexing is None:
            max_z = max(self.atomic_number_to_element.keys()) + 1
            ase_tensor_for_indexing = torch.zeros(max_z)
            for idx in self.atomic_number_to_element:
                if self[idx]:
                    ase_tensor_for_indexing[idx] = self[idx]
                else:
                    ase_tensor_for_indexing[idx] = 0.0
            self._ase_tensor_for_indexing = ase_tensor_for_indexing

        return self._ase_tensor_for_indexing


class ScaleValues(torch.nn.Module):
    """
    Rescales values using the provided mean and standard deviation.
    """

    def __init__(
        self,
        mean: float,
        stddev: float,
    ) -> None:
        """
        Rescales values using the provided mean and standard deviation.
        Parameters
        ----------
        mean : float
            The mean value used for rescaling.
        stddev : float
            The standard deviation value used for rescaling.
        property : str
            The name of the property to be rescaled.
        output_name : str
            The name of the output property.

        """

        super().__init__()
        self.register_buffer("mean", torch.tensor([mean]))
        self.register_buffer("stddev", torch.tensor([stddev]))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Rescales values using the provided mean and standard deviation.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data dictionary containing the property to be rescaled.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output data dictionary containing the rescaled values.
        """
        return data * self.stddev + self.mean


from typing import Union


class PerAtomEnergy(torch.nn.Module):

    def __init__(
        self, per_atom_energy: Dict[str, bool], dataset_statistics: Dict[str, float]
    ):
        super().__init__()

        if per_atom_energy.get("normalize"):
            scale = ScaleValues(
                dataset_statistics["per_atom_energy_mean"],
                dataset_statistics["per_atom_energy_stddev"],
            )
        else:
            scale = ScaleValues(0.0, 1.0)

        self.scale = scale

        if per_atom_energy.get("from_atom_to_molecule_reduction"):
            reduction = FromAtomToMoleculeReduction()

        self.reduction = reduction

    def forward(self, per_atom_property: torch.Tensor, indices: torch.Tensor):
        scaled_values = self.scale(per_atom_property)
        reduced_values = self.reduction(indices, scaled_values)
        return reduced_values


class CalculateAtomicSelfEnergy(torch.nn.Module):
    """
    Calculates the atomic self energy for each molecule.
    """

    def __init__(
        self, atomic_self_energies: Dict[str, Union[unit.Quantity, str]]
    ) -> None:
        """
        Calculates the atomic self energy for each molecule given the atomic self energies provided by the training dataset and the element information.


        Parameters
        ----------
        atomic_self_energies : Dict[str, Union[unit.Quantity, str]]
            A dictionary containing the atomic self energies for each atomic number.

        """

        super().__init__()

        # if values in atomic_self_energies are strings convert them to kJ/mol
        if isinstance(list(atomic_self_energies.values())[0], str):
            atomic_self_energies = {
                key: unit.Quantity(value) for key, value in atomic_self_energies.items()
            }
        self.atomic_self_energies = AtomicSelfEnergies(atomic_self_energies)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates the molecular self energy.

        Parameters
        ----------
        data : dict
            The input data for the model, including atomic numbers and subsystem indices.

        Returns
        -------
        torch.Tensor
            The tensor containing the molecular self energy for each molecule.
        """
        atomic_numbers = data["atomic_numbers"]
        atomic_subsystem_indices = data["atomic_subsystem_indices"]

        atomic_subsystem_indices = atomic_subsystem_indices.to(
            dtype=torch.long, device=atomic_numbers.device
        )

        # atomic_number_to_energy
        ase_tensor_for_indexing = self.atomic_self_energies.ase_tensor_for_indexing.to(
            device=atomic_numbers.device
        )

        # use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        data["ase_tensor"] = ase_tensor
        return data

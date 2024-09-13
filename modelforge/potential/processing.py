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
        per_atom_property_name: str,
        index_name: str,
        output_name: str,
        reduction_mode: str = "sum",
        keep_per_atom_property: bool = False,
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
        self.per_atom_property_name = per_atom_property_name
        self.output_name = output_name
        self.index_name = index_name
        self.keep_per_atom_property = keep_per_atom_property

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        indices = data[self.index_name].to(torch.int64)
        per_atom_property = data[self.per_atom_property_name]
        # Perform scatter add operation for atoms belonging to the same molecule
        property_per_molecule_zeros = torch.zeros(
            len(indices.unique()),
            dtype=per_atom_property.dtype,
            device=per_atom_property.device,
        )

        property_per_molecule = property_per_molecule_zeros.scatter_reduce(
            0, indices, per_atom_property, reduce=self.reduction_mode
        )

        data[self.output_name] = property_per_molecule
        if self.keep_per_atom_property is False:
            del data[self.per_atom_property_name]

        return data


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
        self, mean: float, stddev: float, property: str, output_name: str
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
        self.property = property
        self.output_name = output_name

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        data[self.output_name] = data[self.property] * self.stddev + self.mean
        return data


from typing import Union


class ChargeConservation(torch.nn.Module):
    def __init__(self, method="physnet"):

        super().__init__()
        self.method = method
        if self.method == "physnet":
            self.correct_partial_charges = self.physnet_charge_conservation
        else:
            raise ValueError(f"Unknown charge conservation method: {self.method}")

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ):
        """
        Apply charge conservation to partial charges.

        Parameters
        ----------
        per_atom_partial_charge : torch.Tensor
            Flat tensor of partial charges for all atoms in the batch.
        atomic_subsystem_indices : torch.Tensor
            Tensor of integers indicating which molecule each atom belongs to.
        total_charges : torch.Tensor
            Tensor of desired total charges for each molecule.

        Returns
        -------
        torch.Tensor
            Tensor of corrected partial charges.
        """
        data["per_atom_charge_corrected"] = self.correct_partial_charges(
            data["per_atom_charge"],
            data["atomic_subsystem_indices"],
            data["per_molecule_charge"],
        )
        return data

    def physnet_charge_conservation(
        self,
        per_atom_charge: torch.Tensor,
        mol_indices: torch.Tensor,
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        PhysNet charge conservation method based on equation 14 from the PhysNet
        paper.

        Correct the partial charges such that their sum matches the desired
        total charge for each molecule.

        Parameters
        ----------
        partial_charges : torch.Tensor
            Flat tensor of partial charges for all atoms in all molecules.
        mol_indices : torch.Tensor
            Tensor of integers indicating which molecule each atom belongs to.
        total_charges : torch.Tensor
            Tensor of desired total charges for each molecule.

        Returns
        -------
        torch.Tensor
            Tensor of corrected partial charges.
        """
        # the general approach here is outline in equation 14 in the PhysNet
        # paper: the difference between the sum of the predicted partial charges
        # and the total charge is calculated and then distributed evenly among
        # the predicted partial charges

        # Calculate the sum of partial charges for each molecule

        # for each atom i, calculate the sum of partial charges for all other
        predicted_per_molecule_charge = torch.zeros(
            total_charges.shape,
            dtype=per_atom_charge.dtype,
            device=total_charges.device,
        ).scatter_add_(0, mol_indices.long(), per_atom_charge)

        # Calculate the correction factor for each molecule
        correction_factors = (
            total_charges - predicted_per_molecule_charge
        ) / mol_indices.bincount()

        # Apply the correction to each atom's charge
        per_atom_charge_corrected = per_atom_charge + correction_factors[mol_indices]

        per_molecule_corrected_charge = torch.zeros_like(per_atom_charge).scatter_add_(
            0, mol_indices.long(), per_atom_charge_corrected
        )

        return per_atom_charge_corrected


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

"""
This module contains utility functions and classes for processing the output of the potential model.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, Union

import torch
from openff.units import unit

from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT


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
        per_system_property = torch.zeros(
            nr_of_molecules,
            dtype=per_atom_property.dtype,
            device=per_atom_property.device,
        )

        return per_system_property.scatter_reduce(
            0, indices.long(), per_atom_property, reduce=self.reduction_mode
        )


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


def default_charge_conservation(
    per_atom_charge: torch.Tensor,
    per_system_total_charge: torch.Tensor,
    mol_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Adjusts partial atomic charges so that the sum of charges in each molecule
    matches the desired total charge.

    This method is based on equation 14 from the PhysNet paper.

    Parameters
    ----------
    partial_charges : torch.Tensor
        Tensor of partial charges for all atoms in all molecules.
    per_system_total_charge : torch.Tensor
        Tensor of desired total charges for each molecule.
    mol_indices : torch.Tensor
        Tensor of integers indicating which molecule each atom belongs to.

    Returns
    -------
    torch.Tensor
        Tensor of corrected partial charges.
    """
    # Calculate the sum of partial charges for each molecule
    predicted_per_system_total_charge = torch.zeros(
        per_system_total_charge.shape[0],
        dtype=per_atom_charge.dtype,
        device=per_atom_charge.device,
    ).scatter_add_(0, mol_indices.long(), per_atom_charge)

    # Calculate the number of atoms in each molecule
    num_atoms_per_system = mol_indices.bincount(
        minlength=per_system_total_charge.size(0)
    )

    # Calculate the correction factor for each molecule
    correction_factors = (
        per_system_total_charge.squeeze() - predicted_per_system_total_charge
    ) / num_atoms_per_system

    # Apply the correction to each atom's charge
    per_atom_charge_corrected = per_atom_charge + correction_factors[mol_indices]

    return per_atom_charge_corrected


class ChargeConservation(torch.nn.Module):
    def __init__(self, method="default"):
        """
        Module to enforce charge conservation on partial atomic charges.

        Parameters
        ----------
        method : str, optional, default='default'
            The method to use for charge conservation. Currently, only 'default'
            is supported.

        Methods
        -------
        forward(data)
            Applies charge conservation to the partial charges in the provided
            data dictionary.
        """

        super().__init__()
        self.method = method
        if self.method == "default":
            self.correct_partial_charges = default_charge_conservation
        else:
            raise ValueError(f"Unknown charge conservation method: {self.method}")

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ):
        """
        Apply charge conservation to partial charges in the data dictionary.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing the following keys:
            - "per_atom_charge":
                Tensor of partial charges for all atoms in the batch.
            -  "per_system_total_charge":
                Tensor of desired total charges for each
            molecule.
            - "atomic_subsystem_indices":
                Tensor indicating which molecule each atom belongs to.

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated data dictionary with the key "per_atom_charge_corrected"
            added, containing the corrected per-atom charges.
        """
        data["per_atom_charge_uncorrected"] = data["per_atom_charge"]
        data["per_atom_charge"] = self.correct_partial_charges(
            data["per_atom_charge"],
            data["per_system_total_charge"],
            data["atomic_subsystem_indices"],
        )
        return data


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

        if per_atom_energy.get("from_atom_to_system_reduction"):
            reduction = FromAtomToMoleculeReduction()

        self.reduction = reduction

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        per_atom_property, indices = (
            data["per_atom_energy"],
            data["atomic_subsystem_indices"],
        )
        scaled_values = self.scale(per_atom_property)
        per_system_energy = self.reduction(indices, scaled_values)

        data["per_system_energy"] = per_system_energy
        data["per_atom_energy"] = data["per_atom_energy"].detach()

        return data


class PerAtomCharge(torch.nn.Module):

    def __init__(self, per_atom_charge: Dict[str, bool]):
        super().__init__()
        from torch import nn

        if per_atom_charge["conserve"] == True:
            self.conserve = ChargeConservation(per_atom_charge["conserve_strategy"])
        else:
            self.conserve = nn.Identity()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.conserve(data)


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


class CoulombPotential(torch.nn.Module):
    def __init__(self, cutoff: float):
        """
        Computes the long-range electrostatic energy for a molecular system
        based on predicted partial charges and pairwise distances between atoms.

        The implementation follows the methodology described in the PhysNet
        paper, using a cutoff function to handle long-range interactions.

        Parameters
        ----------
        cutoff : float
            The cutoff distance beyond which the interactions are not
            considered in nanometer.

        Attributes
        ----------
        strategy : str
            The strategy for computing long-range interactions.
        cutoff_function : nn.Module
            The cutoff function applied to the pairwise distances.
        """
        super().__init__()
        from .representation import PhysNetAttenuationFunction

        self.cutoff_function = PhysNetAttenuationFunction(cutoff)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the long-range electrostatic energy.

        This function calculates the long-range electrostatic energy by considering
        pairwise Coulomb interactions between atoms, applying a cutoff function to
        handle long-range interactions.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data containing the following keys:
            - 'per_atom_charge': Tensor of shape (N,) with partial charges for each atom.
            - 'atomic_subsystem_indices': Tensor indicating the molecule each atom belongs to.
            - 'pairwise_properties': Object containing pairwise distances and indices.

        Returns
        -------
        Dict[str, torch.Tensor]
            The input data dictionary with an additional key 'long_range_electrostatic_energy'
            containing the computed long-range electrostatic energy.
        """
        mol_indices = data["atomic_subsystem_indices"]
        idx_i, idx_j = data["pair_indices"]

        # only unique paris
        unique_pairs_mask = idx_i < idx_j
        idx_i = idx_i[unique_pairs_mask]
        idx_j = idx_j[unique_pairs_mask]

        # mask pairwise properties
        pairwise_distances = data["d_ij"][unique_pairs_mask]
        per_atom_charge = data["per_atom_charge"]

        # Initialize the long-range electrostatic energy
        electrostatic_energy = torch.zeros_like(data["per_system_energy"])

        # Apply the cutoff function to pairwise distances
        phi_2r = self.cutoff_function(2 * pairwise_distances)
        chi_r = phi_2r * (1 / torch.sqrt(pairwise_distances**2 + 1)) + (
            1 - phi_2r
        ) * (1 / pairwise_distances)

        # Compute the Coulomb interaction term
        coulomb_interactions = (
            per_atom_charge[idx_i] * per_atom_charge[idx_j]
        ) * chi_r.squeeze(-1)

        # Sum over all interactions for each molecule
        data["electrostatic_energy"] = (
            electrostatic_energy.scatter_add_(
                0, mol_indices.long(), coulomb_interactions
            )
            * 138.96
        )  # in kj/mol nm

        return data

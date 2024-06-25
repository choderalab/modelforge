import torch
from modelforge.dataset.dataset import DatasetStatistics
from modelforge.potential.utils import NeuralNetworkData


class CalculateAtomicSelfEnergy(object):

    @staticmethod
    def calculate_atomic_self_energy(
        data: NeuralNetworkData,
        dataset_statistics: DatasetStatistics,
    ) -> torch.Tensor:
        """
        Calculates the molecular self energy.

        Parameters
        ----------
        data : NeuralNetworkData
            The input data for the model, including atomic numbers and subsystem indices.
        number_of_molecules : int
            The number of molecules in the batch.

        Returns
        -------
        torch.Tensor
            The tensor containing the molecular self energy for each molecule.
        """

        atomic_numbers = data.atomic_numbers
        atomic_subsystem_indices = data.atomic_subsystem_indices.to(
            dtype=torch.long, device=atomic_numbers.device
        )

        # atomic_number_to_energy
        atomic_self_energies = dataset_statistics.atomic_self_energies
        ase_tensor_for_indexing = atomic_self_energies.ase_tensor_for_indexing.to(
            device=atomic_numbers.device
        )

        # first, we need to use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        return ase_tensor


class FromAtomToMoleculeReduction(torch.nn.Module):

    def __init__(
        self,
    ):
        """
        Initializes the per-atom property readout module.
        Performs the reduction of 'per_atom' property to 'per_molecule' property.
        """
        super().__init__()

    def forward(
        self, per_atom_property: torch.Tensor, atomic_subsystem_indices: torch.Tensor
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        per_atom_property: torch.Tensor, shape [nr_of_atoms, 1]. The per-atom property that will be reduced to per-molecule property.
        atomic_subsystem_indices: torch.Tensor, shape [nr_of_atoms]. The atomic subsystem indices

        Returns
        -------
        Tensor, shape [nr_of_moleculs, 1], the per-molecule property.
        """

        # Perform scatter add operation for atoms belonging to the same molecule
        indices = atomic_subsystem_indices.to(torch.int64)
        property_per_molecule_zeros = torch.zeros(
            len(atomic_subsystem_indices.unique()),
            dtype=per_atom_property.dtype,
            device=per_atom_property.device,
        )

        property_per_molecule = property_per_molecule_zeros.scatter_add(
            0, indices, per_atom_property
        )

        # Sum across feature dimension to get final tensor of shape (num_molecules, 1)
        # property_per_molecule = result.sum(dim=1, keepdim=True)
        return property_per_molecule


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


from modelforge.potential.utils import NeuralNetworkData


class EnergyScaling(torch.nn.Module):

    def __init__(self, E_i_mean: float, E_i_stddev: float) -> None:
        """
        Initializes the `DatasetStatistics` object with default values for the scaling factors and atomic self energies.
        """
        super().__init__()
        self.register_buffer("E_i_mean", torch.tensor([E_i_mean]))
        self.register_buffer("E_i_stddev", torch.tensor([E_i_stddev]))

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Rescales energies using the dataset statistics.

        Parameters
        ----------
        energies : torch.Tensor
            The tensor of energies to be rescaled.

        Returns
        -------
        torch.Tensor
            The rescaled energies.
        """

        return energies * self.E_i_stddev + self.E_i_mean

"""
Module of dataclass definitions of properties.
"""

from dataclasses import dataclass
import torch
from typing import NamedTuple, Optional
from loguru import logger as log
from openff.units import unit


@dataclass
class PropertyNames:
    atomic_numbers: str
    positions: str
    E: str
    F: Optional[str] = None
    total_charge: Optional[str] = None
    dipole_moment: Optional[str] = None


PropertyUnits = {
    "atomic_numbers": "dimensionless",
    "positions": unit.nanometer,
    "E": unit.kilojoule_per_mole,
    "F": unit.kilojoule_per_mole / unit.nanometer,
    "total_charge": unit.elementary_charge,
    "dipole_moment": unit.elementary_charge * unit.nanometer,
}


class SpeciesEnergies(NamedTuple):
    species: torch.Tensor
    energies: torch.Tensor


class SpeciesAEV(NamedTuple):
    species: torch.Tensor
    aevs: torch.Tensor


class NNPInput:
    __slots__ = (
        "atomic_numbers",
        "positions",
        "atomic_subsystem_indices",
        "per_system_total_charge",
        "pair_list",
        "per_atom_partial_charge",
        "box_vectors",
        "is_periodic",
    )

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        per_system_total_charge: torch.Tensor,
        box_vectors: torch.Tensor = torch.zeros(3, 3),
        is_periodic: torch.Tensor = torch.tensor([False]),
        pair_list: torch.Tensor = torch.tensor([]),
        per_atom_partial_charge: torch.Tensor = torch.tensor([]),
        populate_element_fields: List[str] = None,
    ):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.atomic_subsystem_indices = atomic_subsystem_indices
        self.per_system_total_charge = per_system_total_charge
        self.pair_list = pair_list
        self.per_atom_partial_charge = per_atom_partial_charge
        self.box_vectors = box_vectors
        self.is_periodic = is_periodic

        # Validate inputs
        self._validate_inputs()

    def _populate_element_fields(self):
        for field in self.populate_element_fields:
            for atomic_number in self.atomic_numbers:
                temp_value = element_data(field, atomic_number)

            self.__setattr__(field, atomic_number)

    def _validate_inputs(self):
        # Get shapes of the arrays
        atomic_numbers_shape = self.atomic_numbers.shape
        positions_shape = self.positions.shape
        atomic_subsystem_indices_shape = self.atomic_subsystem_indices.shape

        # Validate dimensions
        if len(atomic_numbers_shape) != 1:
            raise ValueError("atomic_numbers must be a 1D tensor or array")
        if len(positions_shape) != 2 or positions_shape[1] != 3:
            raise ValueError(
                "positions must be a 2D tensor or array with shape [num_atoms, 3]"
            )
        if self.box_vectors.shape[0] != 3 or self.box_vectors.shape[1] != 3:
            print(f"{self.box_vectors.shape}")
            raise ValueError("box_vectors must be a 3x3 tensor or array")

        if len(atomic_subsystem_indices_shape) != 1:
            raise ValueError("atomic_subsystem_indices must be a 1D tensor or array")

        # Validate lengths
        num_atoms = positions_shape[0]
        if atomic_numbers_shape[0] != num_atoms:
            raise ValueError(
                "The size of atomic_numbers and the first dimension of positions must match"
            )
        if atomic_subsystem_indices_shape[0] != num_atoms:
            raise ValueError(
                "The size of atomic_subsystem_indices and the first dimension of positions must match"
            )

    def to_device(self, device: torch.device):
        """Move all tensors in this instance to the specified device."""

        self.atomic_numbers = self.atomic_numbers.to(device)
        self.positions = self.positions.to(device)
        self.atomic_subsystem_indices = self.atomic_subsystem_indices.to(device)
        self.per_system_total_charge = self.per_system_total_charge.to(device)
        self.box_vectors = self.box_vectors.to(device)
        self.is_periodic = self.is_periodic.to(device)
        self.pair_list = self.pair_list.to(device)
        self.per_atom_partial_charge = self.per_atom_partial_charge.to(device)

        return self

    def to_dtype(self, dtype: torch.dtype):
        """Move all **relevant** tensors to dtype."""
        self.positions = self.positions.to(dtype)
        self.box_vectors = self.box_vectors.to(dtype)
        return self


class Metadata:
    """
    A class to structure metadata for neural network potentials.

    Parameters
    ----------
    per_system_energy : torch.Tensor
        Energies for each system.
    atomic_subsystem_counts : torch.Tensor
        The number of atoms in each subsystem.
    atomic_subsystem_indices_referencing_dataset : torch.Tensor
        Indices referencing the dataset.
    number_of_atoms : int
        Total number of atoms.
    per_atom_force : torch.Tensor, optional
        Forces for each atom.
    per_system_dipole_moment : torch.Tensor, optional
        Dipole moments for each system.

    """

    __slots__ = (
        "per_system_energy",
        "atomic_subsystem_counts",
        "atomic_subsystem_indices_referencing_dataset",
        "number_of_atoms",
        "per_atom_force",
        "per_system_dipole_moment",
    )

    def __init__(
        self,
        per_system_energy: torch.Tensor,
        atomic_subsystem_counts: torch.Tensor,
        atomic_subsystem_indices_referencing_dataset: torch.Tensor,
        number_of_atoms: int,
        per_atom_force: torch.Tensor = None,
        per_system_dipole_moment: torch.Tensor = None,
    ):
        self.per_system_energy = per_system_energy
        self.atomic_subsystem_counts = atomic_subsystem_counts
        self.atomic_subsystem_indices_referencing_dataset = (
            atomic_subsystem_indices_referencing_dataset
        )
        self.number_of_atoms = number_of_atoms
        self.per_atom_force = per_atom_force
        self.per_system_dipole_moment = per_system_dipole_moment

    def to_device(self, device: torch.device):
        """Move all tensors in this instance to the specified device."""
        self.per_system_energy = self.per_system_energy.to(device)
        self.per_atom_force = self.per_atom_force.to(device)
        self.atomic_subsystem_counts = self.atomic_subsystem_counts.to(device)
        self.atomic_subsystem_indices_referencing_dataset = (
            self.atomic_subsystem_indices_referencing_dataset.to(device)
        )
        self.per_system_dipole_moment = self.per_system_dipole_moment.to(device)
        return self

    def to_dtype(self, dtype: torch.dtype):
        self.per_system_energy = self.per_system_energy.to(dtype)
        self.per_atom_force = self.per_atom_force.to(dtype)
        self.per_system_dipole_moment = self.per_system_dipole_moment.to(dtype)
        return self


@dataclass
class BatchData:
    nnp_input: NNPInput
    metadata: Metadata

    def to(
        self,
        device: torch.device,
    ):  # NOTE: this is required to move the data to device
        """Move all data in this batch to the specified device and dtype."""
        self.nnp_input = self.nnp_input.to_device(device=device)
        self.metadata = self.metadata.to_device(device=device)
        return self

    def to_device(
        self,
        device: torch.device,
    ):
        """Move all data in this batch to the specified device and dtype."""
        self.nnp_input = self.nnp_input.to_device(device=device)
        self.metadata = self.metadata.to_device(device=device)
        return self

    def to_dtype(
        self,
        dtype: torch.dtype,
    ):
        """Move all data in this batch to the specified device and dtype."""
        self.nnp_input = self.nnp_input.to_dtype(dtype=dtype)
        self.metadata = self.metadata.to_dtype(dtype=dtype)
        return self

    def batch_size(self) -> int:
        """Return the batch size."""
        return self.metadata.per_system_energy.size(dim=0)

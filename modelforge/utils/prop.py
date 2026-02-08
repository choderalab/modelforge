"""
Module of dataclass definitions of properties.
"""

from dataclasses import dataclass
import torch
from typing import NamedTuple, Optional


@dataclass
class PropertyNames:
    atomic_numbers: str  # per-atom atomic numbers in the system
    positions: str  # per-atom positions in the system
    E: str  # per-system total energy
    F: Optional[str] = None  # per-atom forces
    total_charge: Optional[str] = None  # per-system total charge
    dipole_moment: Optional[str] = None  # per-system dipole moment
    spin_multiplicity: Optional[str] = None  # per-system spin multiplicity
    partial_charges: Optional[str] = None  # per-atom partial charge
    quadrupole_moment: Optional[str] = None  # per-system quadrupole moment
    per_atom_spin_multiplicity: Optional[str] = None  # per-atom spin multiplicity


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
        "per_system_spin_state",
    )

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        per_system_total_charge: torch.Tensor,
        box_vectors: torch.Tensor = torch.zeros(3, 3),
        per_system_spin_state: torch.Tensor = torch.zeros(1),
        is_periodic: torch.Tensor = torch.tensor([False]),
        pair_list: torch.Tensor = torch.tensor([]),
        per_atom_partial_charge: torch.Tensor = torch.tensor([]),
    ):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.atomic_subsystem_indices = atomic_subsystem_indices
        self.per_system_total_charge = per_system_total_charge
        self.pair_list = pair_list
        self.per_atom_partial_charge = per_atom_partial_charge
        self.box_vectors = box_vectors
        self.is_periodic = is_periodic
        self.per_system_spin_state = per_system_spin_state

        # Validate inputs
        self._validate_inputs()

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
        self.per_system_spin_state = self.per_system_spin_state.to(device)

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
    per_atom_spin_multiplicity : torch.Tensor, optional
        Per-atom spin multiplicity for each atom in the system.

    """

    __slots__ = (
        "per_system_energy",
        "atomic_subsystem_counts",
        "atomic_subsystem_indices_referencing_dataset",
        "number_of_atoms",
        "per_atom_force",
        "per_system_dipole_moment",
        "per_atom_charge",
        "per_system_quadrupole_moment",
        "per_atom_spin_multiplicity",
    )

    def __init__(
        self,
        per_system_energy: torch.Tensor,
        atomic_subsystem_counts: torch.Tensor,
        atomic_subsystem_indices_referencing_dataset: torch.Tensor,
        number_of_atoms: int,
        per_atom_force: torch.Tensor = None,
        per_system_dipole_moment: torch.Tensor = None,
        per_atom_charge: torch.Tensor = None,
        per_system_quadrupole_moment: torch.Tensor = None,
        per_atom_spin_multiplicity: torch.Tensor = None,
    ):
        self.per_system_energy = per_system_energy
        self.atomic_subsystem_counts = atomic_subsystem_counts
        self.atomic_subsystem_indices_referencing_dataset = (
            atomic_subsystem_indices_referencing_dataset
        )
        self.number_of_atoms = number_of_atoms
        self.per_atom_force = per_atom_force
        self.per_system_dipole_moment = per_system_dipole_moment
        self.per_atom_charge = per_atom_charge
        self.per_system_quadrupole_moment = per_system_quadrupole_moment
        self.per_atom_spin_multiplicity = per_atom_spin_multiplicity

    def to_device(self, device: torch.device):
        """Move all tensors in this instance to the specified device."""
        self.per_system_energy = self.per_system_energy.to(device)
        self.per_atom_force = self.per_atom_force.to(device)
        self.atomic_subsystem_counts = self.atomic_subsystem_counts.to(device)
        self.atomic_subsystem_indices_referencing_dataset = (
            self.atomic_subsystem_indices_referencing_dataset.to(device)
        )
        self.per_system_dipole_moment = self.per_system_dipole_moment.to(device)
        self.per_atom_charge = self.per_atom_charge.to(device)
        self.per_system_quadrupole_moment = self.per_system_quadrupole_moment.to(device)
        self.per_atom_spin_multiplicity = self.per_atom_spin_multiplicity.to(device)
        return self

    def to_dtype(self, dtype: torch.dtype):
        """Move all **relevant** tensors to specified dtype."""
        self.per_system_energy = self.per_system_energy.to(dtype)
        self.per_atom_force = self.per_atom_force.to(dtype)
        self.per_system_dipole_moment = self.per_system_dipole_moment.to(dtype)
        self.per_atom_charge = self.per_atom_charge.to(dtype)
        self.per_system_quadrupole_moment = self.per_system_quadrupole_moment.to(dtype)
        self.per_atom_spin_multiplicity = self.per_atom_spin_multiplicity.to(dtype)
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

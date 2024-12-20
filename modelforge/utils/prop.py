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
        "atomic_groups",
        "atomic_periods",
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
    ):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.atomic_subsystem_indices = atomic_subsystem_indices
        self.per_system_total_charge = per_system_total_charge
        self.pair_list = pair_list
        self.per_atom_partial_charge = per_atom_partial_charge
        self.box_vectors = box_vectors
        self.is_periodic = is_periodic
        self._assign_atomic_group_period()

        # Validate inputs
        self._validate_inputs()

    def _assign_atomic_group_period(self):
        """
        Assign atomic group to the input data.

        Returns
        -------
        torch.Tensor
            The atomic group tensor.
        """
        period = {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 3,
            12: 3,
            13: 3,
            14: 3,
            15: 3,
            16: 3,
            17: 3,
            18: 3,
            19: 4,
            20: 4,
            21: 4,
            22: 4,
            23: 4,
            24: 4,
            25: 4,
            26: 4,
            27: 4,
            28: 4,
            29: 4,
            30: 4,
            31: 4,
            32: 4,
            33: 4,
            34: 4,
            35: 4,
            36: 4,
            37: 5,
            38: 5,
            39: 5,
            40: 5,
            41: 5,
            42: 5,
            43: 5,
            44: 5,
            45: 5,
            46: 5,
            47: 5,
            48: 5,
            49: 5,
            50: 5,
            51: 5,
            52: 5,
            53: 5,
            54: 5,
            55: 6,
            56: 6,
            57: 6,
            72: 6,
            73: 6,
            74: 6,
            75: 6,
            76: 6,
            77: 6,
            78: 6,
            79: 6,
            80: 6,
            81: 6,
            82: 6,
            83: 6,
            84: 6,
            85: 6,
            86: 6,
            87: 7,
            88: 7,
            104: 7,
            105: 7,
            106: 7,
            107: 7,
            108: 7,
            109: 7,
            110: 7,
            111: 7,
            112: 7,
            113: 7,
            114: 7,
            115: 7,
            116: 7,
            117: 7,
            118: 7,
        }
        group = {
            1: 1,
            2: 18,
            3: 1,
            4: 2,
            5: 13,
            6: 14,
            7: 15,
            8: 16,
            9: 17,
            10: 18,
            11: 1,
            12: 2,
            13: 13,
            14: 14,
            15: 15,
            16: 16,
            17: 17,
            18: 18,
            19: 1,
            20: 2,
            21: 3,
            22: 4,
            23: 5,
            24: 6,
            25: 7,
            26: 8,
            27: 9,
            28: 10,
            29: 11,
            30: 12,
            31: 13,
            32: 14,
            33: 15,
            34: 16,
            35: 17,
            36: 18,
            37: 1,
            38: 2,
            39: 3,
            40: 4,
            41: 5,
            42: 6,
            43: 7,
            44: 8,
            45: 9,
            46: 10,
            47: 11,
            48: 12,
            49: 13,
            50: 14,
            51: 15,
            52: 16,
            53: 17,
            54: 18,
            55: 1,
            56: 2,
            57: 3,
            72: 4,
            73: 5,
            74: 6,
            75: 7,
            76: 8,
            77: 9,
            78: 10,
            79: 11,
            80: 12,
            81: 13,
            82: 14,
            83: 15,
            84: 16,
            85: 17,
            86: 18,
            87: 1,
            88: 2,
            104: 4,
            105: 5,
            106: 6,
            107: 7,
            108: 8,
            109: 9,
            110: 10,
            111: 11,
            112: 12,
            113: 13,
            114: 14,
            115: 15,
            116: 16,
            117: 17,
            118: 18,
        }

        self.atomic_groups = torch.tensor(
            [group[atomic_number.item()] for atomic_number in self.atomic_numbers]
        )
        self.atomic_periods = torch.tensor(
            [period[atomic_number.item()] for atomic_number in self.atomic_numbers]
        )

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

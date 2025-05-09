import torch

# This is directly copied from modelforge.utils.prop.NNPInput so we do not need a dependency on modelforge


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

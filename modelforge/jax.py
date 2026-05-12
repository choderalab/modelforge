"""
JAX conversion utilities for modelforge to allow us to wrap the pytorch potential.

This uses dlpack to convert tensors between torch and JAX without copying data,
but there are some special cases to handle.

DLPack only supports signed/unsigned integers, float, and complex dtypes.
The following NNPInput fields require special handling:

  * ``is_periodic``  — torch.bool → cast to torch.int8 before DLPack;
                       cast back to torch.bool on the PyTorch side.
  * ``pair_list``    — may be an empty tensor (shape (0,)) or None;
                       represented as None on the JAX side if empty.
                       This data is populated internally.
  * ``per_atom_partial_charge`` — same as pair_list.

Only differentiable float leaves should be pytree children traced by JAX and passed via DLPack;
everything else (integer indices, boolean flags, optional empties) must live
in aux_data so JAX never tries to trace or DLPack-convert them.

  children  (JAX-traced, DLPack-safe floats)
  ──────────────────────────────────────────
  0  positions                  float32
  1  per_system_total_charge    float32
  2  box_vectors                float32
  3  per_system_spin_state      float32
  4  per_atom_partial_charge    float32 | None

  aux_data  (static, passed through as Python objects)
  ─────────────────────────────────────────────────────
  0  atomic_numbers             int64   (index — not differentiable)
  1  atomic_subsystem_indices   int64   (index — not differentiable)
  2  is_periodic                int8    (bool cast; static per call)
  3  pair_list                  int64 | None (index — not differentiable)
"""

from __future__ import annotations

import torch
import jax
from modelforge.utils.prop import NNPInput
from typing import Union


def _is_empty(tensor: torch.tensor) -> bool:
    """
    Return True if tensor is None or has zero elements.

    This is just a simple helper function so we can represent empty optional tensors as None on the JAX side,
    which avoids DLPack issues with zero-element tensors of ambiguous dtype.

    The pytree registration will carry None cleanly, whereas an empty tensor would cause DLPack errors.

    Parameters
    -----------
    tensor: torch.tensor
        Input tensor to check.

    Returns
    --------
    bool:
        True if empty, False otherwise.
    """
    return tensor is None or tensor.numel() == 0


def _to_bool_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Convert int8-encoded is_periodic back to torch.bool.

    Parameters
    ----------
    t: torch.Tensor
        The tensor to convert.

    Returns
    --------
    torch.Tensor
    """
    if t is None:
        return _to_bool_tensor(torch.tensor([False]))
    if isinstance(t, torch.Tensor):
        return t.to(torch.bool)


def torch_to_jax(tensor: torch.tensor) -> Union[None, jax.numpy.ndarray]:
    """
    Convert a ``torch.Tensor`` to a ``jax.Array`` via DLPack.

    This will also convert bool to int8 if specified.

    Parameters
    ----------
    tensor:
        Source tensor.  ``None`` is returned unchanged.
    bool_as_int8:
        When ``True``, cast boolean tensors to ``int8`` first (DLPack does
        not support the bool dtype).

    Returns
    -------
    Union[None, jax.numpy.ndarray]
    """
    import jax.dlpack as jdlp

    if tensor is None:
        return None
    if _is_empty(tensor):
        return None

    tensor = tensor.detach()

    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.int8)

    return jdlp.from_dlpack(tensor)


def jax_to_torch(
    array: jax.numpy.ndarray, *, as_bool: bool = False
) -> Union[None, torch.Tensor]:
    """
    Convert a ``jax.Array`` to a ``torch.Tensor`` via DLPack.

    Parameters
    ----------
    array:
        Source array.  ``None`` is returned unchanged.
    as_bool:
        When ``True``, cast the resulting tensor from ``int8`` back to
        ``bool`` (reverses the ``bool_as_int8`` encoding).

    Returns
    --------
    Union[None, torch.Tensor]
    """
    if array is None:
        return None

    t = torch.from_dlpack(array)

    if as_bool:
        t = t.to(torch.bool)

    return t


def nnpinput_flatten(nnp_input: NNPInput):
    """
    Flatten NNPInput for JAX pytree tracing.

    Only float fields that JAX may need to differentiate through are children.
    Integer indices and boolean flags go to aux_data so JAX never attempts
    DLPack conversion on unsupported dtypes.

    Parameters
    ----------
    nnp_input: NNPInput
        Instance of NNPInput to flatten for JAX pytree tracing.

    Returns
    -------
    children, aux_data
    """
    children = (
        nnp_input.positions,  # 0  float32 or float64 — differentiable
        nnp_input.per_system_total_charge,  # 1  float32 or float64
        nnp_input.box_vectors,  # 2  float32 or float64
        nnp_input.per_system_spin_state,  # 3  float32 or float64
        nnp_input.per_atom_partial_charge,  # 4  float32 or float64 | None
    )
    aux_data = (
        nnp_input.atomic_numbers,  # 0  int64  — index, not traced
        nnp_input.atomic_subsystem_indices,  # 1  int64  — index, not traced
        nnp_input.is_periodic,  # 2  bool/int8 — static flag ... jax does not like bools, so we need to conver to int8
        nnp_input.pair_list,  # 3  int64 | None — index
    )
    return children, aux_data


def nnpinput_unflatten(aux_data, children):
    """
    Reconstruct NNPInput from pytree leaves (children) and aux_data.


    """
    (
        positions,
        per_system_total_charge,
        box_vectors,
        per_system_spin_state,
        per_atom_partial_charge,
    ) = children
    (
        atomic_numbers,
        atomic_subsystem_indices,
        is_periodic,
        pair_list,
    ) = aux_data
    """
    During JAX abstract tracing (e.g. inside jax.value_and_grad / jax.jit),
    ``children`` contains JAX abstract tracers rather than concrete arrays.
    ``NNPInput.__init__`` calls ``_validate_inputs()`` which does concrete
    shape comparisons — these fail on abstract values.  As such, we need to
    bypass ``__init__`` entirely by allocating a bare instance via
    ``object.__new__`` and setting slots directly.  This is safe because the
    pytree round-trip guarantees the data has already been validated.
    """
    obj = object.__new__(NNPInput)
    obj.atomic_numbers = atomic_numbers
    obj.positions = positions
    obj.atomic_subsystem_indices = atomic_subsystem_indices
    obj.per_system_total_charge = per_system_total_charge
    obj.box_vectors = box_vectors
    obj.per_system_spin_state = per_system_spin_state
    obj.is_periodic = is_periodic
    obj.pair_list = pair_list
    obj.per_atom_partial_charge = per_atom_partial_charge

    return obj


def convert_NNPInput_torch_to_jax(nnp_input: NNPInput) -> NNPInput:
    """Convert every field of a :class:`NNPInput` from torch to JAX in place.

    Fields that cannot pass through DLPack are handled specially:

    * ``is_periodic``            — cast bool → int8 then DLPack.
    * ``pair_list``              — kept as torch.Tensor (lives in aux_data,
                                   never DLPack-converted by the pytree).
    * ``per_atom_partial_charge``— set to None when empty (avoids DLPack
                                   on a zero-element tensor with ambiguous
                                   dtype).
    """
    import copy

    # nnp_input_out = copy.deepcopy(nnp_input)
    nnp_input_out = nnp_input

    # Float fields — straightforward DLPack
    nnp_input_out.positions = torch_to_jax(nnp_input.positions)
    nnp_input_out.per_system_total_charge = torch_to_jax(
        nnp_input.per_system_total_charge
    )
    nnp_input_out.box_vectors = torch_to_jax(nnp_input.box_vectors)
    nnp_input_out.per_system_spin_state = torch_to_jax(nnp_input.per_system_spin_state)

    # Optional float — None when empty so the pytree carries None cleanly
    nnp_input_out.per_atom_partial_charge = torch_to_jax(
        nnp_input.per_atom_partial_charge
    )

    # Integer index fields — stay as torch tensors in aux_data;
    # the pytree registration never DLPack-converts aux_data leaves.
    # (atomic_numbers, atomic_subsystem_indices, pair_list left as-is)
    if _is_empty(nnp_input.pair_list):
        nnp_input_out.pair_list = None
    # atomic_numbers and atomic_subsystem_indices are left as torch.Tensor

    # Boolean — cast to int8 so DLPack works if it ever crosses the boundary,
    # but since is_periodic is in aux_data it stays as a torch.Tensor.
    # We still cast for safety in case callers inspect the field directly.
    nnp_input_out.is_periodic = torch_to_jax(nnp_input.is_periodic)

    return nnp_input_out


def convert_NNPInput_jax_to_torch(nnp_input: NNPInput) -> NNPInput:
    """Convert every field of a :class:`NNPInput` from JAX to torch in place.

    This is the reverse of :func:`convert_NNPInput_torch_to_jax`, so it also handles
    special cases:

    * ``is_periodic``            — cast int8 → bool after DLPack.
    * ``pair_list``              — kept as torch.Tensor (lives in aux_data,
                                   never DLPack-converted by the pytree).
    * ``per_atom_partial_charge``— set to empty tensor when None (reverses
                                   the None encoding used to avoid DLPack on
                                   zero-element tensors).
    """
    # Float fields — straightforward DLPack

    positions = jax_to_torch(nnp_input.positions)
    positions.requires_grad = True

    per_system_total_charge = jax_to_torch(nnp_input.per_system_total_charge)
    box_vectors = jax_to_torch(nnp_input.box_vectors)
    per_system_spin_state = jax_to_torch(nnp_input.per_system_spin_state)

    # Optional float — None means we set it to an empty tensor with the correct dtype
    if nnp_input.per_atom_partial_charge is None:
        per_atom_partial_charge = torch.tensor([], dtype=torch.float32)
    else:
        per_atom_partial_charge = jax_to_torch(nnp_input.per_atom_partial_charge)

    # Integer index fields — stay as torch tensors in aux_data;
    # the pytree registration never DLPack-converts aux_data leaves.
    # (atomic_numbers, atomic_subsystem_indices, pair_list left as-is)
    if nnp_input.pair_list is None:
        pair_list = torch.tensor([], dtype=torch.int64)
    else:
        pair_list = nnp_input.pair_list
    # atomic_numbers and atomic_subsystem_indices are left as torch.Tensor

    # Boolean — cast back from int8 to bool after DLPack.

    is_periodic = jax_to_torch(nnp_input.is_periodic, as_bool=True)

    nnp_input.positions = positions
    nnp_input.per_system_total_charge = per_system_total_charge
    nnp_input.box_vectors = box_vectors
    nnp_input.is_periodic = is_periodic
    nnp_input.pair_list = pair_list
    nnp_input.per_atom_partial_charge = per_atom_partial_charge
    nnp_input.atomic_numbers = nnp_input.atomic_numbers
    nnp_input.atomic_subsystem_indices = nnp_input.atomic_subsystem_indices
    nnp_input.per_system_spin_state = per_system_spin_state
    return nnp_input

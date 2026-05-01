import torch

# from modelforge.openmm.utils import NNPInput
from modelforge.utils.prop import NNPInput
from typing import List, Optional
from enum import Enum
import numpy as np
from functools import partial


class NeighborlistStrategy(Enum):
    """
    Enum class for the neighborlist strategy to use

    """

    brute_nsq = "brute_nsq"
    verlet_nsq = "verlet_nsq"


def _build_nnp_input(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    is_periodic: bool,
    precision: torch.dtype,
    device: str,
    box_vectors: np.ndarray | None,
    per_system_total_charge: Optional[int] = None,
    per_system_spin_state: Optional[int] = None,
) -> NNPInput:
    """
    Convert raw OpenMM position data into the dict expected by ModelForge.

    Parameters
    -----------
    atomic_numbers: np.ndarray[
        List of atomic numbers for each atom in the system, in the same order as OpenMM
    positions: np.ndarray
        (N, 3) array of atomic positions in nm, as passed by OpenMM
    is_periodic: bool
        is the system periodic
    precision: torch.dtype
        Precision of the position and box vector tensors
    device: str
        Torch device string, e.g. 'cpu' or 'cuda'
    box_vectors: np.ndarray |None
        box vectors in nm, as passed by OpenMM
    per_system_total_charge: Optional[int]
        The total charge of the system
    per_system_spin_state:  Optional[int]
        To spin multiplicity of the system


    Returns
    -------
    NNPInput for input to modelforge potential
    """
    atomic_numbers = atomic_numbers.reshape(-1)
    n_atoms = atomic_numbers.shape[0]

    positions_tensor = torch.tensor(positions, dtype=precision, device=device)
    atomic_numbers_tensor = torch.tensor(
        atomic_numbers, dtype=torch.int64, device=device
    )

    atomic_subsystem_indices = torch.zeros(n_atoms, dtype=torch.long, device=device)

    if box_vectors is not None:
        box_vectors_tensor = torch.tensor(box_vectors, dtype=precision, device=device)
    else:
        box_vectors_tensor = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=precision,
            device=device,
        )
    total_charge_tensor = torch.tensor(
        [[per_system_total_charge]], dtype=torch.int64, device=device
    )
    total_spin_tensor = torch.tensor(
        [[per_system_spin_state]], dtype=torch.int64, device=device
    )

    return NNPInput(
        atomic_numbers=atomic_numbers_tensor,
        positions=positions_tensor.detach().requires_grad_(True),
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=total_charge_tensor,
        per_system_spin_state=total_spin_tensor,
        is_periodic=torch.tensor([is_periodic]),
        box_vectors=box_vectors_tensor,
    )


def generate_compute(
    potential: torch.nn.Module,
    atomic_numbers: list[int],
    per_system_total_charge: int = 0,
    per_system_spin_multiplicity: int = 1,
    periodic: bool = False,
    precision: torch.dtype = torch.float32,
    device: str = "cpu",
    neighborlist_strategy: NeighborlistStrategy = "brute_nsq",
    neighborlist_verlet_skin: float = 0.1,
) -> partial:
    """
    Wrapper to work with OpenMM 8.5+ ``PythonForce``

    Returns a function to pass to PythonForce.

    Parameters
    ----------
    potential : modelforge potential instance
        Any ModelForge potential object that implements ``forward(nnp_input)``.
        The model is moved to ``device`` and put in eval mode in ``__init__``.

    atomic_numbers : list[int]
        Atomic numbers for every particle in the System, in the same order as
        ``system.addParticle`` / ``context.setPositions``.
    per_system_total_charge: [int] =0
        Integer total charge of the system.
    per_system_spin_multiplicity: [int] =1
        Integer total spin multiplicity of the system.
    periodic : bool, optional
        Set to ``True`` for periodic simulations. OpenMM will then pass box
        vectors to ``computeForce``. Default: ``False``.
    precision : torch.dtype, optional
        Floating-point precision used internally. ``torch.float32`` (default)
        works for all production ModelForge models. Use ``torch.float64`` only
        if the model was trained at double precision.
    device : str, optional
        Torch device string, e.g. ``'cpu'`` (default) or ``'cuda'``
    neighborlist_strategy: NeighborlistStrategy, default brute_nsq
        Neighborlist strategy to use for the potential. Options are:
         "brute_nsq" which is the brute force neighbor list which scales as N**2
         "verlet_nsq" which is a verlet neighborlist, where rebuilds are N**2
    neighborlist_verlet_skin: [float], default 0.1
        Skin distance for verlet neighborlist rebuilds, in nm. Only used if neighborlist_strategy is "verlet_nsq".
    """

    potential.to(device)
    potential.set_neighborlist_strategy(neighborlist_strategy, neighborlist_verlet_skin)
    potential.eval()

    atomic_numbers = np.array(atomic_numbers)
    compute = partial(
        _compute_modelforge,
        potential=potential,
        atomic_numbers=atomic_numbers,
        per_system_total_charge=per_system_total_charge,
        per_system_spin_multiplicity=per_system_spin_multiplicity,
        is_periodic=periodic,
        precision=precision,
        device=device,
    )

    return compute


def _compute_modelforge(
    state,
    potential,
    atomic_numbers: np.ndarray,
    per_system_total_charge: int,
    per_system_spin_multiplicity: int,
    is_periodic: bool,
    precision: torch.dtype,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """
    Evaluate the NNP potential and return energy + forces.

    Parameters
    ----------
    state: openmm.State
        openmm state object
    potential: modelforge.Potential instance
        Potential instance that implements ``forward(nnp_input)``.
    atomic_numbers: np.ndarray
        Atomic numbers for every particle in the System, in the same order as
    per_system_total_charge: int
        Integer total charge of the system.
    per_system_spin_multiplicity: int
        Integer total spin multiplicity of the system.
    is_periodic: bool
        Whether the system is periodic. If True, box vectors will be passed in the state and included in the NNP input.

    Returns
    -------
    energy : float
        Potential energy in **kJ/mol**.
    forces : (N, 3) float64 ndarray
        Forces in **kJ/mol/nm**.
    """
    import openmm.unit as omm_unit

    nnp_input = _build_nnp_input(
        atomic_numbers=atomic_numbers,
        positions=state.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer),
        is_periodic=is_periodic,
        box_vectors=(
            state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(omm_unit.nanometer)
            if is_periodic
            else None
        ),
        precision=precision,
        device=device,
        per_system_total_charge=per_system_total_charge,
        per_system_spin_state=per_system_spin_multiplicity,
    )

    # just make sure that we are really on the right device
    nnp_input.to_device(device)
    nnp_input.to_dtype(precision)

    output = potential(nnp_input)

    energy = output["per_system_energy"]
    grad = torch.autograd.grad(
        energy.sum(), nnp_input.positions, create_graph=False, retain_graph=False
    )[0]

    force = -grad.detach().cpu().numpy().astype(np.float64)

    energy = energy.detach().cpu().numpy().astype(np.float64).sum()

    return energy, force

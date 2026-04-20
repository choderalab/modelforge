import torch

# from modelforge.openmm.utils import NNPInput
from modelforge.utils.prop import NNPInput
from typing import List, Optional
from enum import Enum
import openmm
import numpy as np


class NeighborlistStrategy(Enum):
    """
    Enum class for the neighborlist strategy to use

    """

    brute_nsq = "brute_nsq"
    verlet_nsq = "verlet_nsq"


def _build_nnp_input(
    atomic_numbers: list[int],
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
    atomic_numbers: list[int]
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
    n_atoms = len(atomic_numbers)

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
        positions=positions_tensor,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=total_charge_tensor,
        per_system_spin_state=total_spin_tensor,
        is_periodic=torch.tensor([is_periodic]),
        box_vectors=box_vectors_tensor,
    )


class ModelForgeForce:
    """
    Wrapper to work with OpenMM 8.5+ ``PythonForce``


    Parameters
    ----------
    potential : modelforge potential instance
        Any ModelForge potential object that implements ``forward(input_dict)``.
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

    def __init__(
        self,
        potential: torch.nn.Module,
        atomic_numbers: list[int],
        per_system_total_charge: int = 0,
        per_system_spin_multiplicity: int = 1,
        periodic: bool = False,
        precision: torch.dtype = torch.float32,
        device: str = "cpu",
        neighborlist_strategy: NeighborlistStrategy = "brute_nsq",
        neighborlist_verlet_skin: float = 0.1,
    ) -> None:

        self._potential = potential.to(device)
        self._potential.set_neighborlist_strategy(
            neighborlist_strategy, neighborlist_verlet_skin
        )
        self._potential.eval()

        self._atomic_numbers = list(atomic_numbers)
        self._n_atoms = len(atomic_numbers)
        self._periodic = periodic
        self._dtype = precision
        self._device = device
        self._per_system_total_charge = per_system_total_charge
        self._per_system_spin_multiplicity = per_system_spin_multiplicity

    def compute(self, state) -> tuple[float, np.ndarray]:
        """
        Evaluate the NNP potential and return energy + forces.

        Parameters
        ----------
        state, openmm state object
        Returns
        -------
        energy : float
            Potential energy in **kJ/mol**.
        forces : (N, 3) float64 ndarray
            Forces in **kJ/mol/nm**.
        """
        import openmm.unit as omm_unit

        nnp_input = _build_nnp_input(
            atomic_numbers=self._atomic_numbers,
            positions=state.getPositions(asNumpy=True).value_in_unit(
                omm_unit.nanometer
            ),
            is_periodic=self._periodic,
            box_vectors=(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    omm_unit.nanometer
                )
                if self._periodic
                else None
            ),
            precision=self._dtype,
            device=self._device,
            per_system_total_charge=self._per_system_total_charge,
            per_system_spin_state=self._per_system_spin_multiplicity,
        )

        # Enable gradient on positions so we can autograd the forces
        positions_t: torch.Tensor = nnp_input.positions.detach().requires_grad_(True)
        nnp_input.positions = positions_t

        output = self._potential(nnp_input)
        energy = torch.Tensor = output["per_system_energy"]

        # Forces = -dE/dr  (autograd, result is eV/Å)
        grad = torch.autograd.grad(
            energy.sum(), positions_t, create_graph=False, retain_graph=False
        )[0]
        forces = -grad.detach().cpu().numpy().astype(np.float64)

        return (
            energy.detach().cpu().numpy().astype(np.float64).sum()
            * omm_unit.kilojoules_per_mole,
            forces * omm_unit.kilojoules_per_mole / omm_unit.nanometer,
        )

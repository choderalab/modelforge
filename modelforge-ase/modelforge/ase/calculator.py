from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

import numpy as np
import torch
from enum import Enum


from ase.calculators.calculator import Calculator, all_changes

if TYPE_CHECKING:
    from ase import Atoms


# Unit conventions
# ----------------
# +------------------+------------------+-------------------+
# | Quantity         | modelforge       | ASE internal      |
# +==================+==================+===================+
# | Positions        | nanometres (nm)  | Ångström (Å)      |
# | Energy           | kJ mol⁻¹         | eV                |
# | Forces (-dE/dr)  | kJ mol⁻¹ nm⁻¹    | eV Å⁻¹            |
# +------------------+------------------+-------------------+
#
# Note, while inputs to modelforge will have units associated with them (via openff-units),
# internal tensors and output do not have units associated with them.
#
# 1 kJ mol⁻¹ = 0.010364274 eV
# 1 kJ mol⁻¹ nm⁻¹ = 0.0010364274 eV Å⁻¹

# conversion factors
_kJ_PER_MOL_TO_EV = 0.010364274
_KJ_PER_MOL_NM_TO_EV_ANG = 0.0010364274
_ANG_TO_NM = 0.1


class NeighborlistStrategy(Enum):
    """
    Enum class for the neighborlist strategy to use

    """

    brute_nsq = "brute_nsq"
    verlet_nsq = "verlet_nsq"


# note, we'll need to
def _init_nnp_input(
    atoms: Atoms,
    device: torch.device,
    precision: torch.dtype,
    per_system_total_charge: Optional[int] = None,
    per_system_spin_state: Optional[int] = None,
):
    """
    Convert an ASE ``Atoms`` object into a modelforge ``NNPinput`` Dataclass, applying unit conversions.

    Parameters
    ----------
    atoms: [Atoms]
        An ASE ``Atoms`` object representing the system
    device: Optional[torch.device]
        Pytorch device to use for all tensors.
    per_system_total_charge: Optional[int]
        Total charge of the system. This is optional and only required if a system has total charge embedding.
    per_system_spin_state: Optional[int]
        Total spin multiplicity of the system.  This is optional and only required if a system has total spin embedding.


    """
    from modelforge.utils.prop import NNPInput

    n_atoms = len(atoms)

    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(), dtype=torch.int64, device=device
    )  # (n_atoms,)

    positions = torch.tensor(
        atoms.get_positions() * _ANG_TO_NM,
        dtype=precision,
        device=device,
        requires_grad=True,
    )  # (n_atoms, 3)

    # For a single system (batch size = 1) every atom belongs to subsystem 0.
    atomic_subsystem_indices = torch.zeros(n_atoms, dtype=torch.int64, device=device)

    total_charge_tensor = torch.tensor(
        [[per_system_total_charge]], dtype=torch.int64, device=device
    )
    total_spin_tensor = torch.tensor(
        [[per_system_spin_state]], dtype=torch.int64, device=device
    )

    is_periodic = False
    # Periodic boundary conditions: pass cell if PBC is active.
    pbc = atoms.get_pbc()
    if pbc.any():
        box_vectors = torch.tensor(
            atoms.get_cell()[:] * _ANG_TO_NM,  # (3, 3) in Å
            dtype=precision,
            device=device,
        )
        is_periodic = True
    else:
        box_vectors = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=precision,
        )

    return NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=total_charge_tensor,
        per_system_spin_state=total_spin_tensor,
        is_periodic=torch.tensor([is_periodic]),
        box_vectors=box_vectors,
    )


class ModelForgeCalculator(Calculator):
    """
    Energy-only ASE Calculator backed by a modelforge ``Potential``.

    Forces are obtained via ASE's finite-difference numerical differentiation,
    which is slower but works for any potential regardless of whether it
    implements analytic force output.

    Parameters
    ----------
    potential: [Potential]
        A modelforge ``Potential`` instance (``nn.Module``).
    per_system_total_charge: [int] =0
        Integer total charge of the system.
    per_system_spin_multiplicity: [int] =1
        Integer total spin multiplicity of the system.
    device: Optional[str | torch.device]
        Device for pytorch tensors.
    precision: [torch.device], default torch.float32
        precision for pytorch float tensors.
    neighborlist_strategy: NeighborlistStrategy, default brute_nsq
        Neighborlist strategy to use for the potential. Options are:
         "brute_nsq" which is the brute force neighbor list which scales as N**2
         "verlet_nsq" which is a verlet neighborlist, where rebuilds are N**2
    neighborlist_verlet_skin: [float], default 0.1
        Skin distance for verlet neighborlist rebuilds, in nm. Only used if neighborlist_strategy is "verlet_nsq".

    """

    implemented_properties: List[str] = ["energy", "forces"]

    def __init__(
        self,
        potential,
        per_system_total_charge: int = 0,
        per_system_spin_multiplicity: int = 1,
        device: Optional[str | torch.device] = None,
        precision: torch.dtype = torch.float32,
        neighborlist_strategy: NeighborlistStrategy = "brute_nsq",
        neighborlist_verlet_skin: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.potential = potential.to(device).eval()
        # set the neighborlist strategy for the potential
        self.potential.set_neighborlist_strategy(
            neighborlist_strategy, neighborlist_verlet_skin
        )

        self.per_system_total_charge = per_system_total_charge
        self.device = torch.device(device)
        self.precision = precision
        self.per_system_spin_multiplicity = per_system_spin_multiplicity

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # this will perform conversion of positions and boxes
        nnp_input = _init_nnp_input(
            atoms=self.atoms,
            per_system_total_charge=self.per_system_total_charge,
            device=self.device,
            precision=self.precision,
            per_system_spin_state=self.per_system_spin_multiplicity,
        )

        output: dict = self.potential(nnp_input)

        energy_kJ = output["per_system_energy"]
        self.results["energy"] = float(energy_kJ[0].cpu().item()) * _kJ_PER_MOL_TO_EV

        if "per_atom_force" in output:
            forces = output["per_atom_force"]
            self.results["forces"] = (
                forces.cpu().numpy().astype(np.float64) * _KJ_PER_MOL_NM_TO_EV_ANG
            )
        else:
            with torch.enable_grad():
                grad = torch.autograd.grad(
                    energy_kJ.sum(),
                    nnp_input.positions,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                if grad is None:
                    raise RuntimeWarning("Force calculation did not return a gradient")
                # check if nan in the gradient
                if torch.isnan(grad).any():
                    raise RuntimeError(
                        "Gradient of energy used for force calculation contains NaN values."
                    )

            forces = -grad.detach()  #
            self.results["energy"] = (
                energy_kJ[0].detach().cpu().item() * _kJ_PER_MOL_TO_EV
            )
            self.results["forces"] = (
                forces.cpu().numpy().astype(np.float64) * _KJ_PER_MOL_NM_TO_EV_ANG
            )

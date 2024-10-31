import torch

from modelforge.openmm.utils import NNPInput
from typing import List, Optional
from enum import Enum


class NeighborlistStrategy(Enum):
    """
    Enum class for the neighborlist strategy to use

    """

    brute_nsq = "brute_nsq"
    verlet_nsq = "verlet_nsq"


class Potential(torch.nn.Module):
    """
    Class that wraps a modelforge potential, making it compatible with OpenMM TorchForce

    """

    def __init__(
        self,
        modelforge_potential_path: str,
        atomic_numbers: List,
        is_periodic: bool,
        neighborlist_strategy: NeighborlistStrategy = "brute_nsq",
        neighborlist_verlet_skin: float = 0.1,
        energy_contributions: List[str] = [
            "per_system_energy"
        ],  # this will be a list of dictionary keys to sum into the energy, e.g., local_energy, coulombic_energy, vdw_energy
    ):
        super().__init__()

        # Store the atomic numbers
        self.atomic_numbers = torch.tensor(atomic_numbers).squeeze()
        self.is_periodic = is_periodic
        self.energy_contributions = energy_contributions

        n_atoms = self.atomic_numbers.shape[0]

        self.precision = torch.float32

        self.nnp_input = NNPInput(
            atomic_numbers=self.atomic_numbers,
            positions=torch.zeros((n_atoms, 3), dtype=self.precision),
            atomic_subsystem_indices=torch.zeros(n_atoms),
            is_periodic=torch.tensor([self.is_periodic]),
            per_system_total_charge=torch.zeros(1),
        )
        self.modelforge_potential = torch.jit.load(modelforge_potential_path)

        self.modelforge_potential.neighborlist.strategy = neighborlist_strategy
        if neighborlist_strategy == "verlet_nsq":
            self.modelforge_potential.neighborlist.skin = neighborlist_verlet_skin
            self.modelforge_potential.neighborlist.half_skin = (
                neighborlist_verlet_skin * 0.5
            )
            self.modelforge_potential.neighborlist.cutoff_plus_skin = (
                self.modelforge_potential.neighborlist.cutoff + neighborlist_verlet_skin
            )

    def forward(
        self, positions: torch.Tensor, box_vectors: Optional[torch.Tensor] = None
    ):
        if box_vectors is None:
            box_vectors = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=self.precision,
            )
        # if the system isn't periodic, we won't do anything with the box vectors, but we still need to pass them
        self.nnp_input.positions = positions.to(self.precision)
        self.nnp_input.box_vectors = box_vectors.to(self.precision)

        output = self.modelforge_potential.forward_for_jit_inference(
            atomic_numbers=self.nnp_input.atomic_numbers,
            positions=self.nnp_input.positions,
            atomic_subsystem_indices=self.nnp_input.atomic_subsystem_indices,
            per_system_total_charge=self.nnp_input.per_system_total_charge,
            pair_list=self.nnp_input.pair_list,
            per_atom_partial_charge=self.nnp_input.per_atom_partial_charge,
            box_vectors=self.nnp_input.box_vectors,
            is_periodic=self.nnp_input.is_periodic,
        )

        energy = torch.zeros(1, dtype=self.precision)
        for key in self.energy_contributions:
            energy = energy.add(output[key])

        return energy

import torch

from modelforge_openmm.utils import NNPInput
from typing import List
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
        # energy_contributions: List[str], # this will be a list of dictionary keys to sum into the energy, e.g., local_energy, coulombic_energy, vdw_energy
    ):
        super().__init__()

        # Store the atomic numbers
        self.atomic_numbers = torch.tensor(atomic_numbers).squeeze()
        self.is_periodic = is_periodic
        n_atoms = self.atomic_numbers.shape[0]

        self.precision = torch.float32

        self.nnp_input = NNPInput(
            atomic_numbers=self.atomic_numbers,
            positions=torch.tensor((n_atoms, 3), dtype=self.precision),
            atomic_subsystem_indices=torch.zeros(n_atoms),
            is_periodic=torch.tensor([self.is_periodic]),
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
        self,
        positions,
        box_vectors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ):
        # if the system isn't periodic, we won't do anything with the box vectors, but we still need to pass them
        self.nnp_input.positions = torch.tensor(positions, dtype=self.precision)
        self.nnp_input.box_vectors = torch.tensor(box_vectors, dtype=self.precision)

        energy = self.modelforge_potential(NNPInput)["per_molecule_energy"]

        return energy

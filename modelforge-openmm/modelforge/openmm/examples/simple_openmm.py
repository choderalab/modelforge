"""
Demonstrate wrapping a modelforge potential for use with OpenMM TorchForce

"""

# We need to first define the system we want to evaluate
# this is needed because TorchForce will not allow us to pass the atomic numbers to the forward method
# we can just initialize a simple water molecule topology and positions

from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

water, positions = openmm_water_topology()
atomic_numbers = [atom.element.atomic_number for atom in water.atoms()]

# We next need to read in the modelforge potential torchscript file
from importlib import resources

potential_file_path = resources.files("modelforge.openmm.examples.data").joinpath(
    "ani2x_test.pt"
)

# set up the potential to be used with the system
from modelforge.openmm.potential import Potential
import torch

ani2x_potential = Potential(
    modelforge_potential_path=potential_file_path,
    atomic_numbers=atomic_numbers,
    is_periodic=False,
    neighborlist_strategy="brute_nsq",
    energy_contributions=["per_system_energy"],
)

ani2x_jit_potential = torch.jit.script(ani2x_potential)

# Let us do a spot check to ensure the potential is working

print(ani2x_potential(torch.tensor(positions)))
print(ani2x_jit_potential(torch.tensor(positions)))

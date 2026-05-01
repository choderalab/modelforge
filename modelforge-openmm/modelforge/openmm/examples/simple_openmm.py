"""
Demonstrate wrapping a modelforge potential for use with OpenMM PythonForce

"""

from modelforge.utils.io import get_path_string
from modelforge.openmm.examples import data
from modelforge.potential.potential import load_inference_model_from_checkpoint

# checkpoint file is saved in tests/data
checkpoint_file_path = get_path_string(data) + "/model.ckpt"
potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)

# helper functions to load up a water topology and positions for use in OpenMM.
from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

water, positions = openmm_water_topology()
atomic_numbers = [atom.element.atomic_number for atom in water.atoms()]

# initialize the compute object that will be used in the OpenMM PythonForce.
from modelforge.openmm.potential import generate_compute

comp = generate_compute(potential=potential, atomic_numbers=atomic_numbers)

# set up the PythonForce with the compute object.  This will allow us to use the potential for inference within OpenMM.
from openmm import PythonForce

system_force = PythonForce(comp)

# OpenMM simulation setup
import openmm
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    nanometer,
    kilojoules_per_mole,
)

system = openmm.System()
for atom in water.atoms():
    system.addParticle(atom.element.mass)

system.addForce(system_force)

import sys
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation, StateDataReporter

# Create an integrator with a time step of 1 fs
temperature = 298.15 * kelvin
frictionCoeff = 1 / picosecond
timeStep = 0.01 * femtosecond
integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

# Create a simulation and set the initial positions and velocities
simulation = Simulation(water, system, integrator)
simulation.context.setPositions(positions)

reporter = StateDataReporter(
    file=sys.stdout,
    reportInterval=1,
    step=True,
    time=True,
    potentialEnergy=True,
)
simulation.reporters.append(reporter)
simulation.step(10)

import torch
import pytest
from openmm.app import *

from modelforge.potential import _Implemented_NNPs as Implemented_NNPs


@pytest.mark.parametrize("is_periodic", [True, False])
# @pytest.mark.parametrize(
#     "potential_name", Implemented_NNPs.get_all_neural_network_names()
# )
@pytest.mark.parametrize(
    "potential_name", ["SCHNET", "ANI2X", "PHYSNET", "PAINN", "SAKE"]
)
def test_potential_wrapping(is_periodic, potential_name, prep_temp_dir):
    from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

    water_topology, water_positions = openmm_water_topology()
    water_atomic_numbers = [
        atom.element.atomic_number for atom in water_topology.atoms()
    ]

    from modelforge.utils.misc import load_configs_into_pydantic_models

    # just use the helper function even though we will not do anything with the dataset
    config = load_configs_into_pydantic_models(potential_name, "phalkethoh")

    # generate the ani2x potential using the NeuralNetworkPotentialFactory
    # note this is not a trained potential

    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    modelforge_potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        potential_seed=42,
        jit=False,
    )

    from modelforge.openmm.utils import NNPInput

    if is_periodic:
        box_vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        box_vectors = torch.zeros(3, 3)

    n_atoms = len(water_atomic_numbers)
    nnp_input = NNPInput(
        atomic_numbers=torch.tensor(water_atomic_numbers).squeeze(),
        positions=torch.tensor(water_positions, dtype=torch.float32),
        atomic_subsystem_indices=torch.zeros(n_atoms),
        is_periodic=torch.tensor([is_periodic]),
        per_system_total_charge=torch.zeros(1),
        box_vectors=box_vectors,
    )

    modelforge_energy = modelforge_potential(nnp_input)["per_system_energy"]

    from modelforge.openmm.potential import ModelForgeCompute

    openmm_force_compute = ModelForgeCompute(
        potential=modelforge_potential,
        atomic_numbers=water_atomic_numbers,
        is_periodic=is_periodic,
        neighborlist_strategy="verlet_nsq",
    ).generate_compute()

    import openmm

    system = openmm.System()
    for atom in water_topology.atoms():
        system.addParticle(atom.element.mass)

    force = openmm.PythonForce(modelforge_potential)

    system.addForce(force)

    import sys
    from openmm import LangevinMiddleIntegrator
    from openmm.app import Simulation, StateDataReporter
    from openmm.unit import kelvin, picosecond, femtosecond

    # Create an integrator with a time step of 1 fs
    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

    # Create a simulation and set the initial positions and velocities
    simulation = Simulation(water_topology, system, integrator)
    simulation.context.setPositions(water_positions)

    state = simulation.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print(potential_energy)
    print(modelforge_energy)
    assert np.isclose(potential_energy, modelforge_energy)
    # Configure a reporter to print to the console every 0.1 ps (100 steps)
    reporter = StateDataReporter(
        file=sys.stdout,
        reportInterval=1,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
    )
    simulation.reporters.append(reporter)

    # Run the simulation
    simulation.step(10)


def test_openmm(prep_temp_dir):
    from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

    water, positions = openmm_water_topology()
    atomic_numbers = [atom.element.atomic_number for atom in water.atoms()]

    # We next need to read in the modelforge potential torchscript file
    from importlib import resources

    potential_file_path = resources.files("modelforge.openmm.examples.data").joinpath(
        "ani2x_test.pt"
    )

    from modelforge.openmm.potential import Potential
    import torch

    ani2x_potential = Potential(
        modelforge_potential_path=potential_file_path,
        atomic_numbers=atomic_numbers,
        is_periodic=False,
        neighborlist_strategy="verlet_nsq",
        energy_contributions=["per_system_energy"],
    )

    ani2x_jit_potential = torch.jit.script(ani2x_potential)

    import openmm

    system = openmm.System()
    for atom in water.atoms():
        system.addParticle(atom.element.mass)

    from openmmtorch import TorchForce

    force = TorchForce(ani2x_jit_potential)

    system.addForce(force)

    import sys
    from openmm import LangevinMiddleIntegrator
    from openmm.app import Simulation, StateDataReporter
    from openmm.unit import kelvin, picosecond, femtosecond

    # Create an integrator with a time step of 1 fs
    temperature = 298.15 * kelvin
    frictionCoeff = 1 / picosecond
    timeStep = 1 * femtosecond
    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

    # Create a simulation and set the initial positions and velocities
    simulation = Simulation(water, system, integrator)
    simulation.context.setPositions(positions)

    # Configure a reporter to print to the console every 0.1 ps (100 steps)
    reporter = StateDataReporter(
        file=sys.stdout,
        reportInterval=1,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
    )
    simulation.reporters.append(reporter)

    # Run the simulation
    simulation.step(10)

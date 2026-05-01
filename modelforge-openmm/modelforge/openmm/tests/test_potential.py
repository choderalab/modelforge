import torch
import pytest
from openmm.app import *

from modelforge.potential import _Implemented_NNPs as Implemented_NNPs


def test_openmm_wrapping_with_checkpoint_file():

    # We next need to read in the modelforge potential torchscript file
    from modelforge.utils.io import get_path_string
    from modelforge.openmm.examples import data

    checkpoint_file_path = get_path_string(data) + "/model.ckpt"

    from modelforge.potential.potential import load_inference_model_from_checkpoint

    potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)
    from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

    water, positions = openmm_water_topology()
    atomic_numbers = [atom.element.atomic_number for atom in water.atoms()]

    from modelforge.openmm.potential import generate_compute
    from modelforge.openmm.potential import _build_nnp_input
    import openmm
    from openmm import PythonForce
    from openmm.unit import (
        kelvin,
        picosecond,
        femtosecond,
        nanometer,
        kilojoules_per_mole,
    )
    import numpy as np

    system = openmm.System()
    for atom in water.atoms():
        system.addParticle(atom.element.mass)

    comp = generate_compute(potential=potential, atomic_numbers=atomic_numbers)

    system_force = PythonForce(comp)

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
        temperature=True,
    )
    simulation.reporters.append(reporter)
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True
    )
    energy = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)
    print(force)
    assert np.isclose(energy.value_in_unit(kilojoules_per_mole), 5104.08740234375)
    assert np.allclose(
        force.value_in_unit(kilojoules_per_mole / nanometer),
        np.array(
            [
                [0.0, -278369.9375, 0.0],
                [647955.0625, 139184.96875, 0.0],
                [-647955.0625, 139184.96875, 0.0],
            ]
        ),
    )
    simulation.step(10)
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True
    )
    energy = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)

    # we want to check to ensure we aren't just getting zero energies and forces while running
    assert not np.isclose(
        energy.value_in_unit(kilojoules_per_mole),
        0.0,
    )
    assert not np.allclose(force.value_in_unit(kilojoules_per_mole / nanometer), 0.0)


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

    from modelforge.utils.prop import NNPInput

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

    water, positions = openmm_water_topology()
    atomic_numbers = [atom.element.atomic_number for atom in water.atoms()]

    from modelforge.openmm.potential import generate_compute
    from modelforge.openmm.potential import _build_nnp_input
    import openmm
    from openmm import PythonForce
    from openmm.unit import (
        kelvin,
        picosecond,
        femtosecond,
        nanometer,
        kilojoules_per_mole,
    )
    import numpy as np

    system = openmm.System()
    for atom in water.atoms():
        system.addParticle(atom.element.mass)

    comp = generate_compute(
        potential=modelforge_potential, atomic_numbers=atomic_numbers
    )

    system_force = PythonForce(comp)

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
        temperature=True,
    )
    simulation.reporters.append(reporter)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy()

    assert np.allclose(energy.value_in_unit(kilojoules_per_mole), modelforge_energy)

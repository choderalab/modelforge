Inference Mode
##########################

Inference mode is a mode allows us to use the trained model to make predictions. Given that a key usage of the inference mode will be molecule simulation, more efficient schemes for calculating interacting pairs are needed.

Neighborlists
------------------------------------------
Currently, there are two neighborlist strategies implemented within modelforge for inference, the brute force neighbolist and Verlet neighborlist (implemented within a single class :class:`~modelforge.potential.NeighborlistForInference`).    Both neighborlists support periodic and not periodic orthogonal boxes.

The neighborlist strategy can be toggled during potential setup via the `inference_neighborlist_strategy` parameter passed to the :class:`~modelforge.potential.models.NeuralNetworkPotentialFactory`.  The default is the Verlet neighborlist ("verlet_nsq"); brute can be set via "brute_nsq".  This can also be set via set at run time in the potential via Potential.set_neighborlist_strategy(strategy, skin).

Brute force neighborlist
^^^^^^^^^^^^^^^^^^^^^^^^
The brute force neighborlist calculates the pairs within the interaction cutoff by considering all possible pairs each time called, via an order N^2 operation. Typically this approach should only be used for very system sizes, given the scaling; furthermore the N^2 approach used to generate this list utilizes a large amount of memory as the system size grows.



Verlet neighborlist
^^^^^^^^^^^^^^^^^^^^^^^^

The Verlet neighborlist operates under the assumption that under short time windows, the local environment around a given particle does not change significantly.  As such, information about this local environment can be reused between subsequent steps, eliminating the need for a costly build step.

To do this, the local environment of a given particle is identified and saved in a list (e.g., we can call this the verlet list), using the criteria pair distance < cutoff + skin.  The skin is a user modifiable distance that captures a region of space beyond the interaction cutoff.  In the current implementation, this verlet list is generated using the same order N^2 approach as the brute for scheme.  Again, because positions are correlated with time, we typically can avoid performing another order N^2 calculation for several timesteps.  Steps in between rebuilds scale as order N*M, where M is the average number of neighbors (which is typically much less than N).  In our implementation, the verlet list is automatically regenerated when any given particle moves more than skin/2 (since the last build), to ensure that interactions are not missed.

Larger values of skin result in longer time periods between rebuilds, but also typically increase the number of calculations that need to be perform at each timestep (as M will typically be larger). As such, this value can have a significant impact on performance of this calculation.

Note: Since this utilizes an N^2 computation within Torch, the memory footprint may be problematic as system size grows. A cell list based approach will be implemented in the future.


Load inference potential from training checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the trained model for inference, the checkpoint file generated during training must be loaded. The checkpoint file contains the model's weights, optimizer state, and other training-related information. The `load_inference_model_from_checkpoint` function provides a convenient way to load the checkpoint file and generate an inference model. 

.. code-block:: python

    from modelforge.potential.models import load_inference_model_from_checkpoint

    inference_model = load_inference_model_from_checkpoint(checkpoint_file)



.. toctree::
   :maxdepth: 2
   :caption: Contents:

Note, prior to merging PR #299, checkpoint files and state_dicts did not save the `only_unique_pairs` bool parameter, needed to properly generate neighbor information. As such, if you are using a checkpoint file generated prior to this PR, you will need to set this parameter manually.  This can be done by passing the `only_unique_pairs` parameter to the `load_inference_model_from_checkpoint` function. For example, for ANI2x models, where this should be True (other currently implemented potentials require False):

.. code-block:: python

    from modelforge.potential.models import load_inference_model_from_checkpoint

    inference_model = load_inference_model_from_checkpoint(checkpoint_file, only_unique_pairs=True)


To modify state dictionary files, this can be done easily via the `modify_state_dict` function in the file `modify_state_dict.py` in the scripts directory.  This will generate a new copy of the state dictionary file with the appropriate `only_unique_pairs` parameter set.

Loading a checkpoint from weights and biases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checkpoint files can be loaded directly from wandb using the `load_from_wandb` function as part of the `NeuralNetworkPotentialFactory`.  This can be done by passing the wandb run id and appropriate version number.  Note this will require authentication with wandb for users part of the project.  The following code snippet demonstrates how to load a model from wandb.

.. code-block:: python

    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    nn_potential = NeuralNetworkPotentialFactory().load_from_wandb(
        run_path="modelforge_nnps/test_ANI2x_on_dataset/model-qloqn6gk",
        version="v0",
        local_cache_dir=f"{prep_temp_dir}/test_wandb",
    )



Using a model for inference in OpenMM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the trained model for inference in OpenMM, we can save the Potential model to a .pt file the use the wrapping class to put this in a form that will work with the OpenMM TorchForce. The `save` method of the Potential class can be used to save the model to a .pt file.  The follow code snippet demonstrates loading a configuration file, generating a potential, converting it to a torchscript model, and saving it to a .pt file.

.. code-block:: python

    from modelforge.utils.misc import load_configs_into_pydantic_models

    config = load_configs_into_pydantic_models("ani2x", "phalkethoh")

    # generate the ani2x potential using the NeuralNetworkPotentialFactory
    # note this is not a trained potential

    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        potential_seed=42,
        jit=False,
    )

    import torch

    # convert the potential to a torchscript model and save it
    jit_potential = torch.jit.script(potential)
    jit_potential.save("data/ani2x_test.pt")

With the .pt file saved, we can load up the wrapper that will allow us to set TorchForce in OpenMM to use the model for inference.  The following code snippet demonstrates how to load the .pt file and use the Potential wrapper class in `modelforge.openmm.potential`.  Note, to use this function, you'll need to install the modelforce.openmm subpackage from within the root of the modelforge repository (e.g., `pip install modelforge_openmm .`).

.. code-block:: python

    from modelforge.openmm.examples.openmm_water_topology import openmm_water_topology

    # set up the water topology using a helper function we created
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
        neighborlist_strategy="verlet_nsq",
        energy_contributions=["per_system_energy"],
    )

    ani2x_jit_potential = torch.jit.script(ani2x_potential)


We can then use this to set the TorchForce and set up a simple simulation.

.. code-block:: python

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
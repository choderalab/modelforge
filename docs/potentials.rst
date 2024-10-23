Potentials
===============

Potentials: Overview
------------------------

A potential in *modelforge* encapsulates all the operations required to map a
description of a molecular system (which includes the Cartesian coordinates,
atomic numbers, and total charge of a system (optionally also the spin state))
to, at a minimum, its energy. Specifically, a potential takes as input a
:py:class:`~modelforge.dataset.dataset.NNPInput` dataclass and outputs a
dictionary of PyTorch tensors representing per-atom and/or per-molecule
properties, including per-molecule energies. A potential comprises three main
components:

1. **Input Preparation Module**
   (:class:`~modelforge.potential.model.InputPreparation`): Responsible for
   generating the pair list, pair distances, and pair displacement vectors based
   on atomic coordinates and the specified cutoff. This module processes the raw
   input data into a format suitable for the neural network.

2. **Core Model** (:class:`~modelforge.potential.model.CoreNetwork`): The neural
   network containing the learnable parameters, which forms the core of the
   potential. This module generates per-atom scalar and, optionally, tensor properties. The inputs to the core model
   include atom pair indices, pair distances, pair displacement vectors, and atomic properties such as atomic numbers, charges, and spin state.

3. **Postprocessing Module**
   (:class:`~modelforge.potential.model.PostProcessing`): Contains operations
   applied to per-atom properties as well as reduction operations to obtain
   per-molecule properties. Examples include atomic energy scaling and summation of per-atom energies to obtain per-molecule energies for reduction operations.

A specific neural network (e.g., PhysNet) implements the core model, while the
input preparation and postprocessing modules are independent of the neural
network architecture.


Implemented Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Modelforge* currently supports the following potentials:

- Invariant architectures:
   * SchNet
   * ANI2x
- Equivariant architectures:
   * PaiNN
   * PhysNet
   * TensorNet
   * SAKE

Additionally, the following models are currently under development and can be expected in the near future:

- SpookyNet
- DimeNet
- AimNet2

Each potential currently implements the total energy prediction with per-atom
forces within a given cutoff radius. The models can be trained on energies and
forces. PaiNN and PhysNet can also predict partial charges and
calculate long-range interactions using reaction fields or Coulomb potential.
PaiNN can additionally use multipole expansions.

Using TOML files to configure potentials
--------------------------------------------

To initialize a potential, a potential factory is used:
:class:`~modelforge.potential.potential.NeuralNetworkPotentialFactory`.
This takes care of initialization of the potential and the input preparation and postprocessing modules.

A neural network potential is defined by a configuration file in the TOML format.
This configuration file includes parameters for the neural network, as well as
for the input preparation and postprocessing modules. Below is an example
configuration file for the PhysNet model:

.. literalinclude:: ../modelforge/tests/data/potential_defaults/physnet.toml
   :language: toml
   :caption: PhysNet Configuration

There are two main sections in the configuration file: `core_parameter` and
`postprocessing_parameter`. The `core_parameter` section contains the parameters
for the neural network, while the `postprocessing_parameter` section contains
the parameters for the postprocessing operations. Explanation of fields in
`physnet.toml`:

* `potential_name`: Specifies the type of potential to use, in this case, PhysNet.
* `number_of_radial_basis_functions`: Number of radial basis functions.
* `maximum_interaction_radius`: Cutoff radius for considering neighboring atoms.
* `number_of_interaction_residual`: PhysNet hyperparamter defining the depth of the network.
* `number_of_modules`: PhysNet hyperparamter defining the depth of the network;which scales with (number_of_interaction_residual * number_of_modules).
* `activation_function_name`: Activation function used in the neural network.
* `properties_to_featurize`: List of properties to featurize.
* `maximum_atomic_number```: Maximum atomic number in the dataset.
* `number_of_per_atom_features`: Number of features for each atom used for the embedding. This is the number of features that are used to represent each atom in the neural network.
* `normalize`: Whether to normalize energies for training. If this is set to true the mean and standard deviation of the energies are calculated and used to normalize the energies.
* `from_atom_to_system_reduction`: Whether to reduce the per-atom properties to per-molecule properties.
* `keep_per_atom_property`: If this is set to true the per-atom energies are returned as well.


Default parameter files for each potential are available in `modelforge/tests/data/potential_defaults`. These files can be used as starting points for creating new potential configuration files.

.. note:: All parameters in the configuration files have units attached where applicable. Units within modelforge a represented using the `openff.units` package (https://docs.openforcefield.org/projects/units/en/stable/index.html), which is a wrapper around the `pint` package. Definition of units within the TOML files must unit names available in the `openff.units` package (https://github.com/openforcefield/openff-units/blob/main/openff/units/data/defaults.txt).

Use cases of the factory class
--------------------------------------------

There are four main use cases of the :class:`~modelforge.potential.potential.NeuralNetworkPotentialFactory`:
1. Create and train a model, then save the state_dict of its potential. Load the state_dict to the potential of an existing trainer (with defined hyperparameters) to resume training.
2. Load a potential for inference from a saved state_dict.
3. Load a trainer from a checkpoint file
4. Load an inference from a checkpoint file

.. note:: Checkpoint files in modelforge are saved by :class:`~modelforge.train.training.PotentialTrainer`, thus will always be a trainer object. Therefore loading a inference mode potential from a checkpoint file requires load the trainer, then extract the potential from the trainer.

The general idea to handle these use cases is that always call :function:`~modelforge.potential.potential.NeuralNetworkPotentialFactory.generate_trainer()` to create or load a trainer, and `~modelforge.potential.potential.NeuralNetworkPotentialFactory.generate_potential()` for inference.

.. code-block:: python
    :linenos:

    # Use case 1
    trainer = NeuralNetworkPotentialFactory.generate_trainer(
            potential_parameter=config["potential"],
            training_parameter=config["training"],
            runtime_parameter=config["runtime"],
            dataset_parameter=config["dataset"],
        )
    torch.save(trainer.lightning_module.state_dict(), file_path)
    trainer2 = NeuralNetworkPotentialFactory.generate_trainer(
            potential_parameter=config2["potential"],
            training_parameter=config2["training"],
            runtime_parameter=config2["runtime"],
            dataset_parameter=config2["dataset"],
        )
    trainer2.lightning_module.load_state_dict(torch.load(file_path))

    # Use case 2
    potential = NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment="PyTorch",
        potential_parameter=config["potential"],
    )
    potential.load_state_dict(torch.load(file_path))

    # Use case 3

    # Use case 4
    from modelforge.potential.potential import load_inference_model_from_checkpoint
    potential = load_inference_model_from_checkpoint(chkp_file)
    assert potential is not None

Example
------------------------------------

Below is an example of how to create a SchNet model using the potential factory, although we note these operations do not typically need to be performed directly by a user and are handled by routines available in the training module:

.. code-block:: python
   :linenos:

   model_name = "SchNet"

   # reading default parameters
   from modelforge.tests.data import potential_defaults
   from importlib import 
   import toml

   filename = (
      resources.files(potential_defaults) / f"{model_name.lower()}_defaults.toml"
   )
   potential_parameters = toml.load(filename)

   # initialize the models with the given parameter set
   model = NeuralNetworkPotentialFactory.generate_potential(
      use="inference",
      model_type=model_name,
      model_parameter=potential_parameters,
   )
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
- AimNet2 (available, but certain features still under development)

Each potential currently implements the total energy prediction with per-atom
forces within a given cutoff radius. The models can be trained on energies and forces. PaiNN and PhysNet can also predict partial charges and
calculate long-range interactions using a Coulomb potential.
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

There are two main sections in the configuration file: `core_parameter`,  and
`postprocessing_parameter`. The `core_parameter` section contains the core parameters
for the neural network, while the `postprocessing_parameter` section contains
the parameters for the postprocessing operations. Explanation of fields in
`physnet.toml`:

`core_parameter`:

* `potential_name`: Specifies the type of potential to use, in this case, PhysNet.
* `number_of_radial_basis_functions`: Number of radial basis functions.
* `maximum_interaction_radius`: Cutoff radius for considering neighboring atoms.
* `number_of_interaction_residual`: PhysNet hyperparamter defining the depth of the network.
* `number_of_modules`: PhysNet hyperparamter defining the depth of the network;which scales with (number_of_interaction_residual * number_of_modules).
* `featurization.properties_to_featurize`: List of properties to featurize. Currently, must always include `atomic_number`. Other properties can be added to this list.
* `featurization.atomic_number.maximum_atomic_number```: Maximum atomic number in the dataset.
* `featurization.atomic_number.number_of_per_atom_features`: Number of features for each atom used for the embedding. This is the number of features that are used to represent each atom in the neural network.
* `activation_function_parameter.activation_function_name`: Activation function used in the neural network.

`postprocessing_parameter`:
* `properties_to_process`: List of properties to process. Currently, must always include `per_atom_energy`. Other properties can be added to this list, so as 'per_atom_charge', 'electrostatic_potential' and 'zbl_potential'.
* `per_atom_energy.normalize`: Whether to normalize energies for training. If this is set to true the mean and standard deviation of the energies are calculated and used to normalize the energies.
* `per_atom_energy.from_atom_to_system_reduction`: Whether to reduce the per-atom properties to per-molecule properties.
* `per_atom_energy.keep_per_atom_property`: If this is set to true the per-atom energies are returned as well.
* `per_atom_charge.conserve`: Whether to conserve the total charge of the system if charges are predicted (i.e. 'per_atom_charge' is in the 'properties_to_process'. If this is set to true the total charge of the system is calculated and used to normalize the per-atom charges.
*`per_atom_charge.conserve_strategy`: The strategy used to conserve the total charge of the system. Currently, only 'default' is implemented, which uses the default strategy.
* `electrostatic_potential.electrostatic_strategy`: The strategy used to calculate the electrostatic potential. Currently, only 'coulomb' is implemented, which uses the Coulomb potential. Note, to use this option the `per_atom_charge` property must be included in the list of properties to process (as this relies upon the partial charges).
* `electrostatic_potential.maximum_interaction_radius`: Cutoff radius for considering neighboring atoms for the electrostatic potential. Note: this may be different than the cutoff used for the core model.
* `zbl_potential.calculate_zbl_potential`: This is a postprocessing operation that calculates the Ziegler-Biersack-Littmark (ZBL) potential. This is a very short-range potential that is used to prevent overlap of atoms.

Default parameter files for each potential are available in `modelforge/tests/data/potential_defaults`. These files can be used as starting points for creating new potential configuration files.

Example of how the postprocessing section in a .toml file that that calculates the coulomb potential and the ZBL potential:
.. code-block:: toml

    [potential.postprocessing_parameter]
    properties_to_process = ['per_atom_energy', 'per_atom_charge', 'electrostatic_potential', 'zbl_potential']

    [potential.postprocessing_parameter.per_atom_energy]
    normalize = true
    from_atom_to_system_reduction = true
    keep_per_atom_property = true

    [potential.postprocessing_parameter.per_atom_charge]
    conserve = true
    conserve_strategy= "default"

    [potential.postprocessing_parameter.electrostatic_potential]
    electrostatic_strategy = "coulomb"
    maximum_interaction_radius = "10.0 angstrom"

    [potential.postprocessing_parameter.zbl_potential]
    calculate_zbl_potential = true



.. note:: All parameters in the configuration files have units attached where applicable. Units within modelforge a represented using the `openff.units` package (https://docs.openforcefield.org/projects/units/en/stable/index.html), which is a wrapper around the `pint` package. Definition of units within the TOML files must unit names available in the `openff.units` package (https://github.com/openforcefield/openff-units/blob/main/openff/units/data/defaults.txt).


Featurization options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Currently, modelforge requires `atomic_number` to be included in the list of properties to featurize.  Additional properties can be included in the list of properties to featurize, including: `atomic_group`, `atomic_period`, `per_system_total_charge`, and `per_system_spin_state`.  The `atomic_group` and `atomic_period` properties are derived from the atomic number and are used to provide additional information to the neural network.  Similar to `atomic_number` we can provide additional options for `atomic_group` and `atomic_period` that will specify the underlying tensor shape. The snippet below shows the syntax for defining these in the toml file:

.. code-block:: toml

    [potential.core_parameter.featurization]
    properties_to_featurize = ['atomic_number', 'atomic_group', 'atomic_period']

    [potential.core_parameter.featurization.atomic_number]
    maximum_atomic_number = 101
    number_of_per_atom_features = 32

    [potential.core_parameter.featurization.atomic_group]
    maximum_group_number = 18
    number_of_per_group_features = 32

    [potential.core_parameter.featurization.atomic_period]
    maximum_period_number = 8
    number_of_per_period_features = 32



The `per_system_total_charge` property is the total charge of the system and `per_system_spin_state` represents the spin state of the system; these properties does not have any additional parameters that need to be defined in the toml file, as this simply adds an additional feature to the input tensor. Note, to use these in a meaningful the dataset must include data to set for the properties  `total_charge` and `S`, respectively.  These values, if not provided, will be initialized to 0; as such without this information from the datafile, the code will run, these will not necessarily provide any meaningful information for the embedding.




Use cases of the factory class
--------------------------------------------

There are three main use cases of the :class:`~modelforge.potential.potential.NeuralNetworkPotentialFactory`:

1. Create and train a model, then save the state_dict of its potential. Load the state_dict to the potential of an existing trainer (with defined hyperparameters) to resume training.

2. Load a potential for inference from a saved state_dict.

3. Load an inference from a checkpoint file.

.. note:: The general idea to handle these use cases is that always call `generate_trainer()` to create or load a trainer; use `generate_potential()` for loading inference potential (this is also how `load_inference_model_from_checkpoint()` is implemented).

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
    from modelforge.potential.potential import load_inference_model_from_checkpoint
    potential = load_inference_model_from_checkpoint(ckpt_file)

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
Potentials
===============

Potentials: Overview
------------------------

A potential in *modelforge* encapsulates all the operations required to map a
description of a molecular system (which includes the Cartesian coordinates and
atomic numbers (and optionally total system charge and system spin multiplicity)
to, at a minimum, its energy. Specifically, a potential takes as input a
:py:class:`~modelforge.dataset.dataset.NNPInput` dataclass and outputs a
dictionary of PyTorch tensors representing per-atom and/or per-system
properties, including per-system energies.
Note, "per-system" can be considered synonymous with "per-molecule"
(i.e., if the system contains a single molecule, the per-system energy
is the summation of the individual per-atom energies);
the use of "per-system" is to provide a more general terminology,
as a given system in a dataset may contain multiple molecules
(e.g., a cluster of water molecules, or a protein-ligand complex) and modelforge does not differentiate between the
case of single vs. multiple molecules in a system, as we do not consider fixed bonding topologies.

A potential comprises three main
components:

1. **Neighborlist**: Responsible for generating the list of interacting pairs, the pair distances, and pair displacement vectors based on atomic coordinates and the specified cutoff.

2. **Core Model**: The core model is the neural
   network containing the learnable parameters, which forms the core of the
   potential. This module generates per-atom scalar and, optionally, tensor properties. The inputs to the core model
   include atom pair indices, pair distances, pair displacement vectors, and atomic properties such as atomic numbers, charges, and spin multiplicities.

3. **Postprocessing Module**
   (:class:`~modelforge.potential.potential.PostProcessing`): Contains operations
   applied to per-atom properties as well as reduction operations to obtain
   per-molecule properties. Examples include atomic energy scaling and summation of per-atom energies to obtain per-molecule energies for reduction operations, Coulombic interactions, ZBL potential, and DFTD3 dispersion correction.

A specific neural network (e.g., PhysNet) implements the core model, while the
input preparation and postprocessing modules are independent of the neural
network architecture.  While the underlying code may differ, each potential is designed with a similar structure with 3 classes corresponding to the Core, Representation, and Interaction modules.


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
   * AimNet2 (available, but certain features still under development)

Additionally, the following models are currently under development and can be expected in the near future:

- SpookyNet
- DimeNet

Each potential currently implements the total energy prediction with per-atom
forces within a given cutoff radius. The models can be trained on energies and forces. PaiNN and PhysNet can also predict partial charges and
calculate long-range interactions using a Coulomb potential.
PaiNN can additionally use multipole expansions.

Using TOML files to configure potentials
--------------------------------------------

To initialize a potential, a factory is used:
:class:`~modelforge.potential.potential.NeuralNetworkPotentialFactory`.
This takes care of initialization of the potential and the input preparation and postprocessing modules.

A neural network potential is defined by a configuration file(s) in the TOML format.
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

* `potential_name`: Specifies the type of potential to use, in this case, PhysNet.

`core_parameter`:

* `number_of_radial_basis_functions`: Number of radial basis functions.
* `maximum_interaction_radius`: Cutoff radius for considering neighboring atoms.
* `number_of_interaction_residual`: PhysNet hyperparamter defining the depth of the network.
* `number_of_modules`: PhysNet hyperparamter defining the depth of the network;which scales with (number_of_interaction_residual * number_of_modules).
* `predicted_properties`: List of properties to predict from the network. Currently, must always include `per_atom_energy`. Other properties can be aded to this list, such as `per_atom_charge`. While this can, in theory, can accept any property, these predicted properties may not be meaningful if they are not able to be assessed by the loss function.
* `predicted_dim`: List of integers that correspond to the length of each predicted property.  For example, if `predicted_properties` is `['per_atom_energy', 'per_atom_charge']`, then `predicted_dim` should be `[1, 1]` for a single scalar value for each property.
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
* `per_atom_charge.conserve_strategy`: The strategy used to conserve the total charge of the system. Currently, only 'default' is implemented, which uses the default strategy.
* `electrostatic_potential.electrostatic_strategy`: The strategy used to calculate the electrostatic potential. Currently, only 'coulomb' is implemented, which uses the Coulomb potential. Note, to use this option the `per_atom_charge` property must be included in the list of properties to process (as this relies upon the partial charges).
* `electrostatic_potential.maximum_interaction_radius`: Cutoff radius for considering neighboring atoms for the electrostatic potential. Note: this may be different than the cutoff used for the core model.
* `zbl_potential.calculate_zbl_potential`: This is a postprocessing operation that calculates the Ziegler-Biersack-Littmark (ZBL) potential. This is a very short-range potential that is used to prevent overlap of atoms.

Default parameter files for each potential are available in `modelforge/tests/data/potential_defaults`. These files can be used as starting points for creating new potential configuration files.

The following is an example of how the postprocessing section in a .toml file that that:
* calculates the `per_system_energy` from the `per_atom_energy`
* processes the `per_atom_charge` to ensure the total charge of the system is conserved
* calculate the `per_system_electrostatic_energy` using the coulomb potential from the `per_atom_charge`
* calculates the `per_system_zbl_energy` using the ZBL potential
* calculates the `per_system_vdw_energy` using the DFTD3 potential
* sums the `per_system_electrostatic_energy`, `per_system_zbl_energy`, and `per_system_vdw_energy` into the `per_system_energy` using the `sum_per_system_energies` operation.

Note, the order these are defined or listed in `properties_to_process` does not impact the order they are processed.

.. code-block:: toml

    [potential.postprocessing_parameter]
    properties_to_process = ['per_atom_energy', 'per_atom_charge', 'per_system_electrostatic_energy', 'per_system_zbl_energy', 'per_system_vdw_energy', 'sum_per_system_energies']

    [potential.postprocessing_parameter.per_atom_energy]
    normalize = true
    from_atom_to_system_reduction = true
    keep_per_atom_property = true

    [potential.postprocessing_parameter.per_atom_charge]
    conserve = true
    conserve_strategy= "default"

    [potential.postprocessing_parameter.per_system_electrostatic_energy]
    electrostatic_strategy = "coulomb"
    maximum_interaction_radius = "15.0 angstrom"

    [potential.postprocessing_parameter.per_system_zbl_energy]

    [potential.postprocessing_parameter.per_system_vdw_energy]
    maximum_interaction_radius = "10.0 angstrom"

    # this will be added to the `per_system_energy` that results from the `per_atom_energy` reduction operation
    [potential.postprocessing_parameter.sum_per_system_energies]
    contributions = ['per_system_electrostatic_energy', 'per_system_zbl_energy', 'per_system_vdw_energy']


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



The `per_system_total_charge` property is the total charge of the system and `per_system_spin_state` is the spin state of the system; these properties do not have any additional parameters that need to be defined in the toml file, as their inclusion simply adds an additional feature to the input tensor. Note, to use these in a meaningful way, the dataset should include these properties (and assign them in the .toml file), as these values will be initialized to 0 if not assigned from the datafile. This means the code would run, but again, all values would be zero which adds no information to the model.




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


AimNet2: how to define the postprocessing operations
------------------------------------------------------

AimNet2 is unique in that it is devised such that the energy that is predicted from the potential consists of the 3 contributions: the local energy (E_local), the van der Waals energy (E_vdw), and the electrostatic energy (E_electrostatic).  This is done to help address the nearsightedness typical of NNPs.

Rather than having all 3 of these energy contributions returned by the core network, only E_local is returned by the core network; this fits with the modular design of modelforge.  To include the van der Waals and electrostatic contributions, these should be calculated via the post processing module. The following code implements the post processing operations required to include these additional terms.  Note, the charge conservation operation does not need to be included as a post processing operation, as this is performed within the core model.


.. code-block:: toml

    [potential.postprocessing_parameter]
    properties_to_process = ['per_atom_energy', 'per_system_electrostatic_energy', 'per_system_vdw_energy', 'sum_per_system_energies']

    [potential.postprocessing_parameter.per_atom_energy]
    normalize = true
    from_atom_to_system_reduction = true
    keep_per_atom_property = true


    [potential.postprocessing_parameter.per_system_electrostatic_energy]
    electrostatic_strategy = "coulomb"
    maximum_interaction_radius = "15.0 angstrom"


    [potential.postprocessing_parameter.per_system_vdw_energy]
    maximum_interaction_radius = "10.0 angstrom"

    # this will be added to the per_system_energy that results from the `per_atom_energy` reduction operation
    [potential.postprocessing_parameter.sum_per_system_energies]
    contributions = ['per_system_electrostatic_energy', 'per_system_zbl_energy', 'per_system_vdw_energy']
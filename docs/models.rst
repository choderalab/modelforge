Models
===============

Introduction
----------------

A model in modelforge encapsulates all the operations required to map the Cartesian coordinates, atomic numbers, and total charge of a system to, at a minimum, its energy. Specifically, a model takes as input a :py:class:modelforge.dataset.dataset.NNPInput dataclass and outputs a dictionary of PyTorch tensors representing per-atom and/or per-molecule properties, including per-molecule energies. The model comprises three main components:

1. **Input Preparation Module** (:class:modelforge.potential.model.InputPreparation): Responsible for generating the pair list, pair distances, and pair displacement vectors based on atomic coordinates and the specified cutoff.

2. **Core Model** (:class:modelforge.potential.model.CoreNetwork): The neural network containing the learnable parameters, which forms the core of the potential.

3. **Postprocessing Module** (:class:modelforge.potential.model.PostProcessing): Contains operations applied to both per-atom and per-molecule outputs, as well as reduction operations to obtain per-molecule properties. Examples include atomic energy scaling or charge equilibration for per-atom operations, and summation of per-atom energies to obtain per-molecule energies for reduction operations.

A specific neural network (e.g., PhysNet) implements the core model, while the input preparation and postprocessing modules are independent of the neural network architecture.

To initialize a model, the model factory is used: :class:`modelforge.potential.models.NeuralNetworkPotentialFactory`.

.. autoclass:: modelforge.potential.models.NeuralNetworkPotentialFactory
      :members:

Using toml files to configure models
------------------------------------

A neural network model is defined by a configuration file in the TOML format. This configuration file includes parameters for the neural network, as well as for the input preparation and postprocessing modules. Below is an example configuration file for the PhysNet model:

.. literalinclude:: ../modelforge/tests/data/potential_defaults/physnet.toml
   :language: toml
   :caption: PhysNet Configuration

Explanation of fields in `physnet.toml`:

* `model_name`: Specifies the type of model to use, in this case, SchNet.
* `max_Z`: Maximum atomic number.
* `number_of_atom_features`: Number of features for each atom.
* `number_of_radial_basis_functions`: Number of radial basis functions.
* `cutoff`: Cutoff radius for considering neighboring atoms.
* `number_of_interaction_modules`: Number of interaction modules.
* `number_of_filters`: Number of filters.
* `shared_interactions`: Whether interaction modules share weights.


There are default parameter files for each model in `modelforge/tests/data/potential_defaults`. These files can be used as a starting point for creating a new model configuration file. Note that all parameters in the configuration file have units attached where applicable.

The definition of the `processing_operation` and `readout_operation` are explained in the next section.

Postprocessing of the model output
------------------------------------

There are two types of postprocessing operations that can be performed on the model output: processing operations and readout operations. Processing operations are applied to the per-atom outputs, while readout operations are applied to the per-molecule outputs.

The way these operations are defined in the configuration file is as follows:

.. code-block:: python

   processing_operation = [
      { in = [
         "E_i",
      ], out = "E_i", function = "normalization", mean = "E_i_mean", stddev = "E_i_mean" }
   ]

The `in` keyword defines the property the operation is applied to, while the `out` keyword defines the name of the new property. The `function` keyword defines the name of the function that is applied to the input (possible values can be displayed calling `modelforge.potential.processing._REGISTER_PROCESSING_OPERATIONS`). The `mean` and `stddev` keywords are additional paramter necessary for the :class:`modelforge.potential.processing.ScaleValues` function (which is called when the `function` is set to `normalization`).

The readout operations are applied to the per-atom properties to generate per-molecule property. The readout operations are defined as follows:

.. code-block:: python

   readout_operation = [
      { function = "from_atom_to_molecule", mode = 'sum', in = 'E_i', index_key = 'atomic_subsystem_indices', out = 'E' },
      { function = "from_atom_to_molecule", mode = 'sum', in = 'ase', index_key = 'atomic_subsystem_indices', out = 'mse' },
   ]

The `function` keyword defines the name of the function that is applied to the input. The `mode` keyword defines the reduction operation to perform on the per-atom property. The `in` keyword defines the property the operation is applied to, while the `out` keyword defines the name of the new property. The `index_key` keyword defines the name of the property that contains the indices of the atoms that are part of the molecule.

To get an overview of the different readout operations you can displayed  `modelforge.potential.processing._REGISTER_READOUT_OPERATIONS`.

Example
------------------------------------

Below is an example of how to create a SchNet model using the model factory:

.. code-block:: python
   :linenos:

   model_name = "SchNet"

   # reading default parameters
   from modelforge.train.training import return_toml_config
   from modelforge.tests.data import potential_defaults
   from importlib import resources

   filename = (
      resources.files(potential_defaults) / f"{model_name.lower()}_defaults.toml"
   )
   config = return_toml_config(filename)

   # Extract parameters
   potential_parameters = config["potential"].get("potential_parameters", {})

   # initialize the models with the given parameter set
   model = NeuralNetworkPotentialFactory.create_nnp(
      use="inference",
      model_type=model_name,
      model_parameters=potential_parameters,
   )
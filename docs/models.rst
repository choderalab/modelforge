Models
===============

Introduction
----------------

A model contains all operations needed to map (in `inference` mode) or learn the mapping (in `training` mode) the cartesian coordinates, atomic numbers and charge of a system to an energy (and, depending on the model, other properties). Models are typically trained on quantum chemistry data, which are provided by `modelforge` (see :doc:datasets for more information)

A model takes as input a :py:class:`modelforge.dataset.dataset.ModelInput` dataclass, which includes information about the system, such as the atomic numbers, the cartesian coordinates and the charge. 
The model returns a dictionary, which includes the predicted energy and, depending on the model, other properties.

- the input preparation module (:class:`modelforge.potential.model.InputPreparation`): responsible for generating the pairlist, pair distances and pair displacement vectors given the atomic coordinates and the cutoff.
- the core model (:class:`modelforge.potential.model.CoreNetwork`): the neural network containing learnable parameters (what is considered the "core" of the potential).

- the postprocessing module(:class:`modelforge.potential.model.PostProcessing`): contains operations that are applied to per-atom and per-molecule outputs, as well as the reduction operations that are performed to obtain the per-molecule properties. Examples of per-atom operations are atomic energy scaling or charge equilibration, examples for reduction operations are summation of per-atom energies to obtain per-molecule energies.

A given neural network (e.g., SchNet) implements the core model, the input prepartion and postprocessing modules are independent of the neural network. This modularity allows the user to use any neural network for the core model.

The models that are currently implemented in `modelforge` are:
- SchNet (see :class:`modelforge.potential.schnet.SchNet`): a invariant, second generation neural network potential <https://arxiv.org/abs/1706.08566>.

- PaiNN (see :class:`modelforge.potential.painn.PaiNN`): a direct improvement on SchNet, using equivariant features <https://arxiv.org/abs/2102.03150>.

- ANI2x (see :class:`modelforge.potential.ani.ANI2x`): a invariant, second generation neural network potential <https://doi.org/10.26434/chemrxiv.11819268.v1>.

- PhysNet (see :class:`modelforge.potential.physnet.PhysNet`): an invariant, third generation neural network potential <https://doi.org/10.1021/acs.jctc.9b00181.

- SAKE (see :class:`modelforge.potential.sake.SAKE`): a second generation equivariant neural network potential <https://arxiv.org/abs/2301.08893>.

- TensorNet (see :class:`modelforge.potential.tensornet.TensorNet`): a fast and scalable equivariant neural network potential <https://arxiv.org/abs/2306.06482>.


How to use a model
------------------------------------

We are planning to provide trained models in the future, but we are still working on the best way to do so.

In the meantime the way to use a model is to first train it. The workflow to do so is outlined in the :doc:`training` section. Once a model is trained, it can be used to predict energies for new systems. A trained model can save its configuration and weights in a directory, which can be used to initialize a model.

```python
model.save_state_dict('model')
```
To initialize a model, the model factory is used: :class:`modelforge.potential.models.NeuralNetworkPotentialFactory`. 



.. autoclass:: modelforge.potential.models.NeuralNetworkPotentialFactory
      :members:

Using toml files to configure models
------------------------------------

A neural network model is defined by a configuration file, which is a TOML file. The configuration file contains the parameters for the neural network, as well as the parameters for the input preparation and postprocessing modules.
Below is an example of a configuration file for the SchNet model:

.. literalinclude:: ../modelforge/tests/data/potential_defaults/schnet.toml
   :language: toml
   :caption: SchNet Configuration

Explanation of fields in `schnet.toml`:

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

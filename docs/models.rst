Models
===============

Introduction
----------------

A model is defined as a PyTorch model that takes as input a :py:class:`modelforge.dataset.dataset.NNPInput` dataclass, and returns a dictionary of PyTorch tensors. The model consists of three, functional compounds:
- the input preparation module (:class:`modelforge.potential.model.InputPreparation`): responsible for generating the pairlist, pair distances and pair displacement vectors given the atomic coordinates and the cutoff.
- the core model (:class:`modelforge.potential.model.CoreNetwork`): the neural network containing learnable parameters (what is considered the "core" of the potential).
- the postprocessing module: contains operations that are applied to per-atom and per-molecule outputs, as well as the reduction operations that are performed to obtain the per-molecule properties.

A given neural network (e.g., SchNet) implements the core model, the input prepartion and postprocessing modules are independent of the neural network. 

To initialize a model, the model factory is used: :class:`modelforge.potential.models.NeuralNetworkPotentialFactory`.

.. autoclass:: modelforge.potential.models.NeuralNetworkPotentialFactory
      :members:

Using toml files to configure models
------------------------------------

A neural network model is defined by a configuration file, which is a TOML file. The configuration file contains the parameters for the neural network, as well as the parameters for the input preparation and postprocessing modules.
Below is an example of a configuration file for the SchNet model:

.. literalinclude:: ../scripts/configs/potentials/schnet.toml
   :language: toml
   :caption: SchNet Configuration

Explanation of fields in `schnet.toml`:

* `model_name`: Specifies the type of model to use, in this case, SchNet.
* `potential_parameter.max_Z`: Maximum atomic number.
* `potential_parameter.number_of_atom_features`: Number of features for each atom.
* `potential_parameter.number_of_radial_basis_functions`: Number of radial basis functions.
* `potential_parameter.cutoff`: Cutoff radius for considering neighboring atoms.
* `potential_parameter.number_of_interaction_modules`: Number of interaction modules.
* `potential_parameter.number_of_filters`: Number of filters.
* `potential_parameter.shared_interactions`: Whether interaction modules share weights.


There are default parameter files for each model in `modelforge/scripts/configs/potentials`. These files can be used as a starting point for creating a new model configuration file. Note that all parameters in the configuration file have units attached where applicable.

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

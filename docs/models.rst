Models
===============

SchNet Configuration (schnet.toml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

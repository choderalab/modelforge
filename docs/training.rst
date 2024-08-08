Training
========

Introduction
------------------------------------------

During training the parameters of a model are fitted to reproduce the target properties provided by a dataset. These are typically energies and forces, but can also be other properties provided by a dataset (e.g., dipole moments).

The properties a given model can be trained on are deterimend by the model itself and by the loss function used to train the model. Each of the models implemented in *modelforge* have flexible numbers of output heads, each of which can be fitted against a different scalar/tensor property. 

*Modelforge* uses Pytorch Lightning to train models. The training process is controlled by a :class:`~modelforge.train.training.ModelTrainer` object, which is responsible for managing the training loop, the optimizer, the learning rate scheduler, and the early stopping criteria. The training process is controlled by a configuration file, `training.toml`, which specifies the number of epochs, the learning rate, the loss function, and the splitting strategy for the dataset. The training process can be started by 

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The training process is controlled by a configuration file. The configuration file contains sections for the logger, the learning rate scheduler, the loss functions, splitting strategy and early stopping criteria.

.. literalinclude:: ../modelforge/tests/data/training_defaults/default.toml
   :language: toml
   :caption: Training Configuration


Learning rate scheduler
^^^^^^^^^^^^^^^^^^^^^^^^

The learning rate scheduler is responsible for adjusting the learning rate during training. *Modelforge* uses the REduceLROnPlateau scheduler, which reduces the learning rate by a factor of 0.1 when the RMSE of the energy prediction on the validation set does not improve for a given number of epochs. The scheduler is controlled by the parameters in the `[training.lr_scheduler]` section of the `training.toml` file.

Loss function
^^^^^^^^^^^^^^^^^^^^^^^^
The loss function calculate the difference between the model's predictions and the target properties. The loss function is responsible for providing a scalar value that the optimizer can use to update the model's parameters. The loss function is controlled by the parameters in the `[training.loss]` section of the `training.toml` file.

Depending on what is included in the `loss_property` section the loss function can be a combination of different loss functions. *Modelforge* always computes the mean squared loss for the energy prediction, but can also include the mean squared loss for the force prediction, the mean squared loss for the dipole moment prediction, and the mean squared loss for the partial charge prediction.

The formulation of the loss function is closely linked to the form of the energy function. For example, if the energy function is a sum of atomic energies, `loss_property` should only include `per_molecule_energy` and, optionally, `per_atom_force`. 

Predict short-range atomic energies
************************************************************

If the total atomic energy is calculated with a short cutoff radius, we can directly match the sum of the atomic energies (as predicted by the model architecture) to the total energy of the molecule (provided by the dataset).

In that case the total energy is calculated as

.. math:: E = \sum_i^N E_i

and the loss is either

.. math:: L = w_E * (E - E^{pred})^2

or, if the dataset provides per atom forces, we can formulate the loss as

.. math:: L = w_E * (E - E^{pred})^2 + w_F \frac{1}{3N} \sum_i^N \sum_j^3 (F_{ij} - F_{ij}^{pred})^2

Predict short range atomic energy with long range electrostatics
*************************************************************************

In that case additional terms are added to the loss function to account for the
long range electrostatics. Instead of directly predicting the per atom
contribution of the atomic energy, the model predicts atomic charges. The atomic
charges are then used to calculate the long range electrostatics. The expression
for the total energy is then:




Splitting Strategies
^^^^^^^^^^^^^^^^^^^^^^^^

The default splitting strategy for datasets in `modelforge` is to randomly split
the dataset into 80% training, 10% validation, and 10% test set based on
molecules. This approach ensures that different conformations of a molecule are
always part of the same split, thereby avoiding data leakage. 

`modelforge` also provides other splitting strategies, including:

- :class:`modelforge.dataset.utils.FirstComeFirstServeStrategy`: Splits the dataset based on the order of records (molecules).
- :class:`modelforge.dataset.utils.RandomSplittingStrategy`: Splits the dataset randomly based on conformations.

To use a different data split ratio, you can specify a custom split list in the splitting strategy. 
The most effective way to pass this information to the training process is by defining the appropriate fields in the `dataset.toml` file, as shown in :ref:`dataset-configuration`.



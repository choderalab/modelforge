Training
========

Training: Overview
------------------------------------------

During training the parameters of a potential are fitted to reproduce the target properties provided by a dataset. These are typically energies and forces, but can also be other properties provided by a dataset (e.g., dipole moments).

The properties a given potential can be trained on are determined by the potential itself and by the loss function used to train the potential. Each of the models implemented in *modelforge* have flexible numbers of output heads, each of which can be fitted against a different scalar/tensor property.

*Modelforge* uses Pytorch Lightning to train models. The training process is controlled by a :class:`~modelforge.train.training.PotentialTrainer` object, which is responsible for managing the training loop, the optimizer, the learning rate scheduler, and the early stopping criteria. The training process is controlled by a configuration file, `training.toml`, which specifies the number of epochs, the learning rate, the loss function, and the splitting strategy for the dataset. The training process can be started by 


Training Configuration
------------------------------------------

The training process is controlled by a configuration file, typically written in TOML format. This file includes various sections for configuring different aspects of the training, such as logging, learning rate scheduling, loss functions, dataset splitting strategies, and early stopping criteria.


.. literalinclude:: ../modelforge/tests/data/training_defaults/default.toml
   :language: toml
   :caption: Training Configuration

The TOML files are split into four categories: `potential`, `training`,
`runtime` and `dataset`. While it is possible to read in a single TOML file
defining fields for all three categories, often it is useful to read them in
separately (a common use case where this is useful is when a potential is trained on
different datasets — instead of repeating the `training` and `potential`
sections in each TOML file, only the `dataset.toml` file needs to be changed).

Learning rate scheduler
^^^^^^^^^^^^^^^^^^^^^^^^

The learning rate scheduler is responsible for adjusting the learning rate during training. *Modelforge* uses the `REduceLROnPlateau scheduler`, which reduces the learning rate by a factor when the RMSE of the energy prediction on the validation set does not improve for a given number of epochs. The scheduler is controlled by the parameters in the `[training.lr_scheduler]` section of the `training.toml` file.

Loss function
^^^^^^^^^^^^^^^^^^^^^^^^
The loss function quantifies the discrepancy between the model's predictions and the target properties, providing a scalar value that guides the optimizer in updating the model's parameters. This function is configured in the `[training.loss]` section of the training TOML file.

Depending on the specified `loss_components`` section, the loss function can combine various individual loss functions. *Modelforge* always includes the mean squared error (MSE) for energy prediction, and may also incorporate MSE for force prediction, dipole moment prediction, and partial charge prediction.

The design of the loss function is intrinsically linked to the structure of the energy function. For instance, if the energy function aggregates atomic energies, then loss_components should include `per_system_energy` and optionally, `per_atom_force`.


Predicting Short-Range Atomic Energies
************************************************************

If the total atomic energy is calculated with a short cutoff radius, we can directly match the sum of the atomic energies `E_i` (as predicted by the potential model architecture) to the total energy of the molecule (provided by the dataset).

In that case the total energy is calculated as

.. math:: E = \sum_i^N E_i

The loss function can then be formulated as:

.. math:: L(E) = (E - E^{pred})^2

Alternatively, if the dataset includes per atom forces, the loss function can be expressed as:

.. math:: L(E,F) = w_E * (E - E^{pred})^2 + w_F \frac{1}{3N} \sum_i^N \sum_j^3 (F_{ij} - F_{ij}^{pred})^2

where `w_E` and `w_F` are the weights for the energy and force components, respectively.

Predicting Short-Range Atomic Energy with Long-Range Interactions
*************************************************************************

.. warning::
    The following section is under development and may not be fully implemented in the current version of *modelforge*.

In scenarios where long-range interactions are considered, additional terms are incorporated into the loss function. There are two long range interactions that are of interest: long range dispersion interactions and electrostatics.

To calculate long range electrostatics the first moment of the charge density (partial charges) is predicted by the machine learning potential.  The atomic
charges are then used to calculate the long range electrostatics. The expression
for the total energy is then:

.. math:: E = \sum_i^N E_i + k_c \sum_i^N \sum_{j>i}^N \frac{q_i q_j}{r_{ij}}

where `k_c` is the Coulomb constant, `q_i` and `q_j` are the atomic charges, and the loss function is:

.. math:: L(E,F,Q) = L(E,F) + w_Q (\sum_i^N q_i - Q_i^{pred})^2 + \frac{w_p}{3} \sum_j^3 (\sum_i^N q_i r_i,j - p_j^{ref})^2

where `w_Q` is the weight for the charge component, `w_p` the weight for the dipole moment component, and `p_j^{ref}` is the reference dipole moment. 


Splitting Strategies
^^^^^^^^^^^^^^^^^^^^^^^^

The dataset splitting strategy is crucial for ensuring that a potential generalizes well to unseen data. The recommended approach in *modelforge* is to randomly split the dataset into 80% training, 10% validation, and 10% test sets, based on molecules rather than individual conformations. This ensures that different conformations of the same molecule are consistently assigned to the same split, preventing data leakage and ensuring robust model evaluation.


*Modelforge* also provides other splitting strategies, including:

- :class:`~modelforge.dataset.utils.FirstComeFirstServeStrategy`: Splits the dataset based on the order of records (molecules).
- :class:`~modelforge.dataset.utils.RandomSplittingStrategy`: Splits the dataset randomly based on conformations.

To use a different data split ratio, you can specify a custom split list in the
splitting strategy. The most effective way to pass this information to the
training process is by defining the appropriate fields in the TOML file providing the training parameters, see the TOML file above. 


Train a Model
----------------------

The best way to get started is to train a model. We provide a script to
train one of the implemented models on a dataset using default configurations.
The script, along with a default configuration TOML file, can be found in the `scripts`` directory. The TOML file provides parameters to train a `TensorNet` potential on the `QM9` dataset on a single GPU.

The recommended method for training models is through the `perform_training.py`` script. This script automates the training process by reading the TOML configuration files, which define the potential (model), dataset, training routine, and runtime environment. The script then initializes a Trainer class from PyTorch Lightning, which handles the training loop, gradient updates, and evaluation metrics. During training, the model is optimized on the specified dataset according to the defined potential and training routine. After training completes, the script outputs performance metrics on the validation and test sets, providing insights into the model's accuracy and generalization.

Additionally, the script saves the trained model with checkpoint files in the directory specified by the `[save_dir]`` field in the runtime section of the TOML file, enabling you to resume training or use the potential model for inference later.

To initiate the training process, execute the following command in your terminal (inside the `scripts` directory):

.. code-block:: bash
   
    python perform_training.py
        --condensed_config_path="config.toml"
        --accelerator="gpu"
        --device=1

This command specifies the path to the configuration file, selects GPU acceleration, and designates the device index to be used during training. Adjust these parameters as necessary to fit your computational setup and training needs.


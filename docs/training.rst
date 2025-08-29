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

Depending on the specified `loss_components`` section, the loss function can combine various individual loss functions. *Modelforge* always includes the mean squared error (MSE) for energy prediction, and may also incorporate MSE for force prediction, dipole moment prediction, total_charge, and partial charge prediction.

The following keys can be used in the `loss_components` section. Note, to use these losses, corresponding known values need to be set within modelforge (see the dataset documentation for more details regarding defining the PropertyNames).  Each key is listed with corresponding properties that need to be defined in the dataset toml file:

* `per_system_energy' -- corresponds to `E` in the property names defitinion.
* `per_atom_force`  -- corresponds to `F` in the property names definition.
* `per_system_total_charge` -- corresponds to `total_charge` in the property names definition.
* `per_system_dipole_moment` -- corresponds to `dipole_moment` in the property names definition.
* `per_atom_charge` -- corresponds to `partial_charges` in the property names definition.

The design of the loss function is intrinsically linked to the structure of the energy function. For instance, if the energy function aggregates atomic energies, then loss_components should include `per_system_energy` and optionally, `per_atom_force`.

For each component in the loss, a weight must be specified in the `loss_weights` section of the training TOML file. The weights determine the relative importance of each component in the overall loss function. The loss function is then computed as a weighted sum of the individual components.

The weights for the losses can also be changed during the training run by providing a target weight (i.e., the final weight) and a mixing step (i.e., the step size for the linear interpolation between the current and target weights). This is useful for gradually increasing/decreasing the importance of certain loss components during training, which can help stabilize the training process. Note, that the target weights and mixing steps are optional, and if not provided, the loss weights will remain constant throughout the training process. In the snippet below, only `per_system_dipole_moment` has a target weight and a mixing step defined, meaning that the weight for this component will be gradually increased from 0.1 to 1 by increments of 0.01 over the course of training, while the other components will remain at their initial weights.

.. code-block:: toml

    [training.loss_parameter]
    loss_components = ['per_system_energy', 'per_system_dipole_moment', "per_atom_charge"]

    [training.loss_parameter.weight]
    per_system_energy = 0.00001
    per_system_dipole_moment = 0.1
    per_atom_charge = 1

    [training.loss_parameter.target_weight]
    per_system_dipole_moment = 1

    [training.loss_parameter.mixing_steps]
    per_system_dipole_moment= 0.01


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

Note, the ZBL potential can also be included, providing a repulsive interaction at extremely short distances (i.e., atomic overlap), as discussed in the :doc:`potentials` documentation. As with electrostatic or vdw interactions, these contributions must be summed into the total `per_system_energy` using postprocessing operations. An example code block is included below that shows how to set up postprocessing operations to include the ZBL interactions, as well as how to sum these into the total `per_system_energy`.

.. code-block:: toml

    [potential.postprocessing_parameter]
    properties_to_process = ['per_atom_energy', 'per_system_zbl_energy',  'sum_per_system_energy']

    [potential.postprocessing_parameter.per_atom_energy]
    normalize = true
    from_atom_to_system_reduction = true
    keep_per_atom_property = true

    [potential.postprocessing_parameter.per_system_zbl_energy]

    # this will be added to the `per_system_energy` that results from the `per_atom_energy` reduction operation
    [potential.postprocessing_parameter.sum_per_system_energy]
    contributions = ['per_system_zbl_energy']

Predicting Longer-Range Interactions
*************************************************************************



In scenarios where longer-range interactions are considered, additional terms are incorporated into the loss function. There are two longer range interactions that are of interest: dispersion interactions and electrostatics.

To calculate  electrostatics the first moment of the charge density (partial charges) is predicted by the machine learning potential.  The atomic
charges are then used to calculate the long range electrostatics. The expression
for the total energy is then:

.. math:: E = \sum_i^N E_i + k_c \sum_i^N \sum_{j>i}^N \frac{q_i q_j}{r_{ij}}

where `k_c` is the Coulomb constant, `q_i` and `q_j` are the atomic charges.  To avoid singularities at `r_{ij} = 0`, the Coulomb interaction is modified following the approach taken by PhysNet (J. Chem. Theory Comput. 2019, 15, 3678-3693, DOI:10.1021/acs.jctc.9b00181, see eqn: 12 and 13 in the reference). Specifically,

.. math:: E_coulomb = k_c \sum_i^N \sum_{j>i}^N q_i q_j\chi(r_{ij})

where:

.. math:: \chi(r_{ij}) = \phi(2r_{ij})\frac{1}/{\sqrt{r_{ij}^2 + 1}} + (1-\phi(2r_{ij}))\frac{1}{r_{ij}}.

and `\phi(r)` is represented by the PhysNet cutoff function:

.. math:: \phi(r) = \begin{cases} 1-6(\frac{r}{r_{cut}})^5 + 15(\frac{r}{r_{cut}})^4 - 10(\frac{r}{r_{cut}})^3 & r \leq r_{cut} \\ 0 & r > r_{cut} \end{cases}

The loss function to optimize charges is represented as, where we note that this includes both the partial charges and the dipole moment; these can be toggled on and off separately:

.. math:: L(E,F,Q) = L(E,F) + w_Q (\sum_i^N q_i - Q_i^{pred})^2 + \frac{w_p}{3} \sum_j^3 (\sum_i^N q_i r_i,j - p_j^{ref})^2

where `w_Q` is the weight for the charge component, `w_p` the weight for the dipole moment component, and `p_j^{ref}` is the reference dipole moment. 

van Der Waals interactions can also be calculated; this is done using the tad-DFTD3 package. These interactions are calculated outside of the machine learning potential, and added to the total `per_system_energy`.

As also discussed in the context of AIMNET2 in the :doc:`potentials` documentation, to include electrostic and/or dispersion interactions, we must sum these into the total `per_system_energy`. This is not done automatically, and must be specificied in the provided .toml file, specifically as post processing operations.  An example code block is included below that shows how to set up postprocessing operations to include electrostatics and van der Waals interactions, as well as how to sum these into the total `per_system_energy`.

.. code-block:: toml

    [potential.postprocessing_parameter]
    properties_to_process = ['per_atom_energy', 'per_system_electrostatic_energy', 'per_system_vdw_energy', 'sum_per_system_energy']

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
    [potential.postprocessing_parameter.sum_per_system_energy]
    contributions = ['per_system_electrostatic_energy', 'per_system_vdw_energy']

Splitting Strategies
^^^^^^^^^^^^^^^^^^^^^^^^

The dataset splitting strategy is crucial for ensuring that a potential generalizes well to unseen data. The recommended approach in *modelforge* is to randomly split the dataset into 80% training, 10% validation, and 10% test sets, based on molecules rather than individual conformations. This ensures that different conformations of the same molecule are consistently assigned to the same split, preventing data leakage and ensuring robust model evaluation.


*Modelforge* also provides other splitting strategies, including:

- :class:`~modelforge.dataset.utils.FirstComeFirstServeStrategy`: Splits the dataset based on the order of records (molecules).
- :class:`~modelforge.dataset.utils.RandomSplittingStrategy`: Splits the dataset randomly based on configurations.
- :class:`~modelforge.dataset.utils.RandomRecordSplittingStrategy`: Splits the dataset randomly based on records (i.e., systems, keeping all configurations of a system in the same split).

To use a different data split ratio, you can specify a custom split list in the
splitting strategy. The most effective way to pass this information to the
training process is by defining the appropriate fields in the TOML file providing the training parameters, see the TOML file above. 

Note, The `RandomSplittingStrategy` and `RandomRecordSplittingStrategy` classes accept a `seed` parameter, which can be set to ensure reproducibility of the data splits across different training runs.  To provide further control, the `RandomRecordSplittingStrategy` class also accepts a `test_seed` parameter, which allows for independent control over the randomness of the test set split, separate from the training and validation splits. For a given dataset, if test_seed is not changed, the test set will remain the same even if the `seed` for the training and validation sets is altered (training and validation will change if `seed` is changed in this case). This feature is particularly useful for benchmarking and comparing model performance across different training configurations while keeping the test set consistent.

To specify in the toml file, use the following keywords for the strategy argument:
- "first_come_first_serve_strategy"
- "random_splitting_strategy"
- "random_record_splitting_strategy"

For example, the following snippet would specify the RandomRecordSplittingStrategy with a 70/15/15 split, a seed of 42 for the training and validation sets, and a test_seed of 7 for the test set:

.. code-block:: toml

    [dataset.splitting_strategy]
    strategy = "random_record_splitting_strategy"
    split = [0.7, 0.15, 0.15]
    seed = 42
    test_seed = 7

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


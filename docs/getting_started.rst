Getting Started
##########################


This page details how to get started with *modelforge*, a library designed for training and evaluating machine learning models for interatomic potentials.


Installation
**************


To ensure a clean environment for *modelforge*, we recommend creating a new conda environment using the provided environment YAML file located in the devtools directory of the *modelforge* repository.


.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yml --name modelforge

This command will create a new conda environment named *modelforge* with all necessary dependencies installed. Note, this package has currently been tested and validated with Python 3.10 and 3.11.

Next, clone the source code from the GitHub repository:


.. code-block:: bash

    git clone https://github.com/choderalab/modelforge

Navigate to the top level of the *modelforge* directory and install the package using pip:

.. code-block:: bash
   
    pip install  .

.. note::
    Modelforge has currently been tested and validated with Python 3.10 and 3.11 on various Linux distributions. While it may work on other platforms, at this time we recommend using a Linux environment.

Use Cases for *Modelforge* 
****************************


*Modelforge* may be a good fit for your project if you are interested in:

- Training machine-learned interatomic potentials using PyTorch and PyTorch Lightning.
- Utilizing pre-trained models for inference tasks.
- Exporting trained models to Jax for accelerated computation.
- Investigating the impact of hyperparameters on model performance.
- Developing new machine learning models without managing dataset and training infrastructure.



How to Use *Modelforge*
****************************

Training a Model
============================

Before training a model, consider the following:

1. **Architecture**: Which neural network architecture do you want to use?
2. **Training Set**: What dataset will you use for training?
3. **Loss Function**: Which properties will you include in the loss function?

*Modelforge* currently supports the following architectures:

- Invariant architectures:
    * SchNet
    * ANI2x

- Equivariant architectures:
    * PaiNN
    * PhysNet
    * TensorNet
    * SAKE

These architectures can be trained on the following datasets (distributed via zenodo https://zenodo.org/communities/modelforge/ ):

- Ani1x
- Ani2x
- PHALKETOH
- QM9
- SPICE1 (/openff)
- SPICE2

By default, potentials predict the total energy  and per-atom forces within a given cutoff radius and can be trained on energies and forces.

.. note:: PaiNN and PhysNet can also predict partial charges and calculate long-range interactions. PaiNN can additionally use multipole expansions. These features will introduce additional terms to the loss function.

In the following example, we will train a SchNet model on the ANI1x dataset with energies and forces as the target properties. TOML files are used to define the potential architecture, dataset, training routine, and runtime parameters.

Defining the Potential
+++++++++++++++++++++++++++++++++++++++

The potential architecture and relevant parameters are defined in a TOML configuration file. Here is an example of a potential definition for a SchNet model. Note that we use 16 radial basis functions, a maximum interaction radius of 5.0 angstroms, and 16 filters. We use a `ShiftedSoftplus`` activation (the fully differentiable version of ReLu) function and featurize the atomic number of the atoms in the dataset. Finally, we normalize the per-atom energy and reduce the per-atom energy to the per-molecule energy (which will then be returned)..


.. code-block:: toml

    [potential]
    potential_name = "SchNet"

    [potential.core_parameter]  # Parameters defining the architecture of the model
    number_of_radial_basis_functions = 16
    maximum_interaction_radius = "5.0 angstrom"
    number_of_interaction_modules = 3
    number_of_filters = 16
    shared_interactions = false

    [potential.core_parameter.activation_function_parameter]
    activation_function_name = "ShiftedSoftplus"

    [potential.core_parameter.featurization]  # Parameters defining the embedding of the input data
    properties_to_featurize = ['atomic_number']
    maximum_atomic_number = 101
    number_of_per_atom_features = 32

    [potential.postprocessing_parameter]
    [potential.postprocessing_parameter.per_atom_energy]
    normalize = true
    from_atom_to_molecule_reduction = true
    keep_per_atom_property = true

Defining the Dataset
+++++++++++++++++++++++++++++++++++++++

The following TOML file defines the ANI1x dataset, allowing users to specify a specific version, as well as parameters used by the torch dataloaders (num_workers and pin_memory):

.. code-block:: toml

    [dataset]
    dataset_name = "ANI1x"
    version_select = "latest"
    num_workers = 4
    pin_memory = true


Defining the Training Routine
+++++++++++++++++++++++++++++++++++++++


The training TOML file includes the number of epochs, batch size, learning rate, logger, callback parameters, and other training parameters (including dataset splitting).
Each of these settings plays a crucial role in the training process.

Here is an example of a training routine definition:

.. code-block:: toml

    [training]
    number_of_epochs = 2  # Total number of training epochs
    remove_self_energies = true  # Whether to remove self-energies from the dataset
    batch_size = 128  # Number of samples per batch
    lr = 1e-3  # Learning rate for the optimizer
    monitor = "val/per_molecule_energy/rmse"  # Metric to monitor for checkpointing


    [training.experiment_logger]
    logger_name = "wandb"  # Logger to use for tracking the training process

    [training.experiment_logger.wandb_configuration]
    save_dir = "logs"  # Directory to save logs
    project = "training_test"  # WandB project name
    group = "modelforge_nnps"  # WandB group name
    log_model = true  # Whether to log the model in WandB
    job_type = "training"  # Job type for WandB logging
    tags = ["modelforge", "v_0.1.0"]  # Tags for WandB logging
    notes = "testing training"  # Notes for WandB logging

    [training.lr_scheduler]
    frequency = 1  # Frequency of learning rate updates
    mode = "min"  # Mode for the learning rate scheduler (minimizing the monitored metric)
    factor = 0.1  # Factor by which the learning rate will be reduced
    patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
    cooldown = 5  # Number of epochs to wait before resuming normal operation after learning rate has been reduced
    min_lr = 1e-8  # Minimum learning rate
    threshold = 0.1  # Threshold for measuring the new optimum, to only focus on significant changes
    threshold_mode = "abs"  # Mode for the threshold (absolute or relative)
    monitor = "val/per_molecule_energy/rmse"  # Metric to monitor for learning rate adjustments
    interval = "epoch"  # Interval for learning rate updates (per epoch)

    [training.loss_parameter]
    loss_property = ['per_molecule_energy', 'per_atom_force']  # Properties to include in the loss function

    [training.loss_parameter.weight]
    per_molecule_energy = 0.999  # Weight for per molecule energy in the loss calculation
    per_atom_force = 0.001  # Weight for per atom force in the loss calculation

    [training.early_stopping]
    verbose = true  # Whether to print early stopping messages
    monitor = "val/per_molecule_energy/rmse"  # Metric to monitor for early stopping
    min_delta = 0.001  # Minimum change to qualify as an improvement
    patience = 50  # Number of epochs with no improvement after which training will be stopped

    [training.splitting_strategy]
    name = "random_record_splitting_strategy"  # Strategy for splitting the dataset
    data_split = [0.8, 0.1, 0.1]  # Proportions for training, validation, and test sets
    seed = 42  # Random seed for reproducibility

Defining Runtime Variables
+++++++++++++++++++++++++++++++++++++++

To define various aspects of the compute environment, various runtime parameters can be set. Here is an example of a runtime variable definition:

.. code-block:: toml

    [runtime]
    save_dir = "lightning_logs"  # Directory to save logs and checkpoints
    experiment_name = "exp1"  # Name of the experiment
    local_cache_dir = "./cache"  # Directory for caching data
    accelerator = "cpu"  # Type of accelerator to use (e.g., 'cpu' or 'gpu')
    number_of_nodes = 1  # Number of nodes to use for distributed training
    devices = 1  # Number of devices to use
    checkpoint_path = "None"  # Path to a checkpoint to resume training
    simulation_environment = "PyTorch"  # Simulation environment
    log_every_n_steps = 50  # Frequency of logging steps

All of the above TOML files can be passed invididually or combined into a single TOML file that defines the training run. Assuming the combined TOML file is called `training.toml`, start the training by passing the TOML file to the perform_training.py script.


.. code-block:: bash
   
    python scripts/perform_training.py
        --condensed_config_path="training.toml"

*modelforge* uses Pydantic to validate the TOML files, ensuring that all required fields are present and that the values are of the correct type before any expensive computational operations are performed. This validation process helps to catch errors early in the training process. If the TOML file is not valid, an error message will be displayed, indicating the missing or incorrect fields.

Using a Pretrained Model
============================
.. warning:: This feature is currently a work in progress.


All training runs performed with *modelforge* are logged in a `wandb` project. You can access this project via the following link : https://https://wandb.ai/modelforge_nnps/projects/latest. Using the wandb API, you can download the trained models and use them for inference tasks.


Investigating Hyperparameter Impact on Model Performance
========================================================

For each supported architecture, modelforge provides reasonable priors for hyperparameters. Hyperparameter optimization is conducted using `Ray`.

.. autoclass:: modelforge.train.tuning.RayTuner
    :noindex:


*Modelforge* offers the :py:class:`~modelforge.train.tuning.RayTuner` class, which facilitates the exploration of hyperparameter impacts on model performance given specific training and dataset parameters within a defined computational budget.





.. toctree::
   :maxdepth: 2
   :caption: Contents:
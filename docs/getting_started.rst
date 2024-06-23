Getting Started
===============

This page details how to get started with modelforge. 

Installation
----------------------

We recommend building a clean conda environment for modelforge using the 
environment yaml file in the `devctools` modelforge directory.

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yml --name modelforge

This will create a new conda environment called `modelforge` with all 
the necessary dependencies.
THen check out the source from the GitHub repository:

.. code-block:: bash

    git clone https://github.com/choderalab/modelforge

In the top level of the modelforge directory, use pip to install:

.. code-block:: bash
   
    pip install -e .

Train a Model
----------------------

The best way to get started is to train a model. We provide a simple
script to train one of the implemented models on a dataset using default 
configurations.

We provide default parameters in TOML files in the `scripts/config` directory.
The TOML files are split into three categories: `potential`, `training`, and `dataset`.
While it is possible to read in a single TOML file defining fields for all three categories, 
often it is much more useful to read them in separately (a common use case where this is useful 
is when a model is trained on different datasets — instead of repeating the `training` and `potential` 
sections in each TOML file, only the `dataset.toml` file needs to be changed).

For further information on the default parameters, see the models documentation.

TOML Configuration Files
---------------------------------

QM9 Dataset Configuration (qm9.toml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../scripts/configs/datasets/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

Explanation of fields in `qm9.toml`:

- `dataset_name`: Name of the dataset, here it is QM9.
- `number_of_worker`: Number of worker threads for data loading.

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

Training Configuration (training.toml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../scripts/configs/training/default.toml
   :language: toml
   :caption: Training Configuration

Explanation of fields in `training.toml`:

- `nr_of_epochs`: Number of training epochs.
- `save_dir`: Directory to save training outputs.
- `experiment_name`: Name of the experiment.
- `accelerator`: Type of accelerator to use (e.g., gpu).
- `num_nodes`: Number of nodes.
- `devices`: Number of devices (GPUs).
- `batch_size`: Size of the batches for training.
- `remove_self_energies`: Whether to remove self-energies.
- `pin_memory`: Whether to pin memory during training.
- `training_parameter.lr`: Learning rate for the optimizer.
- `training_parameter.lr_scheduler_config`: Configuration for the learning rate scheduler.
- `early_stopping`: Configuration for early stopping criteria.
- `loss_parameter.loss_type`: Type of loss function used.

Training Script
----------------------

The training script is located in the `scripts` folder. If you want to train the `SchNet` potential 
on the `QM9` dataset on a single GPU using the default parameters, run the following command:

.. code-block:: bash
   
    python perform_training.py
        --potential_path="configs/potentials/schnet.toml"
        --dataset_path="configs/datasets/qm9.toml"
        --training_path="configs/training.toml"
        --accelerator="gpu"
        --device=1

This command calls the :py:func:`training.perform_training` function in the `modelforge.train.training` module.
The function reads in the TOML files and trains the model on the specified dataset using the specified potential and training parameters.

.. autoclass:: modelforge.train.training.perform_training
    :members:



Indices and Tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Examples
===============

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

Training Script
^^^^^^^^^^^^^^^^^

The training script is located in the `scripts` folder. If you want to train the `SchNet` potential 
on the `QM9` dataset on a single GPU using the default parameters, run the following command:

.. code-block:: bash
   
    python perform_training.py
        --potential_path="configs/potentials/schnet.toml"
        --dataset_path="configs/datasets/qm9.toml"
        --training_path="configs/training.toml"
        --accelerator="gpu"
        --device=1

This command initializes the :py:class:`modelforge.train.training.ModelTrainer` class. This class takes at initialization time the TOML files defining the potential, dataset, training routine and runtime and sets up a `Lightning` `Trainer` class. It then trains the model on the dataset using the potential and training routine, and at the end of training prints performance metrics on the validation and test sets.

.. autoclass:: modelforge.train.training.ModelTrainer
    :members:

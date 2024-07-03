Training
========

Introduction
------------------------------------------

Learning rate scheduler
------------------------------------------



Loss function
------------------------------------------



Splitting Strategies
------------------------------------------

The default splitting strategy for datasets in `modelforge` is to randomly split the dataset into 80% training, 10% validation, and 10% test sets based on records. 
This approach ensures that different conformations of a molecule are always part of the same split, thereby avoiding data leakage. 

`modelforge` also provides other splitting strategies, including:

- :class:`modelforge.dataset.utils.FirstComeFirstServeStrategy`: Splits the dataset based on the order of records.
- :class:`modelforge.dataset.utils.RandomSplittingStrategy`: Splits the dataset randomly based on conformations.

To use a different data split ratio, you can specify a custom split list in the splitting strategy. 
The most effective way to pass this information to the training process is by defining the appropriate fields in the `dataset.toml` file, as shown in :ref:`dataset-configuration`.

.. autoclass:: modelforge.dataset.utils.RandomRecordSplittingStrategy



Training Configuration (training.toml)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../modelforge/tests/data/training_defaults/default.toml
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
- `splitting_strategy`: The splitting strategy to use, possible values are 'first_come_first_serve', 'random_record_splitting_strategy', and 'random_conformer_splitting_strategy'.

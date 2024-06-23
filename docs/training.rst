Training
========


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


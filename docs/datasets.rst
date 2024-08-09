Datasets
===============

The dataset module in modelforge provides a comprehensive suite of functions and classes designed to retrieve, transform, and store quantum mechanics (QM) datasets from QCArchive. These datasets are delivered in a format compatible with `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, facilitating the training of machine learning potentials. The module supports actions related to data storage, caching, retrieval, and the conversion of stored HDF5 files into PyTorch-compatible datasets for training purposes.


General Workflow
----------------
The typical workflow to interact with public datasets includes the following steps:

1. *Obtaining the Dataset*: Download the raw dataset from QCArchive or another source.
2. *Processing the Dataset*: Convert the raw data into a standardized format and store it in an HDF5 file with consistent naming conventions and units.
3. *Uploading and Updating*: Upload the processed dataset to Zenodo and update the retrieval link in the dataset implementation within *modelforge*.

For more information on how units are handled within the dataset, available properties, and instructions on developing custom datasets for *modelforge*, please refer to the `developer documentation <for_developer.html>`_.

Available Datasets
------------------

The following datasets are available for use with `modelforge`:

- :py:class:`~modelforge.dataset.QM9Dataset`
- :py:class:`~modelforge.dataset.ANI1xDataset`
- :py:class:`~modelforge.dataset.ANI2xDataset`
- :py:class:`~modelforge.dataset.SPICE1Dataset`
- :py:class:`~modelforge.dataset.SPICE1_OPENFF`
- :py:class:`~modelforge.dataset.SPICE2Dataset`
- :py:class:`~modelforge.dataset.PhAlkEthOHDataset`

These datasets encompass a variety of molecular structures and properties, providing robust training data for developing machine learning potentials.

Postprocessing of dataset entries
-----------------------------------

Two common postprocessing operations are performed for training machine learned potentials:

- *Removing Self-Energies*: Self-energies are per-element offsets added to the
  total energy of a molecule. These offsets are not useful for training
  machine-learned potentials and can be removed to provide cleaner training
  data.
- *Normalization and Scaling*: Normalize the energies and other properties to
  ensure they are on a comparable scale, which can improve the stability and
  performance of the machine learning model. Note that this is done when atomic energies are predicted, i.e. the atomic energy (`E_i`) is scaled using the atomic energy distribution obtained from the training dataset: `E_i = E_i_stddev * E_i_pred + E_i_mean`.


Interacting with the Dataset Module
-----------------------------------

The dataset module provides a :class:`~modelforge.dataset.DataModule` class for
preparing and setting up datasets for training. Designed to integrate seamlessly
with PyTorch Lightning, the :class:`~modelforge.dataset.DataModule` class
provides a user-friendly interface for dataset preparation and loading.

The following example demonstrates how to use the :class:`~modelforge.dataset.DataModule` class to prepare and set up a dataset for training:


.. code-block:: python

    from modelforge.dataset import DataModule
    from modelforge.dataset.utils import RandomRecordSplittingStrategy

    dataset_name = "QM9"
    splitting_strategy = RandomRecordSplittingStrategy() # split randomly on molecules
    batch_size = 64
    version_select = "latest" 
    remove_self_energies = True # remove the atomic self energies
    regression_ase = False      # use the atomic self energies provided by the dataset

    data_module = DataModule(
        name=dataset_name,
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        version_select=version_select,
        remove_self_energies=remove_self_energies,
        regression_ase=regression_ase,
    )

    # Prepare the data (downloads, processes, and caches if necessary)
    data_module.prepare_data()

    # Setup the data for training, validation, and testing
    data_module.setup()

.. _dataset-configuration:

Dataset Configuration
------------------------------------

Dataset configuration in modelforge is typically managed using a TOML file. This configuration file is crucial during the training process as it overrides the default values specified in the :class:`~modelforge.dataset.DataModule` class, ensuring a flexible and customizable setup.

Below is a minimal example of a dataset configuration for the QM9 dataset.

.. literalinclude:: ../modelforge/tests/data/dataset_defaults/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

.. warning::
    The ``version_select`` field in the example indicates the use of a small subset of the QM9 dataset. To utilize the full dataset, set this variable to ``latest``.


Explanation of fields in `qm9.toml`:

- `dataset_name`: Specifies the name of the dataset. For this example, it is QM9.
- `number_of_worker`: Determines the number of worker threads for data loading. Increasing the number of workers can speed up data loading but requires more memory.
- `version_select`: Indicates the version of the dataset to use. In this example, it points to a small subset of the dataset for quick testing. To use the full QM9 dataset, set this variable to `latest`.


Datasets
===============
The dataset module in `modelforge` provides a comprehensive suite of functions and classes designed to retrieve, transform, and store QM datasets from QCArchive. 
These datasets are delivered in a format compatible with `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_, facilitating the training of neural network potentials (NNPs). 
The module supports actions related to data storage, caching, retrieval, and the conversion of stored HDF5 files into PyTorch-compatible datasets for training purposes.

General Workflow
----------------
The typical workflow to interact with public datasets includes the following steps:

- Obtaining the dataset
- Processing the dataset and storing it in an HDF5 file with standardizednaming and units
- Uploading the processed dataset to Zenodo and updating the retrieval link in the dataset implementation

For more information on how units are handled within the dataset, available properties, and instructions on developing custom datasets for `modelforge`, please refer to the `developer documentation <for_developer.html>`_.

Available Datasets
------------------

The following datasets are available for use with `modelforge`:

- :py:class:`modelforge.dataset.QM9Dataset`
- :py:class:`modelforge.dataset.ANI1xDataset`
- :py:class:`modelforge.dataset.ANI2xDataset`
- :py:class:`modelforge.dataset.SPICE114Dataset`
- :py:class:`modelforge.dataset.SPICE114OpenFFDataset`
- :py:class:`modelforge.dataset.PhAlkEthOHDataset`


Splitting Strategies
---------------------

The default splitting strategy for datasets in `modelforge` is to randomly split the dataset into 80% training, 10% validation, and 10% test sets based on records. 
This approach ensures that different conformations of a molecule are always part of the same split, thereby avoiding data leakage. 

`modelforge` also provides other splitting strategies, including:

- :class:`modelforge.dataset.utils.FirstComeFirstServeStrategy`: Splits the dataset based on the order of records.
- :class:`modelforge.dataset.utils.RandomSplittingStrategy`: Splits the dataset randomly based on conformations.

To use a different data split ratio, you can specify a custom split list in the splitting strategy. 
The most effective way to pass this information to the training process is by defining the appropriate fields in the `dataset.toml` file, as shown in :ref:`dataset-configuration`.

.. autoclass:: modelforge.dataset.utils.RandomRecordSplittingStrategy


Postprocessing of dataset entries
-----------------------------------

Two common postprocessing operations are performed for training machine learned potentials:

- Remove self-energies for each molecule of the dataset. Self-energies are per element offsets that are added to the total energy of the molecule. These self-energies are not useful for training machine learned potentials and can be removed.


Interacting with the Dataset Module
-----------------------------------

The dataset module provides a :class:`modelforge.dataset.DataModule` class for preparing and setting up datasets for training. 
Designed to integrate seamlessly with PyTorch Lightning, the :class:`modelforge.dataset.DataModule` class provides a user-friendly interface for dataset preparation and loading.

.. autoclass:: modelforge.dataset.DataModule

The following example demonstrates how to use the :class:`modelforge.dataset.DataModule` class to prepare and set up a dataset for training:


.. code-block:: python

    from modelforge.dataset import DataModule
    from modelforge.dataset.utils import RandomRecordSplittingStrategy

    dataset_name = "QM9"
    splitting_strategy = RandomRecordSplittingStrategy()
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

Typically, the dataset configuration is stored in a TOML file, which is used for the training process. This configuration file overrides the default values specified in the :class:`modelforge.dataset.DataModule` class. 
Below is a minimal example of a dataset configuration for the QM9 dataset.

.. literalinclude:: ../modelforge/tests/data/dataset_defaults/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

Explanation of fields in `qm9.toml`:

- `dataset_name`: Name of the dataset, here it is QM9.
- `number_of_worker`: Number of worker threads for data loading.
- `splitting_strategy`: The splitting strategy to use, possible values are 'first_come_first_serve', 'random_record_splitting_strategy', and 'random_conformer_splitting_strategy'.
- `number_of_worker`: Number of worker threads for data loading.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

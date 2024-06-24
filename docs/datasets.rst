Datasets
===============

The dataset module provides functions and classes to retrieve, transform, 
and store QM datasets from QCArchive and delivers them as 
`torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_ to train NNPs. 
The dataset module implements actions associated with data storage, caching, and retrieval, 
as well as the pipeline from the stored hdf5 files to the PyTorch dataset class that can be 
used for training. The general workflow to interact with public datasets will be as follows:

- obtaining the dataset
- processing the dataset and storing it in an HDF5 file with standard naming and units
- uploading to Zenodo and updating the retrieval link in the dataset implementation

For more information on how units are handled in the dataset, which properties are available and 
how to develop your own dataset for `modelforge` we refer to the 
`developer documentation <for_developer.html>`_.


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


Interacting with the Dataset Module
-----------------------------------

The dataset module provides a `DataModule` class that can be used to prepare 
and set up a dataset for training. The `DataModule` class is designed to be used with 
PyTorch Lightning, and provides a convenient interface for preparing and loading datasets.



The dataset module allows for flexible and efficient handling of datasets. Below is an example of how to use the `DataModule` class to prepare and set up a dataset for training:

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

Explanation of the DataModule Attributes and Methods
----------------------------------------------------

.. autoclass:: modelforge.dataset.DataModule

Dataset Configuration
------------------------------------

Typically, the dataset configuration is stored in a TOML file for training.
The config TOML file overwrites in the `DataModule` class.
Below is a minmal example of a dataset configuration for the QM9 dataset.


.. literalinclude:: ../scripts/configs/datasets/qm9.toml
   :language: toml
   :caption: QM9 Dataset Configuration

Explanation of fields in `qm9.toml`:

- `dataset_name`: Name of the dataset, here it is QM9.
- `number_of_worker`: Number of worker threads for data loading.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

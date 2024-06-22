Getting Started
===============

This page details how to get started with modelforge. 

Installation
===============

We recommend to build a clean conda environment for modelforge using the 
environment yaml file in the `devctools` modelforge directory.

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yml --name modelforge

This will create a new conda environment called `modelforge` with all 
the necessary dependencies.

pip install
-----------

Check out the source from the github repository:

.. code-block:: bash

   git clone https://github.com/choderalab/modelforge

In the top level of modelforge directory, use pip to install:

.. code-block:: bash
   
   pip install -e .

Train a model
-----------

.. autoclass:: modelforge.train.training.perform_training
    :members:



Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


=========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



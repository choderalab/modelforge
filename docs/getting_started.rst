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


TOML Configuration Files
---------------------------------




Indices and Tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

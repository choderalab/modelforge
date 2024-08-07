Getting Started
===============

This page details how to get started with modelforge. 

Installation
----------------------

We recommend building a clean conda environment for modelforge using the 
environment yaml file in the `devtools` modelforge directory.

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yml --name modelforge

This will create a new conda environment called `modelforge` with all 
the necessary dependencies. Then check out the source from the GitHub repository:

.. code-block:: bash

    git clone https://github.com/choderalab/modelforge

In the top level of the modelforge directory, use pip to install:

.. code-block:: bash
   
    pip install -e .


Use cases for modelforge
-------------------------

`Modelforge` might be a good fit for your project if you are interested in:

- Training machine learned interatomic potentials using PyTorch and Lightning.
- Using pre-trained models for inference tasks.
- Exporting trained models to `Jax`.
- Investigating the impact of hyperparatmeter on model performance.
- Developing new machine learning models with out having to deal with dataset and training infrastructure.


How to use modelforge
----------------------

If you want to train a model
++++++++++++++++++++++++++++

There are certain considerations before training a model: 

1. what architecture do you want to use?
2. what training set do you want to use?
3. and what properties do you want to include in the loss function?

Currently, modelforge supports training models with the following architectures: SchNet, PaiNN, ANI2x, PhysNet, SAKE and TensorNet.
Each of these networks can be trained on the supported datasets (Ani1x, Ani2x, PHALKETOH, QM9, SPICE1(/openff) and SPICE2).
By default models predict the total energy and force per atom within a given cutoff radius and can be trained on energies and forces.

.. note:: PaiNN and PhysNet can also predict partial charges and calculate long range interactions. PaiNN can also use mutlipole expensions. This will add additional terms to the loss function.


If you want to use a pretrained model
+++++++++++++++++++++++++++++++++++++++
.. warning:: This is work in progress.

All training runs performed with `modelforge` are saved in a `wandb` project. You can access the project by following this link: https://https://wandb.ai/modelforge_nnps/projects/latest
Using the `wandb` API you can download the trained models and use them for inference tasks.

If you want to Investigate the impact of hyperparatmeter on model performance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





Modelforge



.. toctree::
   :maxdepth: 2
   :caption: Contents:
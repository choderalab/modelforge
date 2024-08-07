Getting Started
===============

This page details how to get started with *modelforge*, a library designed for training and evaluating machine learning models for interatomic potentials.


Installation
----------------------

To ensure a clean environment for *modelforge*, we recommend creating a new conda environment using the provided environment YAML file located in the devtools directory of the *modelforge* repository.


.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yml --name modelforge

This command will create a new conda environment named *modelforge* with all necessary dependencies installed. Next, clone the source code from the GitHub repository:


.. code-block:: bash

    git clone https://github.com/choderalab/modelforge

Navigate to the top level of the *modelforge* directory and install the package using pip:

.. code-block:: bash
   
    pip install  .


Use Cases for *Modelforge* 
-------------------------

*Modelforge* might be a good fit for your project if you are interested in:

- Training machine-learned interatomic potentials using PyTorch and PyTorch Lightning.
- Utilizing pre-trained models for inference tasks.
- Exporting trained models to Jax for accelerated computation.
- Investigating the impact of hyperparameters on model performance.
- Developing new machine learning models without managing dataset and training infrastructure.



How to use modelforge
----------------------

Training a Model
++++++++++++++++++++++++++++

Before training a model, consider the following:

1. **Architecture**: Which neural network architecture do you want to use?
2. **Training Set**: What dataset will you use for training?
3. **Loss Function**: Which properties will you include in the loss function?

*Modelforge* currently supports the following architectures:

- Invariant arhitectures: SchNet, ANI2x
- Equivariant architectures: PaiNN, PhysNet, TensorNet, SAKE

These architectures can be trained on the following datasets:

- Ani1x
- Ani2x
- PHALKETOH
- QM9
- SPICE1 (/openff)
- SPICE2

By default, models predict the total energy and per-atom forces within a given cutoff radius and can be trained on energies and forces.

.. note:: PaiNN and PhysNet can also predict partial charges and calculate long-range interactions. PaiNN can additionally use multipole expansions. These features will introduce additional terms to the loss function.

In the following example, we will train a SchNet model on the ANI1x dataset with energies and forces as the target properties.

To train a model we will use the following configuration file:

```toml





Using a Pretrained Model
+++++++++++++++++++++++++++++++++++++++
.. warning:: This feature is currently a work in progress.


All training runs performed with *modelforge* are logged in a `wandb` project. You can access this project via the following link : https://https://wandb.ai/modelforge_nnps/projects/latest. Using the wandb API, you can download the trained models and use them for inference tasks.


Investigating Hyperparameter Impact on Model Performance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For each supported architecture, modelforge provides reasonable priors for hyperparameters. Hyperparameter optimization is conducted using `Ray`.

.. autoclass:: modelforge.train.tuning.RayTuner


*Modelforge* offers the :py:class:`~modelforge.train.tuning.RayTuner` class, which facilitates the exploration of hyperparameter impacts on model performance given specific training and dataset parameters within a defined computational budget.





.. toctree::
   :maxdepth: 2
   :caption: Contents:
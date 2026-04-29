modelforge
==============================
[//]: # (Badges)
[![CI](https://github.com/choderalab/modelforge/actions/workflows/CI.yaml/badge.svg)](https://github.com/choderalab/modelforge/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/choderalab/modelforge/branch/main/graph/badge.svg)](https://codecov.io/gh/choderalab/modelforge/branch/main)
[![Documentation Status](https://readthedocs.org/projects/modelforge/badge/?version=latest)](https://modelforge.readthedocs.io/en/latest/?badge=latest)
[![Github release](https://badgen.net/github/release/choderalab/modelforge)](https://github.com/choderalab/modelforge/)
[![GitHub license](https://img.shields.io/github/license/choderalab/modelforge?color=green)](https://github.com/choderalab/modelforge/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/choderalab/modelforge?style=flat)](https://github.com/choderalab/modelforge/issues)
[![GitHub stars](https://img.shields.io/github/stars/choderalab/modelforge)](https://github.com/choderalab/modelforge/stargazers)

### 
This package is centered around the implementation and infrastructure to train, create, optimize and store neural network potentials (NNPs) effectively. 
Datasets are provided to enable accurate training and validation of the neural network structures, and users can include their own using the API in modelforge-curate. 
The technical roadmap for the modelforge package is outlined in the wiki. 

Documentation for how to use the package can be found at: https://modelforge.readthedocs.io/en/latest/

Subpackages
-----------

- `modelforge`: Core infrastructure for training, evaluating, and managing neural network potentials.
- `modelforge-curate`: Dataset curation API and tools for building standardized datasets.
- `modelforge-ase`: ASE calculator wrapper for running modelforge potentials through ASE workflows.
- `modelforge-openmm`: OpenMM wrapper for using modelforge models in OpenMM simulations.

Installation
------------

To set up any environment, first clone the repository:
```bash
git clone https://github.com/choderalab/modelforge.git
```
navigate to the modelforge root directory:
```bash
cd modelforge
```

### Install Core Package (`modelforge`)

Create and activate the base environment:
```bash
micromamba create -n modelforge -f devtools/conda-envs/env.yaml
micromamba activate modelforge
``` 

Install the core package:
```bash
pip install -e . --no-deps --config-settings editable-mode=strict
```

### Install Dataset Curation Subpackage (`modelforge-curate`)

From the base `modelforge` environment:
```bash
pip install -e modelforge-curate --no-deps --config-settings editable-mode=strict
```

### Install ASE Integration Subpackage (`modelforge-ase`)

Create and activate the ASE runtime environment:
```bash
micromamba create -n modelforge-ase -f devtools/conda-envs/env_modelforge_ase.yaml
micromamba activate modelforge-ase
```

Install `modelforge` and the ASE integration package:
```bash
pip install -e . --no-deps --config-settings editable-mode=strict
pip install -e modelforge-ase --no-deps --config-settings editable-mode=strict
```

### Install ASE Examples/Notebook Dependencies

If you want to run ASE notebooks/examples (includes `ase`, `ipykernel`, `nglview`):

```bash
micromamba create -n modelforge-ase-examples -f devtools/conda-envs/env_modelforge_ase_examples.yaml
micromamba activate modelforge-ase-examples
pip install -e . --no-deps --config-settings editable-mode=strict
pip install -e modelforge-ase --no-deps --config-settings editable-mode=strict
python -m ipykernel install --user --name modelforge-ase-examples --display-name "Python (modelforge-ase-examples)"
```

### Install OpenMM Integration Subpackage (`modelforge-openmm`)

Create and activate the OpenMM runtime environment:

```bash
micromamba create -n modelforge-openmm -f devtools/conda-envs/env_modelforge_openmm.yaml
micromamba activate modelforge-openmm
```

Install `modelforge` and the OpenMM integration package:

```bash
pip install -e . --no-deps --config-settings editable-mode=strict
pip install -e modelforge-openmm --no-deps --config-settings editable-mode=strict
```

Note: Test environments in `devtools/conda-envs/test_env*.yaml` are intended for CI/development testing and are intentionally separate from runtime/example environments.


### Copyright

Copyright (c) 2023-2026, Chodera Lab https://www.choderalab.org/


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

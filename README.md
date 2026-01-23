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

Installation
------------

To set up a development environment, clone the repository and install the package in editable mode:
```bash
git clone https://github.com/choderalab/modelforge.git
```
navigate to the modelforge root directory:
```bash
cd modelforge
```

Create a micromamba environment (or similar package manager) and activate it:
```bash
micromamba create -n modelforge -f devtools/conda-envs/test_env.yaml
micromamba activate modelforge
``` 

install via pip in editable mode:
```bash

pip install -e . --config-settings editable-mode=strict
```

Install the curate package:
```bash
pip install -e modelforge-curate --config-settings editable-mode=strict
```



### Copyright

Copyright (c) 2023-2026, Chodera Lab https://www.choderalab.org/


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

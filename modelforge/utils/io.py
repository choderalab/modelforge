""" Module for handling importing external libraries that are optional dependencies.

The general approach here is to wrap an import in a try/except structure, where failure
to import a module results in a descriptive message being printed to the console, e.g.,
providing a description of how to install.

This is useful for cases where a module is not required for the core functionality of the code,
and thus is not included as a base dependency in conda.  This is also useful for helping users install
packages that are only available via pip (and thus would not be installed in the conda release).

The approach taken here is adapted from MDTraj's delay_import module.
MDTraj is released under GNU 2.1 license; see <http://www.gnu.org/licenses/>

"""

import importlib
import inspect
import textwrap

MESSAGES = dict()

MESSAGES[
    "jax"
] = """
The code at {filename}:{line_number} requires the "jax" package.

jax is a library for numerical computing that is designed for high-performance machine learning research.

jax can be installed via conda:

    conda install -c conda-forge jax

To install on a machine with an NVIDIA GPU, use:
    
    conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia

"""

MESSAGES[
    "pytorch2jax"
] = """
The code at {filename}:{line_number} requires the "pytorch2jax" package.

Pytorch2Jax is a small Python library that provides functions to wrap PyTorch models into Jax functions and Flax modules. 

Pytorch2Jax can be installed via pip:
    
        pip install pytorch2jax
    
    """

MESSAGES[
    "flax"
] = """
The code at {filename}:{line_number} requires the "flax" package.

Flax: A neural network library and ecosystem for JAX designed for flexibility.

Flax can be installed via conda:
    
        conda install conda-forge::flax
    
    """

MESSAGES[
    "ray"
] = """
The code at {filename}:{line_number} requires the "ray" package.

Ray is an open-source unified framework for scaling AI and Python applications like machine learning.

On Linux, Ray can be install via conda:

    conda install -c conda-forge ray-all
    
"""

MESSAGES[
    "torchviz"
] = """
PyTorchViz is a small package to create visualizations of PyTorch execution graphs and traces.
https://github.com/szagoruyko/pytorchviz

PyTorchViz can be installed via pip:

    pip install torchviz

"""

MESSAGES[
    "qcelemental"
] = """
QCElemental is a resource module for quantum chemistry containing physical constants and periodic table data from NIST and molecule handlers.

https://molssi.github.io/QCElemental/

QCElemental can be installed via conda:

    conda install qcelemental -c conda-forge
"""

MESSAGES[
    "qcportal"
] = """
QCPortal is a data generation and retrieval platform specialized for quantum chemistry calculation.

https://molssi.github.io/QCFractal/

QCPortal can be installed via conda:
    
        conda install qcportal">=0.50" -c conda-forge

"""

MESSAGES[
    "rdkit"
] = """
RDKit is a collection of cheminformatics and machine learning tools.

https://www.rdkit.org/

RDKit can be installed via conda:

    conda install -c conda-forge rdkit
    
"""

MESSAGES[
    "retry"
] = """
Retry is a simple Python library for retrying failed operations.
    
Retry can be installed via conda:
    
        conda install -c conda-forge retry
    
"""

MESSAGES[
    "sqlitedict"
] = """

SqliteDict is a lightweight wrapper around Python's sqlite3 database.

SqliteDict can be installed via conda:
    
        conda install -c conda-forge sqlitedict
    
"""

MESSAGES[
    "wandb"
] = """

Weights and Biases is a tool for tracking and visualizing machine learning experiments.

Weights and Biases can be installed via conda:

    conda install conda-forge::wandb
    
"""
MESSAGES[
    "openmmtools"
] = """
A batteries-included toolkit for the GPU-accelerated OpenMM molecular simulation engine.
OpenMMTools can be installed via conda:
    conda install conda-forge::openmmtools
    
"""


def import_(module: str):
    """Import a module or print a descriptive message and raise an ImportError

    Parameters
    ----------
    module : str
        The name of the module to import.

    Returns
    -------
    module
        The imported module

    Raises
    ------
    ImportError
        If the module cannot be imported, a descriptive message is printed to the console
        and an ImportError is raised.

    Examples
    --------
    >>> from modelforge.utils.io import import_
    >>> jax = import_("jax")
    >>>
    >>> # to import a submodule, the following are equivalent:
    >>> from jax import numpy as jnp
    >>> jnp = import_("jax").numpy


    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        (
            frame,
            filename,
            line_number,
            function_name,
            lines,
            index,
        ) = inspect.getouterframes(inspect.currentframe())[1]

        basemodule = module.split(".")[0]

        if basemodule in MESSAGES:
            message = MESSAGES[basemodule].format(
                filename=filename, line_number=line_number
            )
        else:
            message = f" The code at {filename}:{line_number} requires the {module} package. Could not import {module}: {e}"

        message = textwrap.dedent(message)

        raise ImportError(message)


def check_import(module: str):
    """This tries to import a given module. If it fails, raise an error and provide a descriptive message.

    If successful, this will unload the module.

    This is useful to first check if a module has been installed, before trying to
    import specific submodules.

    Parameters
    ----------
    module : str
        The name of the module to import.

    Raises
    ------
    ImportError
        If the module cannot be imported, a descriptive message is printed to the console
        and an ImportError is raised.
    Examples
    --------
    >>> from modelforge.utils.io import check_import
    >>> check_import(module="ray")
    >>> from ray import tune
    """

    imported_module = import_(module)
    del imported_module


from typing import Union, List


def parse_devices(value: str) -> Union[int, List[int]]:
    """
    Parse the devices argument which can be either a single integer or a list of
    integers.

    Parameters
    ----------
    value : str
        The input string representing either a single integer or a list of
        integers.

    Returns
    -------
    Union[int, List[int]]
        Either a single integer or a list of integers.
    """
    import ast

    # if multiple comma delimited values are passed, split them into a list
    if value.startswith("[") and value.endswith("]"):
        # Safely evaluate the string as a Python literal (list of ints)
        return list(ast.literal_eval(value))
    else:
        return int(value)

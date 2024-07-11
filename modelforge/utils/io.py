""" Module for handling importing of external libraries.

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

jax can be installed via pip:

    pip install jax

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

Flax can be installed via pip:
    
        pip install flax
    
    """

MESSAGES[
    "ray"
] = """
The code at {filename}:{line_number} requires the "ray" package.

Ray is an open-source unified framework for scaling AI and Python applications like machine learning.

Ray can be installed via pip:
    
    pip install -U "ray[data,train,tune,serve]"
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
    >>> from modelforge.utils.package_import import import_
    >>> jax = import_("jax")


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

        if module in MESSAGES:
            message = MESSAGES[module].format(
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
    >>> from modelforge.utils.package_import import check_import
    >>> check_import(module="ray")
    >>> from ray import tune
    """

    imported_module = import_(module)
    del imported_module

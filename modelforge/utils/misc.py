"""
Module of miscellaneous utilities.
"""

import torch
from loguru import logger


class Welford:
    def __init__(self):
        """
        Implements Welford's online algorithm for computing running variance
        and standard deviation incrementally.

        This class maintains the count, mean, and M2 sufficient statistics,
        which is used to calculate variance and standard deviation on the fly
        as new samples are added via the update() method.

        The running variance/stddev can be retrieved via the variance/stddev properties.
        """
        self._n = 0
        self._mean = 0
        self._M2 = 0

    def update(self, batch: torch.Tensor) -> None:
        """
        Updates the running mean and variance statistics with a new batch of data.

        The mean and sum of squared differences from the mean (M2) are updated with the new batch of data.
        Parameters
        ----------
        batch: torch.Tensor, required
            Batch of data to be added to the running statistics.
        """
        # Convert batch to a numpy array to handle both scalars and arrays
        batch_size = len(batch)

        new_mean = torch.mean(batch)
        new_M2 = torch.sum((batch - new_mean) ** 2)

        delta = new_mean - self._mean
        self._mean += delta * batch_size / (self._n + batch_size)
        self._M2 += new_M2 + delta**2 * self._n * batch_size / (self._n + batch_size)
        self._n += batch_size

    @property
    def variance(self) -> torch.Tensor:
        """
        Returns the running variance calculated from the mean and M2 statistics.
        """
        return self._M2 / self._n if self._n > 1 else 0

    @property
    def stddev(self) -> torch.Tensor:
        return torch.sqrt(self.variance)

    @property
    def mean(self) -> torch.Tensor:
        """Returns the current mean."""
        return self._mean


def list_files(directory: str, extension: str) -> list:
    """
    Returns a list of files in a directory with a given extension.

    Parameters
    ----------
    directory: str, required
        Directory of interest.
    extension: str, required
        Only consider files with this given file extension

    Returns
    -------
    list
        List of files in the given directory with desired extension.

    Examples
    --------
    List only the xyz files in a test_directory
    >>> files = list_files('test_directory', '.xyz')
    """
    import os

    if not os.path.exists(directory):
        raise Exception(f"{directory} not found")

    logger.debug(f"Gathering {extension} files in {directory}.")

    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files.append(file)
    files.sort()
    return files


def str_to_float(x: str) -> float:
    """
    Converts a string to a float, changing Mathematica style scientific notion to python style.

    For example, this will convert str(1*^-6) to float(1e-6).

    Parameters
    ----------
    x : str, required
        String to process.

    Returns
    -------
    float
        Float value of the string.

    Examples
    --------
    >>> output_float = str_to_float('1*^6')
    >>> output_float = str_to_float('10123.0')
    """

    xf = float(x.replace("*^", "e"))
    return xf


def extract_tarred_file(
    input_path_dir: str,
    file_name: str,
    output_path_dir: str,
    mode: str = "r:gz",
) -> None:
    """
    Extract the contents of the tar file.
    Supports extracting gz and bz compressed files as well via the mode argument.

    Parameters
    ----------
    input_path_dir: str, required
        Path to the directory that contains the tar file.
    file_name: str, required
        Name of the tar file.
    output_path_dir: str, required
        Path to the directory to extract the tar file to.
    mode: str, optional, default='r:gz'
        Mode to open the tar file. options are 'r:gz', 'r:bz2', 'r'

    Examples
    --------
    from modelforge.utils.misc import extract_tarred_file
    >>> extract_tarred_file('test_directory', 'test.tar.gz', 'output_directory')

    """

    import tarfile

    logger.debug(f"Extracting tarred file {input_path_dir}/{file_name}")

    tar = tarfile.open(f"{input_path_dir}/{file_name}", mode)
    tar.extractall(output_path_dir)
    tar.close()


def ungzip_file(input_path_dir: str, file_name: str, output_path_dir: str) -> None:
    import gzip
    import shutil

    assert file_name.endswith(".gz")
    outputfile_name = file_name.split(".gz")[0]
    with gzip.open(f"{input_path_dir}/{file_name}", "rb") as f_in:
        with open(f"{output_path_dir}/{outputfile_name}", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def lock_file(file_handle):
    """
    Locks the file stream for exclusive access using fcntl.

    Parameters
    ----------
    file_handle: file stream, required
        File stream to lock.

    Examples
    --------
    >>> from modelforge.utils.misc import lock_file
    >>>
    >>> with open('test.txt', 'r') as f:
    >>>    lock_file(f)
    """

    import fcntl

    fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)


def unlock_file(file_handle):
    """
    Unlocks the file stream using fcntl.

    Parameters
    ----------
    file_handle: file stream, required
        File stream to unlock.

    Examples
    --------
    >>> from modelforge.utils.misc import unlock_file
    >>>
    >>> with open('test.txt', 'r') as f:
    >>>    unlock_file(f)
    """

    import fcntl

    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def check_file_lock(file_handle):
    """
    Checks if the file stream is locked using fcntl.

    Parameters
    ----------
    file_handle: file stream, required
        File stream to check.

    Returns
    -------
    bool
        True if the file is locked, False if the file is unlocked.

    Examples
    --------
    >>> from modelforge.utils.misc import check_file_lock
    >>>
    >>> with open('test.txt', 'r') as f:
    >>>    is_locked = check_file_lock(f)
    """

    import fcntl

    try:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except:
        return True

    return False


class OpenWithLock:
    """
    Context manager for opening a file that also locks the file for exclusive access.

    This will automatically check if the file is locked by another process and wait until the lock is released.

    Parameters
    ----------
    file_path: str, required
        Path to the file to open.
    mode: str, optional, default='r'
        Specifies how to open the file, matching the python open function.
        Options are 'r', 'w', 'a', 'r+', 'w+', 'a+', 'rb', 'wb', 'ab', 'r+b', 'w+b', 'a+b'

    Examples
    --------
    >>> from modelforge.utils.misc import LockOpen
    >>>
    >>> with OpenWithLock('test.txt', 'r') as f:
    >>>    print(f.read())

    """

    def __init__(self, file_path: str, mode: str = "r"):
        import os

        # expand the file path in case we have a tilde or environment variables
        self._file_path = os.path.expanduser(os.path.expandvars(file_path))
        self._mode = mode
        self._file_handle = None

    def __enter__(self):
        # open the file
        self._file_handle = open(self._file_path, mode=self._mode)

        # check to see if the file is already locked
        if check_file_lock(self._file_handle):
            logger.debug(
                f"{self._file_path} in locked by another process; waiting until lock is released."
            )

        # try to lock the file; if the file is already locked, this will wait
        # until the file is released. I added helper function definitions that
        # call fcntl, as we might not always want to use a context manager and
        # to test the function in isolation

        lock_file(self._file_handle)

        # return the opened file stream
        return self._file_handle

    def __exit__(self, *args):
        # unlock the file and close the file stream
        unlock_file(self._file_handle)
        self._file_handle.close()


import os
from functools import wraps


def lock_with_attribute(attribute_name):
    """
    Decorator for locking a method using a lock file path stored in an instance
    attribute. The attribute is accessed on the instance (`self`) at runtime.

    Parameters
    ----------
    attribute_name : str
        The name of the instance attribute that contains the lock file path.

    Examples
    --------
    >>> from modelforge.utils.misc import lock_with_attribute
    >>>
    >>> class MyClass:
    >>>     def __init__(self, lock_file):
    >>>         self.method_lock = lock_file
    >>>
    >>>     @lock_with_attribute('method_lock')
    >>>     def critical_section(self):
    >>>         print("Executing critical section")
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieve the instance (`self`)
            instance = args[0]
            # Get the lock file path from the specified attribute
            lock_file_path = getattr(instance, attribute_name)

            with OpenWithLock(lock_file_path, "w+") as lock_file_handle:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def seed_random_number(seed: int):
    """
    Seed the random number generator for reproducibility.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    """
    import random

    random.seed(seed)

    import numpy as np

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_configs_into_pydantic_models(potential_name: str, dataset_name: str):
    from modelforge.tests.data import (
        potential_defaults,
        training_defaults,
        dataset_defaults,
        runtime_defaults,
    )
    from modelforge.utils.io import get_path_string
    import toml

    potential_path = (
        get_path_string(potential_defaults) + f"/{potential_name.lower()}.toml"
    )
    dataset_path = get_path_string(dataset_defaults) + f"/{dataset_name.lower()}.toml"
    training_path = get_path_string(training_defaults) + "/default.toml"
    runtime_path = get_path_string(runtime_defaults) + "/runtime.toml"

    training_config_dict = toml.load(training_path)
    dataset_config_dict = toml.load(dataset_path)
    potential_config_dict = toml.load(potential_path)
    runtime_config_dict = toml.load(runtime_path)

    potential_name = potential_config_dict["potential"]["potential_name"]

    from modelforge.potential import _Implemented_NNP_Parameters

    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )
    potential_parameters = PotentialParameters(**potential_config_dict["potential"])

    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.train.parameters import TrainingParameters, RuntimeParameters

    dataset_parameters = DatasetParameters(**dataset_config_dict["dataset"])
    training_parameters = TrainingParameters(**training_config_dict["training"])
    runtime_parameters = RuntimeParameters(**runtime_config_dict["runtime"])

    return {
        "potential": potential_parameters,
        "dataset": dataset_parameters,
        "training": training_parameters,
        "runtime": runtime_parameters,
    }

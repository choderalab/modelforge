import os
from loguru import logger


def mkdir(path: str) -> bool:
    """
    Will create a specified directory if it does not exist.

    This calls os.makedirs which will recursively create the entire path,
    constructing any missing subdirectories while creating the leaf in the hierarchy.

    Parameters
    ----------
    path: str, required
        Path to directory to create

    Returns
    -------
        bool,
            Return status: True if the directory needed to be created, False if it exists.
    Examples:
    >>> status = mkdir('/path/to/file')
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


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

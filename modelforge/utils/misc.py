from loguru import logger
import torch


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

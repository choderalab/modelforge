from abc import ABC, abstractmethod

from typing import Dict, List, Optional
from loguru import logger
from openff.units import unit


class DatasetCuration(ABC):
    """
    Abstract base class with routines to fetch and process a dataset into a curated hdf5 file.
    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_dir: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./datasets",
        version_select: str = "latest",
    ):
        """
        Sets input and output parameters.

        Parameters
        ----------
        hdf5_file_name: str, required
            Name of the hdf5 file that will be generated.
        output_file_dir: str, optional, default='./'
            Location to write the output hdf5 file.
        local_cache_dir: str, optional, default='./qm9_datafiles'
            Location to save downloaded dataset.
        version_select: str, optional, default='latest'
            Version of the dataset to use as defined in the associated yaml file.

        """
        import os

        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.version_select = version_select
        os.makedirs(self.local_cache_dir, exist_ok=True)

        # initialize parameter information
        self._init_dataset_parameters()

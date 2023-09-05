from loguru import logger
import os
from abc import ABC, abstractmethod

from typing import Optional
from openff.units import unit, Quantity
import pint

from modelforge.curation.utils import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class ANI1_curation(ABC):
    """
    Routines to fetch and process the ANI-1 dataset into a curated hdf5 file.

    This class serves as a baseclass;  children for ANI-1x and ANI-1cc will use the same
    basic routines, but primarily with the properties of interest function defined.


    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_path: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./AN1x_dataset",
        convert_units: Optional[bool] = False,
    ):
        self.hdf5_file_name = hdf5_file_name
        self.output_file_path = output_file_path
        self.local_cache_dir = local_cache_dir
        self.convert_units = convert_units

        self.n_cores = os.cpu_count()

        self.dataset_download_url = (
            "https://springernature.figshare.com/ndownloader/files/3195389"
        )

        # temporary for initial development and testing
        self.input_file_name = (
            "/Users/cri/Documents/Projects-msk/datasets/ani1x_raw/ani1x-release.h5"
        )

        self._properties_of_interest()

    def _iterate_dataset(self, name, i):
        data_array_temp = []

        n_configs = self.raw_data_dict[name]["coordinates"].shape[0]

        for i in range(n_configs):
            temp_data = {}
            temp_data["name"] = f"{name}_{i}"
            temp_data["atomic_numbers"] = self.raw_data_dict[name]["atomic_numbers"]

            for param_in, param_data in self.qm_parameters_of_interest.items():
                temp = self.raw_data_dict[name][param_in][i]
                if isinstance(temp, np.ndarray) or isinstance(temp, float):
                    if not np.isnan(temp).any():
                        param_out = param_data["out_name"]
                        param_unit = param_data["u_in"]
                        if self.convert_units:
                            param_unit_out = param_data["u_out"]

                            temp_data[param_out] = (temp * param_unit).to(
                                param_unit_out, "chem"
                            )
                        else:
                            temp_data[param_out] = temp * param_unit

            # appending to lists is threadsafe
            # however order may be different, will need to sort
            self.data.append(temp_data)

    # recursive function to load up an entire dataset into memory
    def _load_full_hdf5(self, temp_dict):
        if isinstance(temp_dict, h5py.Dataset):
            return temp_dict[()]

        ret_temp_dict = {}
        for key, val in temp_dict.items():
            ret_temp_dict[key] = self._load_full_hdf5(val)
        return ret_temp_dict

    def _fetch_data_set(self):
        with h5py.File(self.input_file_name, "r") as f:
            self.raw_data_dict = self._load_full_hdf5(f)

    def _process_downloaded(
        self,
        local_path_to_tar: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information and writes an hdf5 file.

        Parameters
        ----------
        local_path_to_tar: str, required
            Path to the tar.bz2 file.
        name: str, required
            name of the tar.bz2 file,
        unit_testing_max_records: int, optional, default=None
            If set to an integer, 'n', the routine will only process the first 'n' records, useful for unit tests.

        Examples
        --------
        """

    self._fetch_data_set()
    logger.debug("fetched dataset")
    self.data = []

    completed = 0
    names = list(self.raw_data_dict.keys())
    threads = []
    self.results = []
    self.results1 = []
    self.results2 = []
    # for i in range(len(names)):
    #    self.results.append( 0)

    # seem need to reserve enough cores for i/o
    # will do some more checking on a machine with more cores
    n_workers = int(self.n_cores / 2) if int(self.n_cores / 2) >= 1 else 1

    with tqdm(total=len(names)) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i, name in enumerate(names):
                threads.append(executor.submit(self._iterate_dataset, name, i))

            for task in as_completed(threads):
                pbar.update(1)

    # because we don't guarantee the order we add to self.data, we need to sort by the name to ensure consistent order
    # which will be important for reproducibility.
    self.data = sorted(self.data, key=lambda x: x["name"])

    @abstractmethod
    def _properties_of_interest(self):
        pass

    def process(
        self,
        force_download: bool = False,
        unit_testing_max_records: Optional[int] = None,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        unit_testing_max_records: int, optional, default=None
            If set to an integer, 'n', the routine will only process the first 'n' records, useful for unit tests.

        Examples
        --------
        >>> ani1x_data = ANI_curation(hdf5_file_name='ani1x_dataset.hdf5', local_cache_dir='~/datasets/ani1x_dataset')
        >>> ani1x_data.process()

        """
        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_figshare(
            url=url,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )
        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from figshare.")
        self._process_downloaded(
            self.local_cache_dir, self.name, unit_testing_max_records
        )

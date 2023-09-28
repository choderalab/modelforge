from typing import Optional
from loguru import logger

from abc import ABC, abstractmethod


def dict_to_hdf5(file_name: str, data: list, series_info: dict, id_key: str) -> None:
    """
    Writes an hdf5 file from a list of dicts.

    This will include units, if provided as attributes and also denote

    Parameters
    ----------
    file_name: str, required
        Name and path of hdf5 file to write.
    data: list of dicts, required
        List that contains dictionaries of properties for each molecule to write to file.
    series_info: dict, required
        Defines whether a piece of data containers a series of data associated with different conformers.
        Dictionary keys match keys in dicts as part of data, where  "series" a
    id_key: str, required
        Name of the key in the dicts that uniquely describes each record.

    Examples
    --------
    >>> series = {'name':False, 'atomic_numbers': False, 'n_configs': False, 'geometry': True, 'energy':True}
    >>> dict_to_hdf5(file_name='qm9.hdf5', data=data, series_info=series, id_key='name')
    """

    import h5py
    from tqdm import tqdm
    import numpy as np
    import pint

    assert file_name.endswith(".hdf5")

    dt = h5py.special_dtype(vlen=str)

    with h5py.File(file_name, "w") as f:
        for datapoint in tqdm(data):
            try:
                record_name = datapoint[id_key]
            except Exception:
                print(f"id_key {id_key} not found in the data.")
            group = f.create_group(record_name)
            for key, val in datapoint.items():
                if key != id_key:
                    if isinstance(val, pint.Quantity):
                        val_m = val.m
                        val_u = str(val.u)
                    else:
                        val_m = val
                        val_u = None
                    if isinstance(val_m, str):
                        group.create_dataset(name=key, data=val_m, dtype=dt)
                    elif isinstance(val_m, (float, int)):
                        group.create_dataset(name=key, data=val_m)
                    elif isinstance(val_m, np.ndarray):
                        group.create_dataset(name=key, data=val_m, shape=val_m.shape)
                    if not val_u is None:
                        group[key].attrs["u"] = val_u
                    if series_info[key]:
                        group[key].attrs["series"] = True
                    else:
                        group[key].attrs["series"] = False


class dataset_curation(ABC):
    """
    Abstract base class with routines to fetch and process a dataset into a curated hdf5 file.
    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_dir: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./datasets",
        convert_units: Optional[bool] = True,
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
        convert_units: bool, optional, default=True
            Convert from e.g., source units [angstrom, hartree]
            to output units [nanometer, kJ/mol]
        """

        from modelforge.utils.misc import mkdir

        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        self.local_cache_dir = local_cache_dir
        self.convert_units = convert_units

        mkdir(self.local_cache_dir)

        # Overall list that will contain a dictionary for each record
        self.data = []

        # initialize parameter information
        self._init_dataset_parameters()
        self._init_record_entries_series()

    def _clear_data(self) -> None:
        """
        Clears the processed data from the list.

        """
        self.data = []

    def _generate_hdf5(self) -> None:
        """
        Creates an HDF5 file of the data at the path specified by output_file_path.

        """
        from modelforge.utils.misc import mkdir

        mkdir(self.output_file_dir)

        full_output_path = f"{self.output_file_dir}/{self.hdf5_file_name}"

        # generate the hdf5 file from the list of dicts
        logger.debug("Writing HDF5 file.")
        dict_to_hdf5(
            full_output_path,
            self.data,
            series_info=self._record_entries_series,
            id_key="name",
        )

    @abstractmethod
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
        >>> qm9_data = QM9_curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
        >>> qm9_data.process()

        """
        pass

    @abstractmethod
    def _init_dataset_parameters(self):
        # Define all relevant parameters here, such as unit conversion, metadata, data file source
        pass

    @abstractmethod
    def _init_record_entries_series(self):
        # For data efficiency, information for different conformers will be grouped together
        # To make it clear to the dataset loader which pieces of information are common to all
        # conformers, or which pieces encode the series, we will label each value.
        # The keys in this dictionary correspond to the label of the entries in each record.
        # The value indicates if the entry contains series data (True) or a single common entry (False).
        # If the entry has a value of True, the "series" attribute in hdf5 file will be set to True; False, if False.
        # This information will be used by the code to read in the datafile to know how to parse underlying records.
        # Example where the name and atomic numbers fields contain only a single common entry, but geometry
        # and energy are a series for each conformer:
        # self._record_entries_series = {'name':False, 'atomic_numbers': False, 'geometry': True, 'energy':True}

        self._record_entries_series = {}

    @abstractmethod
    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        pass

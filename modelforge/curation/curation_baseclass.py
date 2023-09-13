from typing import Optional
from loguru import logger

from abc import ABC, abstractmethod
import modelforge.curation.units


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

        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        self.local_cache_dir = local_cache_dir
        self.convert_units = convert_units

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
        from modelforge.curation.utils import dict_to_hdf5, mkdir

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
        # The keys in this dictionary correspond to the label of the entries in each record.
        # In this dictionary, the value indicates if the entry contains series data or just a single datapoint.
        # If the entry has a value of "series", the "series" attribute in hdf5 file will be set to True (false if single)
        # This information will be used by the code to read in the datafile to know how to parse underlying records.
        # While we could create separate records for every configuration, this vastly increases the time for generating
        # and reading hdf5 files.

        self._record_entries_series = {}

    @abstractmethod
    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        pass

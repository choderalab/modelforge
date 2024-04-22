from abc import ABC, abstractmethod

from typing import Dict, List, Optional
from loguru import logger
from openff.units import unit


def dict_to_hdf5(
    file_name: str, data: List[dict], series_info: Dict[str, str], id_key: str
) -> None:
    """
    Writes an hdf5 file from a list of dicts.

    This will include units as attributes for each quantity (if defined as openff-units quantities ) and also will
    include a 'format' attribute for each quantity that indicates whether the quantity is a single value, a series
    (i.e., values are per conformer), and if the value is per atom, per molecule, or a scalar/string for the record.

    Parameters
    ----------
    file_name: str, required
        Name and path of hdf5 file to write.
    data: list of dicts, required
        List that contains dictionaries of properties for each molecule to write to file.
    series_info: dict, required
        Defines whether a piece of data containers a series of data associated with different conformers
        and/or per-atom or per-molecule quantitites.
        Options in dictionary include 'single_rec', 'single_atom', 'single_mol', 'series_atom', 'series_mol'.
    id_key: str, required
        Name of the key in the dicts that uniquely describes each record.

    Examples
    --------
    >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
    ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
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
            except KeyError:
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
                    else:
                        raise ValueError(f"Type {type(val_m)} not recognized.")
                    if not val_u is None:
                        group[key].attrs["u"] = val_u

                    group[key].attrs["format"] = series_info[key]


class DatasetCuration(ABC):
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
            Convert from [e.g., angstrom, bohr, hartree] (i.e., source units)
            to [nanometer, kJ/mol] (i.e., target units)
        """
        import os

        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        self.local_cache_dir = local_cache_dir
        self.convert_units = convert_units

        os.makedirs(self.local_cache_dir, exist_ok=True)

        # Overall list that will contain a dictionary for each record
        self.data = []

        # initialize parameter information
        self._init_dataset_parameters()
        self._init_record_entries_series()

    def _clear_data(self) -> None:
        """
        Clears the processed data from the list.

        """
        self.data.clear()

    def _generate_hdf5(self) -> None:
        """
        Creates an HDF5 file of the data at the path specified by output_file_path.

        """
        import os

        os.makedirs(self.output_file_dir, exist_ok=True)

        full_output_path = f"{self.output_file_dir}/{self.hdf5_file_name}"

        # generate the hdf5 file from the list of dicts
        logger.debug("Writing HDF5 file.")
        dict_to_hdf5(
            full_output_path,
            self.data,
            series_info=self._record_entries_series,
            id_key="name",
        )

    def _convert_units(self):
        """
        Converts the units of properties in self.data to desired output values.

        """
        import pint

        # this is needed for the "chem" context to convert hartrees to kj/mol
        from modelforge.utils.units import chem_context

        for datapoint in self.data:
            for key, val in datapoint.items():
                if isinstance(val, pint.Quantity):
                    try:
                        datapoint[key] = val.to(
                            self.qm_parameters[key]["u_out"], "chem"
                        )
                    except:
                        # if the unit conversion can't be done
                        raise Exception(
                            f"could not convert {key} with unit {val.u} to {self.qm_parameters[key]['u_out']}"
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
        """
        Init the dictionary that defines the format of the data.

        For data efficiency, information for different conformers will be grouped together
        To make it clear to the dataset loader which pieces of information are common to all
        conformers or which quantities are series (i.e., have different values for each conformer).
        These labels will also allow us to define whether a given entry is per-atom, per-molecule,
        or is a scalar/string that applies to the entire record.
        Options include:
        single_rec, e.g., name, n_configs, smiles
        single_atom, e.g., atomic_numbers (these are the same for all conformers)
        series_atom, e.g., charges
        series_mol, e.g., dft energy, dipole moment, etc.
        These ultimately appear under the "format" attribute in the hdf5 file.

        Examples
        >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
                      ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
        """
        self._record_entries_series = {}
        pass

    @abstractmethod
    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        pass

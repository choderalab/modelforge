from loguru import logger
import os

from typing import Optional
from openff.units import unit, Quantity
import pint
import h5py
from tqdm import tqdm

from modelforge.curation.utils import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class ANI1_curation:
    """
    Routines to fetch and process the ANI-1x dataset into a curated hdf5 file.

    Parameters
    ----------
    hdf5_file_name, str, required
        name of the hdf5 file generated for the ANI-1x dataset
    output_file_dir: str, optional, default='./'
        Path to write the output hdf5 files.
    local_cache_dir: str, optional, default='./AN1_dataset'
        Location to save downloaded dataset.
    convert_units: bool, optional, default=True
        Convert from [angstrom, hartree] (i.e., source units)
        to [nanometer, kJ/mol]

    Examples
    --------
    >>> ani1_data = ANI1_curation(hdf5_file_name='ani1x_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/ani1x_dataset')
    >>> ani1_data.process()

    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_dir: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./AN1_dataset",
        convert_units: Optional[bool] = True,
    ):
        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        self.local_cache_dir = local_cache_dir
        self.convert_units = convert_units

        self.dataset_download_url = (
            "https://springernature.figshare.com/ndownloader/files/18112775"
        )

        # temporary for initial development and testing
        self.input_file_name = (
            "/Users/cri/Documents/Projects-msk/datasets/ani1x_raw/ani1x-release.h5"
        )

        # list of data
        self.data = []

        self.record_entries_series = {
            "name": "single",
            "atomic_numbers": "single",
            "n_configs": "single",
            "geometry": "series",
            "wb97x_dz.energy": "series",
            "wb97x_tz.energy": "series",
            "ccsd(t)_cbs.energy": "series",
            "hf_dz.energy": "series",
            "hf_tz.energy": "series",
            "hf_qz.energy": "series",
            "npno_ccsd(t)_dz.corr_energy": "series",
            "npno_ccsd(t)_tz.corr_energy": "series",
            "tpno_ccsd(t)_dz.corr_energy": "series",
            "mp2_dz.corr_energy": "series",
            "mp2_tz.corr_energy": "series",
            "mp2_qz.corr_energy": "series",
            "wb97x_dz.forces": "series",
            "wb97x_tz.forces": "series",
            "wb97x_dz.dipole": "series",
            "wb97x_tz.dipole": "series",
            "wb97x_dz.quadrupole": "series",
            "wb97x_dz.cm5_charges": "series",
            "wb97x_dz.hirshfeld_charges": "series",
            "wb97x_tz.mbis_charges": "series",
            "wb97x_tz.mbis_dipoles": "series",
            "wb97x_tz.mbis_quadrupoles": "series",
            "wb97x_tz.mbis_octupoles": "series",
            "wb97x_tz.mbis_volumes": "series",
        }

        self.qm_parameters = {
            "coordinates": {
                "u_in": unit.angstrom,
                "u_out": unit.nanometer,
            },
            "wb97x_dz.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "wb97x_tz.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "ccsd(t)_cbs.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "hf_dz.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "hf_tz.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "hf_qz.energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "npno_ccsd(t)_dz.corr_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "npno_ccsd(t)_tz.corr_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "tpno_ccsd(t)_dz.corr_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "mp2_dz.corr_energy": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "mp2_tz.corr_energy": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "mp2_qz.corr_energy": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "wb97x_dz.forces": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "wb97x_tz.forces": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "wb97x_dz.dipole": {
                "u_in": unit.elementary_charge * unit.angstrom,
                "u_out": unit.debye,
            },
            "wb97x_tz.dipole": {
                "u_in": unit.elementary_charge * unit.angstrom,
                "u_out": unit.debye,
            },
            "wb97x_dz.quadrupole": {
                "u_in": unit.hartree / unit.angstrom / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.angstrom / unit.angstrom,
            },
            "wb97x_dz.cm5_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "wb97x_dz.hirshfeld_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "wb97x_tz.mbis_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "wb97x_tz.mbis_dipoles": {
                "u_in": None,
                "u_out": None,
            },
            "wb97x_tz.mbis_quadrupoles": {
                "u_in": None,
                "u_out": None,
            },
            "wb97x_tz.mbis_octupoles": {
                "u_in": None,
                "u_out": None,
            },
            "wb97x_tz.mbis_volumes": {
                "u_in": None,
                "u_out": None,
            },
        }

    def _clear_data(self) -> None:
        """ "
        Clears out all processed data.
        """
        self.data = []

    def _generate_hdf5(self) -> None:
        """
        Creates an HDF5 file of the data at the path specified by output_file_dir.

        """
        mkdir(self.output_file_dir)

        full_output_path = f"{self.output_file_dir}/{self.hdf5_file_name}"

        # generate the hdf5 file from the list of dicts
        logger.debug("Writing HDF5 file.")
        dict_to_hdf5(
            full_output_path, self.data, self.record_entries_series, id_key="name"
        )

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        name: str, required
            Name of the raw hdf5 file,
        unit_testing_max_records: int, optional, default=None
            If set to an integer ('n') the routine will only process the first 'n' records; useful for unit tests.

        Examples
        --------
        """
        input_file_name = f"{local_path_dir}/{name}"

        with h5py.File(input_file_name, "r") as hf:
            names = list(hf.keys())
            if unit_testing_max_records is None:
                n_max = len(names)
            else:
                n_max = unit_testing_max_records

            for i, name in enumerate(names[0:n_max]):
                # Extract the total number of configurations for a given molecule
                n_configs = hf[name]["coordinates"].shape[0]

                keys_list = list(hf[name].keys())

                # temp dictionary for ANI-1x and ANI-1ccx data
                ani1x_temp = {}

                ani1x_temp["name"] = f"{name}"
                ani1x_temp["atomic_numbers"] = hf[name]["atomic_numbers"][()]
                ani1x_temp["n_configs"] = n_configs

                # param_in is the name of the entry, param_data contains input (u_in) and output (u_out) units
                for param_in, param_data in self.qm_parameters.items():
                    temp = hf[name][param_in][()]

                    # if not np.isnan(temp).any():

                    param_out = param_in
                    # we always want the particle positions to be called geometry
                    if param_in == "coordinates":
                        param_out = "geometry"

                    param_unit = param_data["u_in"]
                    if not param_unit is None:
                        if self.convert_units:
                            param_unit_out = param_data["u_out"]
                            try:
                                ani1x_temp[param_out] = (temp * param_unit).to(
                                    param_unit_out, "chem"
                                )

                            except Exception:
                                print(
                                    f"Could not convert {param_unit} to {param_unit_out} for {param_in}."
                                )
                        else:
                            ani1x_temp[param_out] = temp * param_unit
                    else:
                        ani1x_temp[param_out] = temp

                self.data.append(ani1x_temp)

        # From documentation: By default, objects inside group are iterated in alphanumeric order.
        # However, if group is created with track_order=True, the insertion order for the group is remembered (tracked)
        # in HDF5 file, and group contents are iterated in that order.
        # As such, we shouldn't need to do sort the objects to ensure reproducibility.
        # self.data = sorted(self.data, key=lambda x: x["name"])

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
        >>> ani1_data = ANI1_curation(hdf5_file_name='ani1x_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/ani1x_dataset')
        >>> ani1_data.process()

        """
        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_figshare(
            url=url,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )

        self._clear_data()

        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from figshare.")
        self._process_downloaded(
            self.local_cache_dir, self.name, unit_testing_max_records
        )

        self._generate_hdf5()

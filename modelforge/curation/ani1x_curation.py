from modelforge.curation.curation_baseclass import DatasetCuration
from modelforge.utils.units import *
from typing import Optional
from loguru import logger


class ANI1xCuration(DatasetCuration):
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
    >>> ani1_data = ANI1xCuration(hdf5_file_name='ani1x_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/ani1x_dataset')
    >>> ani1_data.process()

    """

    def _init_dataset_parameters(self):
        self.dataset_download_url = (
            "https://springernature.figshare.com/ndownloader/files/18112775"
        )

        self.qm_parameters = {
            "geometry": {
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

    def _init_record_entries_series(self):
        # For data efficiency, information for different conformers will be grouped together
        # To make it clear to the dataset loader which pieces of information are common to all
        # conformers, or which pieces encode the series, we will label each value.
        # The keys in this dictionary correspond to the label of the entries in each record.
        # The value indicates if the entry contains series data (True) or a single common entry (False).
        # If the entry has a value of True, the "series" attribute in hdf5 file will be set to True; False, if False.
        # This information will be used by the code to read in the datafile to know how to parse underlying records.

        self._record_entries_series = {
            "name": False,
            "atomic_numbers": False,
            "n_configs": False,
            "geometry": True,
            "wb97x_dz.energy": True,
            "wb97x_tz.energy": True,
            "ccsd(t)_cbs.energy": True,
            "hf_dz.energy": True,
            "hf_tz.energy": True,
            "hf_qz.energy": True,
            "npno_ccsd(t)_dz.corr_energy": True,
            "npno_ccsd(t)_tz.corr_energy": True,
            "tpno_ccsd(t)_dz.corr_energy": True,
            "mp2_dz.corr_energy": True,
            "mp2_tz.corr_energy": True,
            "mp2_qz.corr_energy": True,
            "wb97x_dz.forces": True,
            "wb97x_tz.forces": True,
            "wb97x_dz.dipole": True,
            "wb97x_tz.dipole": True,
            "wb97x_dz.quadrupole": True,
            "wb97x_dz.cm5_charges": True,
            "wb97x_dz.hirshfeld_charges": True,
            "wb97x_tz.mbis_charges": True,
            "wb97x_tz.mbis_dipoles": True,
            "wb97x_tz.mbis_quadrupoles": True,
            "wb97x_tz.mbis_octupoles": True,
            "wb97x_tz.mbis_volumes": True,
        }

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
        import h5py
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"

        with h5py.File(input_file_name, "r") as hf:
            names = list(hf.keys())
            if unit_testing_max_records is None:
                n_max = len(names)
            else:
                n_max = unit_testing_max_records

            for i, name in tqdm(enumerate(names[0:n_max]), total=n_max):
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
                    param_out = param_in
                    # we always want the particle positions to be called geometry to make life easier
                    if param_out == "geometry":
                        param_in = "coordinates"

                    temp = hf[name][param_in][()]

                    param_unit = param_data["u_in"]
                    if param_unit is not None:
                        ani1x_temp[param_out] = temp * param_unit
                    else:
                        ani1x_temp[param_out] = temp

                self.data.append(ani1x_temp)
        if self.convert_units:
            self._convert_units()
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
        >>> ani1_data = ANI1xCuration(hdf5_file_name='ani1x_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/ani1x_dataset')
        >>> ani1_data.process()

        """
        from modelforge.utils.remote import download_from_figshare

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

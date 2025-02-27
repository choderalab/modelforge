from modelforge.curation.curation_baseclass import DatasetCuration
from typing import Optional
from loguru import logger
from openff.units import unit


class ANI1xCuration(DatasetCuration):
    """
    Routines to fetch and process the ANI-1x dataset into a curated hdf5 file.

    This dataset includes ~5 million density function theory calculations
    for small organic molecules containing H, C, N, and O.
    A subset of ~500k are computed with accurate coupled cluster methods.

    References:

    ANI-1x dataset:
    Smith, J. S.; Nebgen, B.; Lubbers, N.; Isayev, O.; Roitberg, A. E.
    Less Is More: Sampling Chemical Space with Active Learning.
    J. Chem. Phys. 2018, 148 (24), 241733.
    https://doi.org/10.1063/1.5023802
    https://arxiv.org/abs/1801.09319

    ANI-1ccx dataset:
    Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E.
    Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential through Transfer Learning. N
    at. Commun. 2019, 10 (1), 2903.
    https://doi.org/10.1038/s41467-019-10827-4

    wB97x/def2-TZVPP data:
    Zubatyuk, R.; Smith, J. S.; Leszczynski, J.; Isayev, O.
    Accurate and Transferable Multitask Prediction of Chemical Properties with an Atoms-in-Molecules Neural Network.
    Sci. Adv. 2019, 5 (8), eaav6490.
    https://doi.org/10.1126/sciadv.aav6490


    Dataset DOI:
    https://doi.org/10.6084/m9.figshare.c.4712477.v1

    Parameters
    ----------
    hdf5_file_name, str, required
        name of the hdf5 file generated for the ANI-1x dataset
    output_file_dir: str, optional, default='./'
        Path to write the output hdf5 files.
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.


    Examples
    --------
    >>> ani1_data = ANI1xCuration(hdf5_file_name='ani1x_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/ani1x_dataset')
    >>> ani1_data.process()

    """

    def _init_dataset_parameters(self):

        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curation import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "ani1x_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["name"] == "ani1x"

        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Latest version: {self.version_select}")

        self.dataset_download_url = data_inputs[self.version_select][
            "dataset_download_url"
        ]
        self.dataset_md5_checksum = data_inputs[self.version_select][
            "dataset_md5_checksum"
        ]
        self.dataset_length = data_inputs[self.version_select]["dataset_length"]
        self.dataset_filename = data_inputs[self.version_select]["dataset_filename"]
        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
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
        single_mol, e.g., reference energy
        series_atom, e.g., charges
        series_mol, e.g., dft energy, dipole moment, etc.
        These ultimately appear under the "format" attribute in the hdf5 file.

        Examples
        >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
                      ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
        """

        self._record_entries_series = {
            "name": "single_rec",
            "atomic_numbers": "single_atom",
            "n_configs": "single_rec",
            "geometry": "series_atom",
            "wb97x_dz.energy": "series_mol",
            "wb97x_tz.energy": "series_mol",
            "ccsd(t)_cbs.energy": "series_mol",
            "hf_dz.energy": "series_mol",
            "hf_tz.energy": "series_mol",
            "hf_qz.energy": "series_mol",
            "npno_ccsd(t)_dz.corr_energy": "series_mol",
            "npno_ccsd(t)_tz.corr_energy": "series_mol",
            "tpno_ccsd(t)_dz.corr_energy": "series_mol",
            "mp2_dz.corr_energy": "series_mol",
            "mp2_tz.corr_energy": "series_mol",
            "mp2_qz.corr_energy": "series_mol",
            "wb97x_dz.forces": "series_atom",
            "wb97x_tz.forces": "series_atom",
            "wb97x_dz.dipole": "series_mol",
            "wb97x_tz.dipole": "series_mol",
            "wb97x_dz.quadrupole": "series_mol",
            "wb97x_dz.cm5_charges": "series_atom",
            "wb97x_dz.hirshfeld_charges": "series_atom",
            "wb97x_tz.mbis_charges": "series_atom",
            "wb97x_tz.mbis_dipoles": "series_atom",
            "wb97x_tz.mbis_quadrupoles": "series_atom",
            "wb97x_tz.mbis_octupoles": "series_atom",
            "wb97x_tz.mbis_volumes": "series_atom",
        }

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        name: str, required
            Name of the raw hdf5 file,
        max_records: int, optional, default=None
            If set to an integer, 'n_r', the routine will only process the first 'n_r' records, useful for unit tests.
            Can be used in conjunction with max_conformers_per_record and total_conformers.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records and total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with max_records and max_conformers_per_record.

        Examples
        --------
        """
        import h5py
        from tqdm import tqdm
        from numpy import newaxis

        input_file_name = f"{local_path_dir}/{name}"

        add_new_axis = {
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
            "wb97x_dz.cm5_charges": True,
            "wb97x_dz.hirshfeld_charges": True,
            "wb97x_tz.mbis_charges": True,
            "wb97x_tz.mbis_dipoles": True,
            "wb97x_tz.mbis_quadrupoles": True,
            "wb97x_tz.mbis_octupoles": True,
            "wb97x_tz.mbis_volumes": True,
        }
        with h5py.File(input_file_name, "r") as hf:
            names = list(hf.keys())
            if max_records is None:
                n_max = len(names)
            elif max_records is not None:
                n_max = max_records

            conformers_counter = 0

            for i, name in tqdm(enumerate(names[0:n_max]), total=n_max):
                if total_conformers is not None:
                    if conformers_counter >= total_conformers:
                        break

                # Extract the total number of configurations for a given molecule

                if max_conformers_per_record is not None:
                    conformers_per_molecule = min(
                        hf[name]["coordinates"].shape[0], max_conformers_per_record
                    )
                else:
                    conformers_per_molecule = hf[name]["coordinates"].shape[0]

                if total_conformers is not None:
                    if conformers_counter + conformers_per_molecule > total_conformers:
                        conformers_per_molecule = total_conformers - conformers_counter

                n_configs = conformers_per_molecule

                keys_list = list(hf[name].keys())

                # temp dictionary for ANI-1x and ANI-1ccx data
                ani1x_temp = {}

                ani1x_temp["name"] = f"{name}"
                ani1x_temp["atomic_numbers"] = hf[name]["atomic_numbers"][()].reshape(
                    -1, 1
                )
                ani1x_temp["n_configs"] = n_configs

                # param_in is the name of the entry, param_data contains input (u_in) and output (u_out) units
                for param_in, param_data in self.qm_parameters.items():
                    param_out = param_in
                    # we always want the particle positions to be called geometry to make life easier
                    if param_out == "geometry":
                        param_in = "coordinates"

                    temp = hf[name][param_in][()]
                    if param_in in add_new_axis:
                        temp = temp[..., newaxis]

                    temp = temp[0:conformers_per_molecule]

                    param_unit = param_data["u_in"]
                    if param_unit is not None:
                        ani1x_temp[param_out] = temp * param_unit
                    else:
                        ani1x_temp[param_out] = temp

                self.data.append(ani1x_temp)
                conformers_counter += conformers_per_molecule

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
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        max_records: int, optional, default=None
            If set to an integer, 'n_r', the routine will only process the first 'n_r' records, useful for unit tests.
            Can be used in conjunction with max_conformers_per_record and total_conformers.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records and total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with max_records and max_conformers_per_record.

        Examples
        --------
        >>> ani1_data = ANI1xCuration(hdf5_file_name='ani1x_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/ani1x_dataset')
        >>> ani1_data.process()

        """
        if max_records is not None and total_conformers is not None:
            raise Exception(
                "max_records and total_conformers cannot be set at the same time."
            )

        from modelforge.utils.remote import download_from_url

        url = self.dataset_download_url

        # download the dataset
        download_from_url(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.dataset_filename,
            length=self.dataset_length,
            force_download=force_download,
        )

        self._clear_data()

        # process the rest of the dataset

        self._process_downloaded(
            self.local_cache_dir,
            self.dataset_filename,
            max_records=max_records,
            max_conformers_per_record=max_conformers_per_record,
            total_conformers=total_conformers,
        )

        self._generate_hdf5()

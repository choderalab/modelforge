from modelforge.curation.curation_baseclass import DatasetCuration
from modelforge.utils.units import chem_context
import numpy as np

from typing import Optional, List
from loguru import logger
from openff.units import unit


class tmQMCuration(DatasetCuration):
    """
    Routines to fetch and process the transition metal quantum mechanics (tmQM) dataset into a curated hdf5 file.

    The tmQM dataset contains the geometries and properties of 86,665 mononuclear complexes extracted from the
    Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large
    variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    Citation:

    David Balcells and Bastian Bjerkem Skjelstad,
    tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes
    Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146
    DOI: 10.1021/acs.jcim.0c01041

    Original dataset source: https://github.com/uiocompcat/tmQM

    forked to be able to create releases:  https://github.com/chrisiacovella/tmQM/

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

    Examples
    --------
    >>> tmQM_data = tmQMCuration(hdf5_file_name='tmQM_dataset.hdf5', local_cache_dir='~/datasets/tmQM_dataset')
    >>> tmQM_data.process()


    """

    def _init_dataset_parameters(self) -> None:
        """
        Define the key parameters for the QM9 dataset.
        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curation import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "tmqm_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "tmqm"

        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Latest version: {self.version_select}")

        self.dataset_download_url = data_inputs[self.version_select][
            "dataset_download_url"
        ]
        self.dataset_md5_checksum = data_inputs[self.version_select][
            "dataset_md5_checksum"
        ]
        self.dataset_filename = data_inputs[self.version_select]["dataset_filename"]
        self.dataset_length = data_inputs[self.version_select]["dataset_length"]

        self.extracted_filepath = data_inputs[self.version_select]["extracted_filepath"]
        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
        )

        # if convert_units is True, which it is by default
        # we will convert each input unit (key) to the following output units (val)

        self.qm_parameters = {
            "geometry": {
                "u_in": unit.angstrom,
                "u_out": unit.nanometer,
            },
            "total_charge": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "partial_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "metal_center_charge": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "electronic_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dispersion_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "total_energy": {"u_in": unit.hartree, "u_out": unit.kilojoule_per_mole},
            "energy_of_homo": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "energy_of_lumo": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "homo_lumo_gap": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dipole_moment": {
                "u_in": unit.debye,
                "u_out": unit.elementary_charge * unit.nanometer,
            },
            "polarizability": {
                "u_in": unit.bohr * unit.bohr * unit.bohr,
                "u_out": unit.nanometer * unit.nanometer * unit.nanometer,
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
        series_atom, e.g., charges
        series_mol, e.g., dft energy, dipole moment, etc.
        These ultimately appear under the "format" attribute in the hdf5 file.

        Examples
        >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
                      ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
        """

        self._record_entries_series = {
            "name": "single_rec",
            "n_configs": "single_rec",
            "CSD_code": "single_rec",
            "geometry": "series_atom",
            "atomic_numbers": "single_atom",
            "partial_charges": "series_atom",
            "electronic_energy": "series_mol",
            "dispersion_energy": "series_mol",
            "total_energy": "series_mol",
            "dipole_moment": "series_mol",
            "energy_of_homo": "series_mol",
            "energy_of_lumo": "series_mol",
            "homo_lumo_gap": "series_mol",
            "metal_center_charge": "series_mol",
            "total_charge": "series_mol",
            "polarizability": "series_mol",
            "spin_multiplicity": "series_mol",
            "stoichiometry": "single_rec",
            "metal_n_ligands": "single_mol",
        }

    def _parse_properties(self, line: str) -> dict:
        """
        Parses the line in the xyz file that contains property information.

        Properties
        ----------
        line: str, required
            String to parse that contains property information. The structure of this line
            following the description in the original manuscript (See tables 2 and 3).

        Returns
        -------
        dict
            Dictionary of properties, with units added when appropriate.
        """

        from modelforge.utils.misc import str_to_float

        temp_prop = line.split("|")
        temp_data_dict = {}
        for split1 in temp_prop:
            temp2 = split1.split("=")
            if len(temp2) == 2:
                temp_data_dict[temp2[0].strip()] = temp2[1].strip()

        labels = [
            "CSD_code",
            "q",
            "S",
            "Stoichiometry",
            "MND",
        ]

        data_temp = {}
        for label in labels:
            if label == "CSD_code" or label == "Stoichiometry":
                data_temp[label] = temp_data_dict[label]
            else:
                data_temp[label] = str_to_float(temp_data_dict[label])

        return data_temp

    def _parse_snapshot_data(self, snapshots, snapshots_charges) -> dict:

        from modelforge.utils.io import import_

        qcel = import_("qcelemental")
        from modelforge.utils.misc import str_to_float
        from tqdm import tqdm

        # set up a dictionary to make it easier to index into the right snapshot
        # we'll need to eventually convert to a list of dictionaries

        data_all_temp = {}
        # while we could probably zip the snapshots and charges, I haven't set up the code to
        # ensure the order of the xyz files that were read. So we'll loop over the snapshots
        # then over charges
        for snapshot in tqdm(snapshots):
            # temporary dictionary to store data for each snapshot
            data_temp = {}

            # line 1: provides the number of atoms
            n_atoms = int(snapshot[0])

            # line 2: provides properties that we will parse into a dict
            properties = self._parse_properties(snapshot[1])

            geometry = []
            atomic_numbers = []
            charges = []
            # Lines 3 to 3+n: loop over the atoms to get coordinates and charges
            for i in range(n_atoms):
                line = snapshot[2 + i]
                element, x, y, z = line.split()

                atomic_numbers.append(qcel.periodictable.to_atomic_number(element))
                temp = [
                    str_to_float(x),
                    str_to_float(y),
                    str_to_float(z),
                ]
                geometry.append(temp)

            # end of file, now parse the inputs

            data_temp["name"] = properties["CSD_code"]
            data_temp["n_configs"] = 1

            data_temp["total_charge"] = (
                np.array(properties["q"]).reshape(1, 1)
                * self.qm_parameters["total_charge"]["u_in"]
            )
            data_temp["spin_multiplicity"] = np.array(float(properties["S"])).reshape(
                1, 1
            )
            data_temp["stoichiometry"] = properties["Stoichiometry"]
            data_temp["metal_n_ligands"] = np.array(int(properties["MND"])).reshape(
                1, 1
            )

            data_temp["geometry"] = (
                np.array(geometry).reshape(1, -1, 3)
                * self.qm_parameters["geometry"]["u_in"]
            )
            # atomic_numbers are written as an [n,1] array
            data_temp["atomic_numbers"] = np.array(atomic_numbers).reshape(-1, 1)
            data_all_temp[data_temp["name"]] = data_temp

        for snapshot in tqdm(snapshots_charges):

            name = snapshot[0].split("|")[0].split("=")[1].strip()

            charges = []
            for i in range(1, len(snapshot) - 1):
                charges.append(str_to_float(snapshot[i].split()[1]))
            # charges are written as an [m,n,1] array; note m =1 in this case
            data_all_temp[name]["partial_charges"] = (
                np.array(charges).reshape(1, -1, 1)
                * self.qm_parameters["partial_charges"]["u_in"]
            )

            assert (
                data_all_temp[name]["partial_charges"].shape[1]
                == data_all_temp[name]["geometry"].shape[1]
            )

            # calculate the reference energy at 0K and 298.15K
            # this is done by summing the reference energies of the atoms
            # note this has units already attached

        return data_all_temp

    def _process_downloaded(
        self,
        local_path_dir: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        xyz_files: List[str] = None,
        q_files: List[str] = None,
        BO_files: List[str] = None,
        csv_files: List[str] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information into a list of dicts.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the tar.bz2 file.
        max_records: int, optional, default=None
            If set to an integer, 'n_r', the routine will only process the first 'n_r' records, useful for unit tests.
            Can be used in conjunction with umax_conformers_per_record and total_conformers.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records and total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with max_records and max_conformers_per_record.
        xyz_files: List[str], optional, default=None
            List of xyz files in the directory.
        q_files: List[str], optional, default=None
            List of q files in the directory.
        BO_files: List[str], optional, default=None
            List of BO files in the directory.
        csv_files: List[str], optional, default=None
            List of csv files in the directory.


        Examples
        --------

        """
        from tqdm import tqdm
        import csv

        # let us load up the csv file first

        # aggregate the snapshot contents into a list
        snapshots = []
        for xyz_file in xyz_files:

            with open(f"{local_path_dir}/{xyz_file}", "r") as f:

                lines = f.readlines()
                temp = []
                for line in lines:
                    if line != "\n":
                        temp.append(line.rstrip("\n"))
                    else:
                        snapshots.append(temp)
                        temp = []

        snapshot_charges = []
        for q_file in q_files:
            with open(f"{local_path_dir}/{q_file}", "r") as f:
                lines = f.readlines()
                temp = []
                for line in lines:
                    if line != "\n":
                        temp.append(line.rstrip("\n"))
                    else:
                        snapshot_charges.append(temp)
                        temp = []

        print(len(snapshot_charges[0]))
        snapshots_temp_dict = self._parse_snapshot_data(snapshots, snapshot_charges)

        columns = []
        csv_temp_dict = {}
        csv_input_file = f"{local_path_dir}/{csv_files[0]}"

        with open(csv_input_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    columns = row
                    line_count += 1
                else:
                    # CSD_code is the key
                    temp_dict = {columns[i]: row[i] for i in range(len(columns))}
                    name = temp_dict["CSD_code"]
                    snapshots_temp_dict[name]["electronic_energy"] = (
                        np.array(float(temp_dict["Electronic_E"])).reshape(1, 1)
                        * self.qm_parameters["electronic_energy"]["u_in"]
                    )
                    snapshots_temp_dict[name]["dispersion_energy"] = (
                        np.array(float(temp_dict["Dispersion_E"])).reshape(1, 1)
                        * self.qm_parameters["dispersion_energy"]["u_in"]
                    )
                    snapshots_temp_dict[name]["dipole_moment"] = (
                        np.array(float(temp_dict["Dipole_M"])).reshape(1, 1)
                        * self.qm_parameters["dipole_moment"]["u_in"]
                    )
                    snapshots_temp_dict[name]["metal_center_charge"] = (
                        np.array(float(temp_dict["Metal_q"])).reshape(1, 1)
                        * self.qm_parameters["metal_center_charge"]["u_in"]
                    )
                    snapshots_temp_dict[name]["energy_of_lumo"] = (
                        np.array(float(temp_dict["LUMO_Energy"])).reshape(1, 1)
                        * self.qm_parameters["energy_of_lumo"]["u_in"]
                    )

                    snapshots_temp_dict[name]["energy_of_homo"] = (
                        np.array(float(temp_dict["HOMO_Energy"])).reshape(1, 1)
                        * self.qm_parameters["energy_of_homo"]["u_in"]
                    )
                    snapshots_temp_dict[name]["homo_lumo_gap"] = (
                        np.array(float(temp_dict["HL_Gap"])).reshape(1, 1)
                        * self.qm_parameters["homo_lumo_gap"]["u_in"]
                    )
                    snapshots_temp_dict[name]["polarizability"] = (
                        np.array(float(temp_dict["Polarizability"])).reshape(1, 1)
                        * self.qm_parameters["polarizability"]["u_in"]
                    )

                    # csv_temp_dict[temp_dict["CSD_code"]] = temp_dict

                    line_count += 1
        data_temp = []
        for name in snapshots_temp_dict.keys():
            data_temp.append(snapshots_temp_dict[name])

        print(max_records, total_conformers)
        n_max = len(data_temp)
        if max_records is not None:
            n_max = max_records
        elif total_conformers is not None:
            n_max = total_conformers

        print(len(data_temp), n_max)
        self.data = data_temp[0:n_max]

        # if unit outputs were defined perform conversion
        if self.convert_units:
            self._convert_units()

        # When reading in the list of xyz files from the directory, we sort by name
        # so further sorting, like in the commented out below,
        # is likely not necessary unless we were to do asynchronous processing
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

        Note for qm9, only a single conformer is present per record, so max_records and total_conformers behave the same way,
        and max_conformers_per_record does not alter the behavior (i.e., it is always 1).

        Examples
        --------
        >>> tmQM_data = tmQMCuration(hdf5_file_name='tmQM_dataset.hdf5', local_cache_dir='~/datasets/tmQM_dataset')
        >>> tmQM_data.process()

        """
        if max_records is not None and total_conformers is not None:
            raise ValueError(
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
        # clear out the data array before we process
        self._clear_data()

        # untar the dataset
        from modelforge.utils.misc import extract_tarred_file

        # extract the tar.bz2 file into the local_cache_dir
        # creating a directory called qm9_xyz_files to hold the contents
        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.dataset_filename,
            output_path_dir=f"{self.local_cache_dir}/tmqm_files",
            mode="r:gz",
        )
        from modelforge.utils.misc import list_files, ungzip_file

        # list the files in the directory to examine
        files = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".gz",
        )
        for file in files:
            ungzip_file(
                input_path_dir=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
                file_name=file,
                output_path_dir=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            )

        xyz_files = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".xyz",
        )
        q_file = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".q",
        )
        BO_files = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".BO",
        )
        csv_file = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".csv",
        )
        print(xyz_files, q_file, BO_files, csv_file)

        self._process_downloaded(
            f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}/",
            max_records,
            max_conformers_per_record,
            total_conformers,
            xyz_files,
            q_file,
            BO_files,
            csv_file,
        )

        # generate the hdf5 file
        self._generate_hdf5()

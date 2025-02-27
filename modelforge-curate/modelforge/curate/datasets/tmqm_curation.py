from modelforge.curate import Record, SourceDataset
from modelforge.curate.datasets.curation_baseclass import DatasetCuration
from modelforge.curate.properties import (
    AtomicNumbers,
    Positions,
    Energies,
    Forces,
    PartialCharges,
    TotalCharge,
    MetaData,
    DipoleMomentScalarPerSystem,
    SpinMultiplicities,
)

from modelforge.utils.units import chem_context
import numpy as np

from typing import Optional, List
from loguru import logger
from openff.units import unit


class tmQMCuration(DatasetCuration):
    """
    Routines to fetch and process the transition metal quantum mechanics (tmQM) dataset into a curated hdf5 file.

    The tmQM dataset contains the geometries and properties of 108,541  mononuclear complexes extracted from the
    Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large
    variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    Note the original version of the dataset (marked old in the repo and described in the original paper)
    contains 86,665 complexes. The new version of the dataset contains 108,541 complexes by running the same criteria
    on a newer version of the CSD. The new version of the dataset is the one used in this curation.

    Citation:

    David Balcells and Bastian Bjerkem Skjelstad,
    tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes
    Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146
    DOI: 10.1021/acs.jcim.0c01041

    Original dataset source: https://github.com/uiocompcat/tmQM

    forked to be able to create releases:  https://github.com/chrisiacovella/tmQM/

    Parameters
    ----------
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.
    version_select: str, optional, default='latest'
        Version of the dataset to use as defined in the associated yaml file.

    Examples
    --------
    >>> tmQM_data = tmQMCuration(local_cache_dir='~/datasets/tmQM_dataset')
    >>> tmQM_data.process()
    >>> tmQM_data.to_hdf5(hdf5_file_name='tmQM_dataset.hdf5', output_file_dir='~/datasets/hdf5_files')

    """

    def _init_dataset_parameters(self) -> None:
        """
        Define the key parameters for the QM9 dataset.
        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curate.datasets import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "tmqm_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["name"] == "tmqm"

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

    def _process_downloaded(
        self,
        local_path_dir: str,
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
            Path to the directory that contains the downloaded dataset.
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
        from modelforge.dataset.utils import _ATOMIC_ELEMENT_TO_NUMBER
        from modelforge.utils.misc import str_to_float

        dataset = SourceDataset(
            dataset_name=self.dataset_name, local_db_dir=self.local_cache_dir
        )

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

        for snapshot in tqdm(snapshots):

            # line 1: provides the number of atoms
            n_atoms = int(snapshot[0])

            # line 2: provides properties that we will parse into a dict
            properties = self._parse_properties(snapshot[1])

            geometry = []
            atomic_numbers = []
            # Lines 3 to 3+n: loop over the atoms to get coordinates and atomic_element
            for i in range(n_atoms):
                line = snapshot[2 + i]
                element, x, y, z = line.split()

                atomic_numbers.append(_ATOMIC_ELEMENT_TO_NUMBER[element])
                temp = [
                    str_to_float(x),
                    str_to_float(y),
                    str_to_float(z),
                ]
                geometry.append(temp)

            # end of file, now parse the inputs
            record_name = properties["CSD_code"]
            dataset.create_record(record_name=record_name)

            positions = Positions(
                value=np.array(geometry).reshape(1, -1, 3), units=unit.angstrom
            )
            total_charge = TotalCharge(
                value=np.array(properties["q"]).reshape(1, 1),
                units=unit.elementary_charge,
            )
            atomic_numbers = AtomicNumbers(
                value=np.array(atomic_numbers).reshape(-1, 1)
            )

            stoichiometry = MetaData(
                name="stoichiometry", value=properties["Stoichiometry"]
            )
            spin_multiplicity = SpinMultiplicities(
                value=np.array(properties["S"]).reshape(1, 1)
            )
            metal_n_ligands = MetaData(
                name="metal_n_ligands", value=np.array(properties["MND"])
            )
            dataset.add_properties(
                record_name=record_name,
                properties=[
                    positions,
                    total_charge,
                    atomic_numbers,
                    stoichiometry,
                    spin_multiplicity,
                    metal_n_ligands,
                ],
            )

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

        for snapshot in tqdm(snapshot_charges):

            record_name = snapshot[0].split("|")[0].split("=")[1].strip()

            charges = []
            for i in range(1, len(snapshot) - 1):
                charges.append(str_to_float(snapshot[i].split()[1]))
            # charges are written as an [m,n,1] array; note m =1 in this case

            partial_charges = PartialCharges(
                value=np.array(charges).reshape(1, -1, 1), units=unit.elementary_charge
            )
            dataset.add_property(record_name=record_name, property=partial_charges)

        columns = []
        csv_temp_dict = {}
        csv_input_file = f"{local_path_dir}/{csv_files[0]}"

        with open(csv_input_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            line_count = 0
            for row in tqdm(
                csv_reader, desc="Processing csv file", total=len(snapshot_charges)
            ):
                if line_count == 0:
                    columns = row
                    line_count += 1
                else:
                    # CSD_code is the key
                    temp_dict = {columns[i]: row[i] for i in range(len(columns))}
                    record_name = temp_dict["CSD_code"]

                    electronic_energy = Energies(
                        name="electronic_energy",
                        value=np.array(str_to_float(temp_dict["Electronic_E"])).reshape(
                            1, 1
                        ),
                        units=unit.hartree,
                    )
                    dispersion_energy = Energies(
                        name="dispersion_energy",
                        value=np.array(str_to_float(temp_dict["Dispersion_E"])).reshape(
                            1, 1
                        ),
                        units=unit.hartree,
                    )
                    total_energy_temp = str_to_float(
                        temp_dict["Electronic_E"]
                    ) + str_to_float(temp_dict["Dispersion_E"])
                    total_energy = Energies(
                        name="total_energy",
                        value=np.array(total_energy_temp).reshape(1, 1),
                        units=unit.hartree,
                    )

                    dipole_moment_magnitude = DipoleMomentScalarPerSystem(
                        name="dipole_moment_magnitude",
                        value=np.array(float(temp_dict["Dipole_M"])).reshape(1, 1),
                        units=unit.debye,
                    )

                    dataset.add_properties(
                        record_name=record_name,
                        properties=[
                            electronic_energy,
                            dispersion_energy,
                            total_energy,
                            dipole_moment_magnitude,
                        ],
                    )

                    record_temp = dataset.get_record(record_name)

                    dipole_moment_computed_scaled = self.compute_dipole_moment(
                        atomic_numbers=record_temp.atomic_numbers,
                        partial_charges=record_temp.per_atom["partial_charges"],
                        positions=record_temp.per_atom["positions"],
                        dipole_moment_scalar=dipole_moment_magnitude,
                    )
                    dipole_moment_computed_scaled.name = "dipole_moment_computed_scaled"
                    dipole_moment_computed = self.compute_dipole_moment(
                        atomic_numbers=record_temp.atomic_numbers,
                        partial_charges=record_temp.per_atom["partial_charges"],
                        positions=record_temp.per_atom["positions"],
                    )
                    dipole_moment_computed.name = "dipole_moment_computed"

                    dataset.add_properties(
                        record_name=record_name,
                        properties=[
                            dipole_moment_computed,
                            dipole_moment_computed_scaled,
                        ],
                    )

                    metal_center_charge = MetaData(
                        name="metal_center_charge",
                        value=np.array(float(temp_dict["Metal_q"])).reshape(1, 1),
                        units=unit.elementary_charge,
                    )
                    energy_of_lumo = Energies(
                        name="energy_of_lumo",
                        value=np.array(float(temp_dict["LUMO_Energy"])).reshape(1, 1),
                        units=unit.hartree,
                    )
                    energy_of_homo = Energies(
                        name="energy_of_homo",
                        value=np.array(float(temp_dict["HOMO_Energy"])).reshape(1, 1),
                        units=unit.hartree,
                    )
                    homo_lumo_gap = Energies(
                        name="homo_lumo_gap",
                        value=np.array(float(temp_dict["HL_Gap"])).reshape(1, 1),
                        units=unit.hartree,
                    )

                    dataset.add_properties(
                        record_name=record_name,
                        properties=[
                            metal_center_charge,
                            energy_of_lumo,
                            energy_of_homo,
                            homo_lumo_gap,
                        ],
                    )
                    line_count += 1

        return dataset

    def process(
        self,
        force_download: bool = False,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.

        Examples
        --------
        >>> tmQM_data = tmQMCuration(ocal_cache_dir='~/datasets/tmQM_dataset')
        >>> tmQM_data.process()

        """

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

        # list the files in the directory that are gzipped
        gzip_files = list_files(
            directory=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            extension=".gz",
        )
        # ungzip the files
        for file in gzip_files:
            ungzip_file(
                input_path_dir=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
                file_name=file,
                output_path_dir=f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}",
            )

        # list the files in the directory by type
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

        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}/tmqm_files/{self.extracted_filepath}/",
            xyz_files,
            q_file,
            BO_files,
            csv_file,
        )

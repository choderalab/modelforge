from modelforge.curation.curation_baseclass import DatasetCuration
from modelforge.utils.units import chem_context
import numpy as np

from typing import Optional, List
from loguru import logger
from openff.units import unit


class QM9Curation(DatasetCuration):
    """
        Routines to fetch and process the QM9 dataset into a curated hdf5 file.

    The QM9 dataset includes 133,885 organic molecules with up to nine total heavy atoms (C,O,N,or F; excluding H).
        All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.

        Citation: Ramakrishnan, R., Dral, P., Rupp, M. et al.
                    "Quantum chemistry structures and properties of 134 kilo molecules."
                    Sci Data 1, 140022 (2014).
                    https://doi.org/10.1038/sdata.2014.22

        DOI for dataset: 10.6084/m9.figshare.c.978904.v5

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
        >>> qm9_data = QM9Curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
        >>> qm9_data.process()

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

        yaml_file = resources.files(yaml_files) / "qm9_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "qm9"

        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Latest version: {self.version_select}")

        self.dataset_download_url = data_inputs[self.version_select][
            "dataset_download_url"
        ]
        self.dataset_md5_checksum = data_inputs[self.version_select][
            "dataset_md5_checksum"
        ]
        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
        )

        # Below, we define key pieces of information related to the dataset in the form of a dict.
        # Metadata will be used to generate a README to go along with the HDF5 dataset.
        self.dataset_description = {
            "publication_doi": "10.1038/sdata.2014.22",
            "figshare_dataset_doi": "10.6084/m9.figshare.c.978904.v5",
            "figshare_dataset_url": "https://springernature.figshare.com/articles/dataset/Data_for_6095_constitutional_isomers_of_C7H10O2/1057646/2",
            "dataset_download_url": "https://springernature.figshare.com/ndownloader/files/3195389",
            "publication_citation": "Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014).",
            "dataset_citation": "Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014). Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5",
            "description": "QM9 Dataset: Includes 133,885 organic molecules with up to nine heavy atoms (CONF). All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.",
        }

        # if convert_units is True, which it is by default
        # we will convert each input unit (key) to the following output units (val)

        self.qm_parameters = {
            "geometry": {
                "u_in": unit.angstrom,
                "u_out": unit.nanometer,
            },
            "isotropic_polarizability": {
                "u_in": unit.angstrom**3,
                "u_out": unit.angstrom**3,
            },
            "rotational_constants": {
                "u_in": unit.gigahertz,
                "u_out": unit.gigahertz,
            },
            "charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "energy_of_homo": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "energy_of_lumo": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "lumo-homo_gap": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "zero_point_vibrational_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "internal_energy_at_0K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "formation_energy_at_0K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "reference_energy_at_0K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "internal_energy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "reference_energy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "enthalpy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "reference_enthalpy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "free_energy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "reference_free_energy_at_298.15K": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "electronic_spatial_extent": {
                "u_in": unit.angstrom**2,
                "u_out": unit.angstrom**2,
            },
            "dipole_moment": {"u_in": unit.debye, "u_out": unit.debye},
            "heat_capacity_at_298.15K": {
                "u_in": unit.calorie_per_mole / unit.kelvin,
                "u_out": unit.kilojoule_per_mole / unit.kelvin,
            },
            "harmonic_vibrational_frequencies": {
                "u_in": unit.cm**-1,
                "u_out": unit.cm**-1,
            },
        }
        # reference thermochemical data for each atom as provided in the atomref.txt file  provided in the publication
        # Direct link to the atomref.txt datafile: https://ndownloader.figstatic.com/files/3195395
        self.thermochemical_references = {
            "H": {
                "ZPVE": 0.000000 * unit.hartree,
                "U_0K": -0.500273 * unit.hartree,
                "U_298.15K": -0.498857 * unit.hartree,
                "H_298.15K": -0.497912 * unit.hartree,
                "G_298.15K": -0.510927 * unit.hartree,
                "Cv": 2.981 * unit.calorie_per_mole / unit.kelvin,
            },
            "C": {
                "ZPVE": 0.000000 * unit.hartree,
                "U_0K": -37.846772 * unit.hartree,
                "U_298.15K": -37.845355 * unit.hartree,
                "H_298.15K": -37.844411 * unit.hartree,
                "G_298.15K": -37.861317 * unit.hartree,
                "Cv": 2.981 * unit.calorie_per_mole / unit.kelvin,
            },
            "N": {
                "ZPVE": 0.000000 * unit.hartree,
                "U_0K": -54.583861 * unit.hartree,
                "U_298.15K": -54.582445 * unit.hartree,
                "H_298.15K": -54.581501 * unit.hartree,
                "G_298.15K": -54.598897 * unit.hartree,
                "Cv": 2.981 * unit.calorie_per_mole / unit.kelvin,
            },
            "O": {
                "ZPVE": 0.000000 * unit.hartree,
                "U_0K": -75.064579 * unit.hartree,
                "U_298.15K": -75.063163 * unit.hartree,
                "H_298.15K": -75.062219 * unit.hartree,
                "G_298.15K": -75.079532 * unit.hartree,
                "Cv": 2.981 * unit.calorie_per_mole / unit.kelvin,
            },
            "F": {
                "ZPVE": 0.000000 * unit.hartree,
                "U_0K": -99.718730 * unit.hartree,
                "U_298.15K": -99.717314 * unit.hartree,
                "H_298.15K": -99.716370 * unit.hartree,
                "G_298.15K": -99.733544 * unit.hartree,
                "Cv": 2.981 * unit.calorie_per_mole / unit.kelvin,
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
            "smiles_gdb-17": "single_rec",
            "smiles_b3lyp": "single_rec",
            "inchi_corina": "single_rec",
            "inchi_b3lyp": "single_rec",
            "geometry": "series_atom",
            "atomic_numbers": "single_atom",
            "charges": "series_atom",
            "idx": "single_rec",
            "rotational_constants": "series_mol",
            "dipole_moment": "series_mol",
            "isotropic_polarizability": "series_mol",
            "energy_of_homo": "series_mol",
            "energy_of_lumo": "series_mol",
            "lumo-homo_gap": "series_mol",
            "electronic_spatial_extent": "series_mol",
            "zero_point_vibrational_energy": "series_mol",
            "internal_energy_at_0K": "series_mol",
            "formation_energy_at_0K": "series_mol",
            "reference_energy_at_0K": "single_rec",
            "internal_energy_at_298.15K": "series_mol",
            "reference_energy_at_298.15K": "single_rec",
            "enthalpy_at_298.15K": "series_mol",
            "reference_enthalpy_at_298.15K": "single_rec",
            "free_energy_at_298.15K": "series_mol",
            "reference_free_energy_at_298.15K": "single_rec",
            "heat_capacity_at_298.15K": "series_mol",
            "harmonic_vibrational_frequencies": "series_mol",
        }

    def _calculate_reference_thermochemistry(
        self, molecule: List[str], thermo_key: str
    ):
        sum_of_energy = 0
        for atom in molecule:
            sum_of_energy += self.thermochemical_references[atom][thermo_key]

        return sum_of_energy

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

        temp_prop = line.split()

        # List of properties in the order they appear in the file and if they have units
        # This is used in parsing the properties line in the .xyz file
        labels = [
            "tag",
            "idx",
            "rotational_constant_A",
            "rotational_constant_B",
            "rotational_constant_C",
            "dipole_moment",
            "isotropic_polarizability",
            "energy_of_homo",
            "energy_of_lumo",
            "lumo-homo_gap",
            "electronic_spatial_extent",
            "zero_point_vibrational_energy",
            "internal_energy_at_0K",
            "internal_energy_at_298.15K",
            "enthalpy_at_298.15K",
            "free_energy_at_298.15K",
            "heat_capacity_at_298.15K",
        ]

        assert len(labels) == len(temp_prop)

        data_temp = {}
        for prop, label in zip(temp_prop, labels):
            if label == "tag" or label == "idx":
                data_temp[label] = prop
            else:
                data_temp[label] = str_to_float(prop)

        return data_temp

    def _parse_xyzfile(self, file_name: str) -> dict:
        """
        Parses the file containing information for each molecule.

        Structure of the file (based on tables 2 and 3 of the original manuscript)
        is included below.

        Parameters
        ----------
        file_name: str, required
            Name of the file to parse

        Returns
        -------
            dict:
                Dict of parsed properties.

        File format info
        ----------------

        Line            Content
        1               Number of atoms, n_a
        2               Scalar properties (see below)
        3,...mn_a+2     Element type, coords (x,y,z Ang) Mulliken partial charges (in e)
        n_a+3           Harmonic vibrational frequencies (3n_a-5 or 3n_a-6 in cm^-1)
        n_a+4           SMILES strings from GDB-17 and B3LYP relaxation
        n_a+5           InChI strings for Corina and B3LYP geometries

        Scalar properties:

        #   Unit        Description
        1   N/A         gdb9 string to facilitate extraction
        2   N/A         Consecutive, 1-based integer identifier
        3   GHz         Rotational constant A
        4   GHz         Rotational constant B
        5   GHz         Rotational constant C
        6   D           Dipole moment
        7   Ang^3       Isotropic polarizability
        8   Ha          Energy of HOMO
        9   Ha          Energy of LUMO
        10  Ha          LUMO-HOMO gap
        11  Ang^2       Electronic spatial extent
        12  Ha          Zero point vibrational energy
        13  Ha          Internal energy at 0K
        14  Ha          Internal energy at 298.15K
        15  Ha          Enthalpy at 298.15K
        16  Ha          Free energy at 298.15K
        17  cal/mol/K   Heat capacity at 298.15K

        """
        from modelforge.utils.io import import_

        qcel = import_("qcelemental")
        from modelforge.utils.misc import str_to_float

        with open(file_name, "r") as file:
            # temporary dictionary to store data for each file
            data_temp = {}

            # line 1: provides the number of atoms
            n_atoms = int(file.readline())

            # line 2: provides properties that we will parse into a dict
            properties_temp = file.readline()
            properties = self._parse_properties(properties_temp)

            # temporary lists
            elements = []
            atomic_numbers = []
            geometry = []
            charges = []
            hvf = []

            # Lines 3 to 3+n: loop over the atoms to get coordinates and charges
            for i in range(n_atoms):
                line = file.readline()
                element, x, y, z, q = line.split()
                elements.append(element)
                atomic_numbers.append(qcel.periodictable.to_atomic_number(element))
                temp = [
                    str_to_float(x),
                    str_to_float(y),
                    str_to_float(z),
                ]
                geometry.append(temp)
                charges.append([str_to_float(q)])

            # line 3+n+1: read harmonic_vibrational_frequencies
            hvf_temp = file.readline().split()

            # line 3+n+2: SMILES string
            smiles = file.readline().split()

            # line 3+n+3: inchi string
            InChI = file.readline()

            # end of file, now parse the inputs

            data_temp["name"] = file_name.split("/")[-1].split(".")[0]
            data_temp["n_configs"] = 1
            data_temp["smiles_gdb-17"] = smiles[0]
            data_temp["smiles_b3lyp"] = smiles[1]
            data_temp["inchi_corina"] = (
                InChI.split("\n")[0].split()[0].replace("InChI=", "")
            )
            data_temp["inchi_b3lyp"] = (
                InChI.split("\n")[0].split()[1].replace("InChI=", "")
            )
            data_temp["idx"] = properties["idx"]
            # even though we do not have multiple conformers, let us still define
            # geometry as [m,n,3], where number of conformers, m=1
            data_temp["geometry"] = (
                np.array(geometry).reshape(1, -1, 3)
                * self.qm_parameters["geometry"]["u_in"]
            )
            # atomic_numbers are written as an [n,1] array
            data_temp["atomic_numbers"] = np.array(atomic_numbers).reshape(-1, 1)
            # charges are written as an [m,n,1] array; note m =1 in this case
            data_temp["charges"] = (
                np.array(charges).reshape(1, -1, 1)
                * self.qm_parameters["charges"]["u_in"]
            )

            # remove the tag because it does not provide any useful information
            # also remove idx as we've already added it
            properties.pop("tag")
            properties.pop("idx")

            # merge rotational constants into a single energy
            data_temp["rotational_constants"] = self.qm_parameters[
                "rotational_constants"
            ]["u_in"] * np.array(
                [
                    properties["rotational_constant_A"],
                    properties["rotational_constant_B"],
                    properties["rotational_constant_C"],
                ]
            ).reshape(
                1, 3
            )

            properties.pop("rotational_constant_A")
            properties.pop("rotational_constant_B")
            properties.pop("rotational_constant_C")

            # loop over remaining properties and add to the dict
            # all properties are per-molecule, so array size will be [m,1], with m=1
            for property, val in properties.items():
                data_temp[property] = self.qm_parameters[property]["u_in"] * np.array(
                    val
                ).reshape(1, 1)

            # calculate the reference energy at 0K and 298.15K
            # this is done by summing the reference energies of the atoms
            # note this has units already attached
            U_ref_0K = self._calculate_reference_thermochemistry(elements, "U_0K")
            U_ref_298K = self._calculate_reference_thermochemistry(
                elements, "U_298.15K"
            )
            H_ref_298K = self._calculate_reference_thermochemistry(
                elements, "H_298.15K"
            )
            G_ref_298K = self._calculate_reference_thermochemistry(
                elements, "G_298.15K"
            )

            data_temp["reference_energy_at_0K"] = (
                np.array(U_ref_0K.m).reshape(1, 1) * U_ref_0K.u
            )
            data_temp["reference_energy_at_298.15K"] = (
                np.array(U_ref_298K.m).reshape(1, 1) * U_ref_298K.u
            )
            data_temp["reference_enthalpy_at_298.15K"] = (
                np.array(H_ref_298K.m).reshape(1, 1) * H_ref_298K.u
            )

            data_temp["reference_free_energy_at_298.15K"] = (
                np.array(G_ref_298K.m).reshape(1, 1) * G_ref_298K.u
            )

            data_temp["formation_energy_at_0K"] = (
                data_temp["internal_energy_at_0K"] - data_temp["reference_energy_at_0K"]
            )

            for h in hvf_temp:
                hvf.append(str_to_float(h))

            data_temp["harmonic_vibrational_frequencies"] = (
                np.array(hvf).reshape(1, -1)
                * self.qm_parameters["harmonic_vibrational_frequencies"]["u_in"]
            )

        return data_temp

    def _process_downloaded(
        self,
        local_path_dir: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
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


        Examples
        --------

        """
        from tqdm import tqdm
        from modelforge.utils.misc import list_files

        # list the files in the directory to examine
        files = list_files(directory=local_path_dir, extension=".xyz")

        # qm9 only has a single conformer in it, so unit_test_max_records and unit_testing_max_conformers_per_record behave the same way

        if max_records is None and total_conformers is None:
            n_max = len(files)
        elif max_records is not None and total_conformers is None:
            if max_records > len(files):
                n_max = len(files)
                logger.warning(
                    f"max_records ({max_records})is greater than the number of records in the dataset {len(files)}. Using {len(files)}."
                )
            else:
                n_max = max_records
        elif max_records is None and total_conformers is not None:
            if total_conformers > len(files):
                n_max = len(files)
                logger.warning(
                    f"total_conformers ({total_conformers}) is greater than the number of records in the dataset {len(files)}. Using {len(files)}."
                )
            else:
                n_max = total_conformers

        # we do not need to do anything check unit_testing_max_conformers_per_record because qm9 only has a single conformer per record
        if max_conformers_per_record is not None:
            logger.warning(
                "max_conformers_per_record is not used for QM9 dataset as there is only one conformer per record. Using a value of 1"
            )

        for i, file in enumerate(
            tqdm(files[0:n_max], desc="processing", total=len(files))
        ):
            data_temp = self._parse_xyzfile(f"{local_path_dir}/{file}")
            self.data.append(data_temp)

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
        >>> qm9_data = QM9Curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
        >>> qm9_data.process()

        """
        if max_records is not None and total_conformers is not None:
            raise ValueError(
                "max_records and total_conformers cannot be set at the same time."
            )

        from modelforge.utils.remote import download_from_figshare

        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_figshare(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )
        # clear out the data array before we process
        self._clear_data()

        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from figshare.")

        # untar the dataset
        from modelforge.utils.misc import extract_tarred_file

        # extract the tar.bz2 file into the local_cache_dir
        # creating a directory called qm9_xyz_files to hold the contents
        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.name,
            output_path_dir=f"{self.local_cache_dir}/qm9_xyz_files",
            mode="r:bz2",
        )

        self._process_downloaded(
            f"{self.local_cache_dir}/qm9_xyz_files",
            max_records,
            max_conformers_per_record,
            total_conformers,
        )

        # generate the hdf5 file
        self._generate_hdf5()

    def _generate_metadata(self):
        """
        Generates a metadata file to go along with the HDF5 dataset.


        """
        import pint

        with open(
            f"{self.output_file_dir}/{self.hdf5_file_name}.metadata", "w"
        ) as f_md:
            f_md.write("Dataset Description:\n")
            f_md.write(self.dataset_description["description"])
            f_md.write("\n\nPublication Citation:\n")
            f_md.write(self.dataset_description["publication_citation"])
            f_md.write("\n\nPublication DOI:\n")
            f_md.write(self.dataset_description["publication_doi"])
            f_md.write("\n\nSource dataset DOI:\n")
            f_md.write(self.dataset_description["figshare_dataset_url"])
            f_md.write("\n\nSource dataset download URL:\n")
            f_md.write(self.dataset_description["dataset_download_url"])

            f_md.write("\n\nHDF5 dataset curated by modelforge:\n")
            f_md.write(
                "The top level of the HDF5 file contains entries for each record name.\n"
            )
            f_md.write(
                "Each record contains the following data, where units, when appropriate, are stored as attributes.\n"
            )
            f_md.write("Unit naming conventions follow the openff-units package.\n\n")
            f_md.write("property : type : units\n")
            for key, val in self.data[0].items():
                if isinstance(val, pint.Quantity):
                    var_type = str(type(val.m).__name__)
                    f_md.write(f"{key} : {var_type} : {val.u}\n")
                else:
                    var_type = str(type(val).__name__)

                    f_md.write(f"{key} : {var_type} : N/A\n")

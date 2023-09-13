from loguru import logger
import os

from typing import Optional
from openff.units import unit, Quantity
import pint

from modelforge.curation.utils import *
import numpy as np

from modelforge.curation.curation_baseclass import *


class QM9_curation(dataset_curation):
    """
    Routines to fetch and process the QM9 dataset into a curated hdf5 file.

    The QM9 dataset includes 133,885 organic molecules with up to nine heavy atoms (CONF).
    All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.

    Citation: Ramakrishnan, R., Dral, P., Rupp, M. et al.
                Quantum chemistry structures and properties of 134 kilo molecules.
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
        Convert from e.g., source units [angstrom, hartree]
        to output units [nanometer, kJ/mol]

    Examples
    --------
    >>> qm9_data = QM9_curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
    >>> qm9_data.process()

    """

    def _init_dataset_parameters(self) -> None:
        """
        Define the key parameters for the QM9 dataset.
        """
        self.dataset_download_url = (
            "https://springernature.figshare.com/ndownloader/files/3195389"
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

        # if convert_units is True, we will convert each input unit (key) to the following output units (val)
        self.unit_output_dict = {
            unit.angstrom: unit.nanometer,
            unit.hartree: unit.kilojoule_per_mole,
            unit.calorie_per_mole / unit.kelvin: unit.kilojoule_per_mole / unit.kelvin,
            unit.angstrom**2: unit.angstrom**2,
            unit.angstrom**3: unit.angstrom**3,
            unit.debye: unit.debye,
            unit.elementary_charge: unit.elementary_charge,
            unit.cm**-1: unit.cm**-1,
            unit.gigahertz: unit.gigahertz,
        }

    def _init_record_entries_series(self):
        # The keys in this dictionary correspond to the label of the entries in each record.
        # In this dictionary, the value indicates if the entry contains series data or just a single datapoint.
        # If the entry has a value of "series", the "series" attribute in hdf5 file will be set to True (false if single)
        # This information will be used by the code to read in the datafile to know how to parse underlying records.
        # While we could create separate records for every configuration, this vastly increases the time for generating
        # and reading hdf5 files.
        self._record_entries_series = {
            "name": "single",
            "n_configs": "single",
            "smiles_gdb-17": "single",
            "smiles_b3lyp": "single",
            "inchi_corina": "single",
            "inchi_b3lyp": "single",
            "geometry": "single",
            "atomic_numbers": "single",
            "charges": "single",
            "idx": "single",
            "rotational_constant_A": "single",
            "rotational_constant_B": "single",
            "rotational_constant_C": "single",
            "dipole_moment": "single",
            "isotropic_polarizability": "single",
            "energy_of_homo": "single",
            "energy_of_lumo": "single",
            "lumo-homo_gap": "single",
            "electronic_spatial_extent": "single",
            "zero_point_vibrational_energy": "single",
            "internal_energy_at_0K": "single",
            "internal_energy_at_298.15K": "single",
            "enthalpy_at_298.15K": "single",
            "free_energy_at_298.15K": "single",
            "heat_capacity_at_298.15K": "single",
            "harmonic_vibrational_frequencies": "single",
        }

    def _extract(self, file_path: str, cache_directory: str) -> None:
        """
        Extract the contents of a tar.bz2 file.

        Parameters
        ----------
        file_path: str, required
            tar.bz2 to extract.
        cache_directory: str, required
            Location to save the contents from the tar.bz2 file
        """

        import tarfile

        logger.debug(f"Extracting tar {file_path}.")

        tar = tarfile.open(f"{file_path}", "r:bz2")
        tar.extractall(cache_directory)
        tar.close()

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

        temp_prop = line.split()

        # List of properties and their units in the order they appear in the file.
        # This is used in parsing the properties line in the .xyz file
        labels_and_units = [
            ("tag", None),
            ("idx", None),
            ("rotational_constant_A", unit.gigahertz),
            ("rotational_constant_B", unit.gigahertz),
            ("rotational_constant_C", unit.gigahertz),
            ("dipole_moment", unit.debye),
            ("isotropic_polarizability", unit.angstrom**3),
            ("energy_of_homo", unit.hartree),
            ("energy_of_lumo", unit.hartree),
            ("lumo-homo_gap", unit.hartree),
            ("electronic_spatial_extent", unit.angstrom**2),
            ("zero_point_vibrational_energy", unit.hartree),
            ("internal_energy_at_0K", unit.hartree),
            ("internal_energy_at_298.15K", unit.hartree),
            ("enthalpy_at_298.15K", unit.hartree),
            ("free_energy_at_298.15K", unit.hartree),
            ("heat_capacity_at_298.15K", unit.calorie_per_mole / unit.kelvin),
        ]

        assert len(labels_and_units) == len(temp_prop)

        data_temp = {}
        for prop, label_and_unit in zip(temp_prop, labels_and_units):
            label, prop_unit = label_and_unit
            if prop_unit is None:
                data_temp[label] = prop
            else:
                data_temp[label] = str_to_float(prop) * prop_unit
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
        import qcelemental as qcel

        with open(file_name, "r") as file:
            # read the first line that provides the number of atoms
            n_atoms = int(file.readline())

            # the second line provides properties that we will parse into a dict
            properties_temp = file.readline()
            properties = self._parse_properties(properties_temp)

            elements = []
            atomic_numbers = []
            geometry = []
            charges = []
            hvf = []

            geometry_temp = []
            charges_temp = []

            # loop over the atoms to get coordinates and charges
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
                charges.append(str_to_float(q))

            hvf_temp = file.readline().split()

            smiles = file.readline().split()
            InChI = file.readline()

            data_temp = {}
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
            data_temp["geometry"] = np.array(geometry) * unit.angstrom
            # Element symbols are converted to atomic numbers
            # Including an array of strings adds complications when writing the hdf5 file.
            # data["elements"] = np.array(elements, dtype=str)
            data_temp["atomic_numbers"] = np.array(atomic_numbers)
            data_temp["charges"] = np.array(charges) * unit.elementary_charge

            # remove the tag because it does not provide any useful information
            properties.pop("tag")

            # loop over remaining properties and add to the dict
            for property, val in properties.items():
                data_temp[property] = val

            for h in hvf_temp:
                hvf.append(str_to_float(h))

            data_temp["harmonic_vibrational_frequencies"] = np.array(hvf) / unit.cm

            # if unit outputs were defined perform conversion
            if self.convert_units:
                for key, val in data_temp.items():
                    if isinstance(val, pint.Quantity):
                        try:
                            data_temp[key] = val.to(
                                self.unit_output_dict[val.u], "chem"
                            )
                        except Exception:
                            try:
                                # if the unit conversion can't be done
                                print(
                                    f"could not convert {key} with unit {val.u} to {self.unit_output_dict[val.u]}"
                                )
                            except Exception:
                                print(
                                    f"could not convert {key} with unit {val.u}. {val.u} not in the defined unit conversions."
                                )

        return data_temp

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
            Path to the directory that contains the tar.bz2 file.
        name: str, required
            name of the tar.bz2 file,
        unit_testing_max_records: int, optional, default=None
            If set to an integer, 'n', the routine will only process the first 'n' records, useful for unit tests.

        Examples
        --------

        """
        from tqdm import tqdm

        # untar the dataset
        self._extract(
            file_path=f"{local_path_dir}/{name}",
            cache_directory=self.local_cache_dir,
        )

        # list the files in the directory to examine
        files = list_files(directory=self.local_cache_dir, extension=".xyz")
        if unit_testing_max_records is None:
            n_max = len(files)
        else:
            n_max = unit_testing_max_records

        for i, file in enumerate(
            tqdm(files[0:n_max], desc="processing", total=len(files))
        ):
            data_temp = self._parse_xyzfile(f"{self.local_cache_dir}/{file}")
            self.data.append(data_temp)

        # When reading in the list of xyz files from the directory, we sort by name
        # so this line is likely name necessary unless we were to do asynchronous processing
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
        >>> qm9_data = QM9_curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
        >>> qm9_data.process()

        """
        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_figshare(
            url=url,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )
        # clear out the data array before we process
        self._clear_data()

        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from figshare.")

        self._process_downloaded(
            self.local_cache_dir,
            self.name,
            unit_testing_max_records,
        )

        # generate the hdf5 file
        self._generate_hdf5()

    def _generate_metadata(self):
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

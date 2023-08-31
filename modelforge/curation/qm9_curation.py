import requests
from loguru import logger
import os
from tqdm import tqdm

from typing import Optional
from openff.units import unit, Quantity
import pint
import qcelemental as qcel

import tarfile
from modelforge.curation.utils import *
import numpy as np


class QM9_curation:
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
    output_file_path: str, optional, default='./'
        Path to write the output hdf5 file.
    local_cache_dir: str, optional, default='./qm9_datafiles'
        Location to save downloaded dataset.
    convert_units: bool, optional, default=True
        Convert from [angstrom, hartree] (i.e., source units)
        to [nanometer, kJ/mol]

    Examples
    --------
    >>> qm9_data = QM9_curation(hdf5_file_name='qm9_dataset.hdf5', local_cache_dir='~/datasets/qm9_dataset')
    >>> qm9_data.process()

    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_path: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./qm9_datafiles",
        convert_units: Optional[bool] = True,
    ):
        self.local_cache_dir = local_cache_dir
        self.output_file_path = output_file_path
        self.hdf5_file_name = hdf5_file_name
        self.convert_units = convert_units
        self.dataset_download_url = (
            "https://springernature.figshare.com/ndownloader/files/3195389"
        )
        # Below, we define key pieces of information related to the dataset in the form of a dict.
        # `dataset_download_url` is only the only variable used by the code to fetch the data.
        # All other data is metadata that will be used to generate a README to go along with
        # the HDF5 dataset and to document the key info within the code.
        self.dataset_description = {
            "publication_doi": "10.1038/sdata.2014.22",
            "figshare_dataset_doi": "10.6084/m9.figshare.c.978904.v5",
            "figshare_dataset_url": "https://springernature.figshare.com/articles/dataset/Data_for_6095_constitutional_isomers_of_C7H10O2/1057646/2",
            "dataset_download_url": "https://springernature.figshare.com/ndownloader/files/3195389",
            "publication_citation": "Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014).",
            "dataset_citation": "Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014). Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5",
            "description": "QM9 Dataset: Includes 133,885 organic molecules with up to nine heavy atoms (CONF). All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.",
        }
        # if convert_units is True we will
        # convert the following units
        self.unit_output_dict = {
            "geometry": unit.nanometer,
            "energy of homo": unit.kilojoule_per_mole,
            "energy of lumo": unit.kilojoule_per_mole,
            "gap": unit.kilojoule_per_mole,
            "zero point vibrational energy": unit.kilojoule_per_mole,
            "internal energy at 0K": unit.kilojoule_per_mole,
            "internal energy at 298.15K": unit.kilojoule_per_mole,
            "enthalpy at 298.15K": unit.kilojoule_per_mole,
            "free energy at 298.15K": unit.kilojoule_per_mole,
            "heat capacity at 298.15K": unit.kilojoule_per_mole / unit.kelvin,
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
        logger.debug(f"Extracting tar {file_path}.")

        tar = tarfile.open(f"{file_path}", "r:bz2")
        tar.extractall(cache_directory)
        tar.close()

    def _str_to_float(self, x: str) -> float:
        """
        Converts a string to float, changing Mathematica style scientific notion to python style.

        For example, this will convert str(1*^-6) to float(1e-6).

        Parameters
        ----------
        x : str, required
            String to process.

        Returns
        -------
        float
            Float value of the string.
        """
        xf = float(x.replace("*^", "e"))
        return xf

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
        # list of properties and their units in the order they appear in the file.

        labels_and_units = [
            ("tag", None),
            ("idx", None),
            ("rotational constant A", unit.gigahertz),
            ("rotational constant B", unit.gigahertz),
            ("rotational constant C", unit.gigahertz),
            ("dipole moment", unit.debye),
            ("isotropic polarizability", unit.angstrom**3),
            ("energy of homo", unit.hartree),
            ("energy of lumo", unit.hartree),
            ("gap", unit.hartree),
            ("electronic spatial extent", unit.angstrom**2),
            ("zero point vibrational energy", unit.hartree),
            ("internal energy at 0K", unit.hartree),
            ("internal energy at 298.15K", unit.hartree),
            ("enthalpy at 298.15K", unit.hartree),
            ("free energy at 298.15K", unit.hartree),
            ("heat capacity at 298.15K", unit.calorie_per_mole / unit.kelvin),
        ]

        assert len(labels_and_units) == len(temp_prop)

        data = {}
        for prop, label_and_unit in zip(temp_prop, labels_and_units):
            label, prop_unit = label_and_unit
            if prop_unit is None:
                data[label] = prop
            else:
                data[label] = self._str_to_float(prop) * prop_unit
        return data

    def _parse_xyzfile(self, file_name: str) -> dict:
        """
        Parses the file containing information for each molecule.

        Structure of the file (based on tables 2 and 3 of the original manuscript):


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

        with open(file_name, "r") as file:
            n_atoms = int(file.readline())
            properties_temp = file.readline()
            properties = self._parse_properties(properties_temp)
            elements = []
            atomic_numbers = []
            geometry = []
            charges = []
            hvf = []
            for i in range(n_atoms):
                line = file.readline()
                element, x, y, z, q = line.split()
                elements.append(element)
                atomic_numbers.append(qcel.periodictable.to_atomic_number(element))
                temp = [
                    self._str_to_float(x),
                    self._str_to_float(y),
                    self._str_to_float(z),
                ]
                geometry.append(temp)
                charges.append(self._str_to_float(q))

            hvf_temp = file.readline().split()

            smiles = file.readline().split()
            InChI = file.readline()

            data = {}
            data["name"] = file_name.split("/")[-1].split(".")[0]
            data["smiles gdb-17"] = smiles[0]
            data["smiles b3lyp"] = smiles[1]
            data["inchi Corina"] = InChI.split("\n")[0].split()[0].replace("InChI=", "")
            data["inchi B3LYP"] = InChI.split("\n")[0].split()[1].replace("InChI=", "")
            data["geometry"] = np.array(geometry) * unit.angstrom
            # Element symbols are converted to atomic numbers
            # including an array of strings causes complications
            # when writing the hdf5 file.
            # data["elements"] = np.array(elements, dtype=str)
            data["atomic numbers"] = np.array(atomic_numbers)
            data["charges"] = np.array(charges) * unit.elementary_charge

            # remove the tag because it does not provide any useful information
            properties.pop("tag")

            # loop over remaining properties and add to the dict
            for property, val in properties.items():
                data[property] = val

            for h in hvf_temp:
                hvf.append(self._str_to_float(h))

            data["harmonic vibrational frequencies"] = np.array(hvf) / unit.cm

            # if unit outputs were defined perform conversion
            if self.convert_units:
                for key in data.keys():
                    if key in self.unit_output_dict.keys():
                        try:
                            data[key] = data[key].to(self.unit_output_dict[key], "chem")
                        except Exception:
                            print(
                                f"could not convert {key} with units {key.u} to {self.unit_output_dict[key]}"
                            )

        return data

    def _process_downloaded(
        self, local_path_to_tar: str, name: str, unit_testing: bool
    ):
        # untar the dataset
        self._extract(
            file_path=f"{local_path_to_tar}/{name}",
            cache_directory=self.local_cache_dir,
        )

        # list the files in the directory to examine
        files = list_files(directory=self.local_cache_dir, extension=".xyz")

        # parse the information in each datat file, saving to a list of dicts, data
        self.data = []
        for i, file in enumerate(tqdm(files, desc="processing", total=len(files))):
            # first 10 records
            if unit_testing:
                if i > 9:
                    break

            data_temp = self._parse_xyzfile(f"{self.local_cache_dir}/{file}")
            self.data.append(data_temp)

        mkdir(self.output_file_path)

        full_output_path = f"{self.output_file_path}/{self.hdf5_file_name}"

        # generate the hdf5 file from the list of dicts
        logger.debug("Writing HDF5 file.")
        dict_to_hdf5(full_output_path, self.data, id_key="name")

    def process(self, force_download: bool = False, unit_testing: bool = False) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        unit_testing: bool, optional, default=False
            If True, only a subset (first 10 records) of the dataset will be used.
            Primarily meant to ensure unit tests can be completed in a reasonable time period.

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
        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from figshare.")
        self._process_downloaded(self.local_cache_dir, self.name, unit_testing)

    def _generate_metadata(self):
        with open(
            f"{self.output_file_path}/{self.hdf5_file_name}.metadata", "w"
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

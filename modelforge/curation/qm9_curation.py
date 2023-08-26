import requests
from loguru import logger
import os
from tqdm import tqdm

from openff.units import unit, Quantity
import pint
import qcelemental as qcel

import tarfile
from modelforge.curation.utils import dict_to_hdf5
import numpy as np


class QM9_curation:
    """
    Routines to fetch and process QM9 dataset.

    Parameters
    ----------
    hdf5_file_name: str, required
        Name of the hdf5 file that will be generated
    local_cache_dir: str, required, default=qm9_datafiles
        Location to save downloaded dataset.

    """

    def __init__(self, hdf5_file_name: str, local_cache_dir: str = "qm9_datafiles"):
        self.local_cache_dir = local_cache_dir
        self.hdf5_file_name = hdf5_file_name

        # define key pieces of information related to the dataset
        self.dataset_description = {
            "publication_doi": "10.1038/sdata.2014.22",
            "collection_doi": "10.6084/m9.figshare.c.978904.v5",
            "dataset_url": "https://springernature.figshare.com/articles/dataset/Data_for_6095_constitutional_isomers_of_C7H10O2/1057646/2",
            "dataset_download_url": "https://ndownloader.figshare.com/files/3195389",
            "dataset_filename": "dsgdb9nsd.xyz.tar.bz2",
            "publication_citation": """Ramakrishnan, R., Dral, P., Rupp, M. et al. 
                                        Quantum chemistry structures and properties of 134 kilo molecules. 
                                        Sci Data 1, 140022 (2014). 
                                        https://doi.org/10.1038/sdata.2014.22""",
            "dataset_citation": """Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014). 
                                    Quantum chemistry structures and properties of 134 kilo molecules. 
                                    figshare. Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5""",
            "description": """QM9 Dataset: Includes 133,885 organic molecules with up to nine heavy atoms (CONF). 
                                All properties were calculated at the B3LYP/6-31G(2df,p) level of quantum chemistry.""",
        }

    def _download(
        self, url: str, name: str, output_path: str, force_download=False
    ) -> None:
        """
        Downloads the dataset tar file from figshare.

        Parameters
        ----------
        url: str, required
            Figshare url to the data downloader
        name: str, required
            Name of the file downloaded
        output_path: str, required
            Location to download the file to.
        force_download: str, default=False
            If False, the file is not downloaded if it already exists in the directory.
            If True, the file will be downloaded even if it exists.

        """

        if not os.path.isfile(f"{output_path}/{name}") or force_download:
            logger.debug(f"Downloading datafile from figshare to {output_path}/{name}.")
            chunk_size = 512
            # get the head of the request
            head = requests.head(url)

            # because the url is calling a downloader, instead of the direct file
            # we can extract the file location and then fetch the length from this head
            # this is only useful for the download bar status
            temp_url = head.headers["location"].split("?")[0]
            length = int(requests.head(temp_url).headers["Content-Length"])

            r = requests.get(url, stream=True)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(f"{output_path}/{name}", "wb") as fd:
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    ascii=True,
                    desc="downloading",
                    total=(int(length / chunk_size) + 1),
                ):
                    fd.write(chunk)
        else:
            logger.debug("Datafile exists, using cached file.")

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
        Converts a string to float, fixing Mathematica style scientific notion.

        For example converts str(1*^-6) to float(1e-6).

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
        Parses the property line in the xyz file.

        Properties
        ----------
        line: str, required
            String to parse following the description in the original manuscript (See tables 2 and 3)

        Returns
        -------
        dict
            Dictionary of properties, with units added when appropriate.
        """

        temp_prop = line.split()
        # list of properties and their units
        labels_prop_units = [
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

        assert len(labels_prop_units) == len(temp_prop)

        data = {}
        for prop, label_prop_unit in zip(temp_prop, labels_prop_units):
            label, prop_unit = label_prop_unit
            if prop_unit is None:
                data[label] = prop
            else:
                data[label] = self._str_to_float(prop) * prop_unit
        return data

    def _parse_xyzfile(self, file_name: str) -> dict:
        """
        Parses the file containing information for each molecule.

        Parameters
        ----------
        file_name: str, required
            Name of the file to parse

        Return
        -------
            dict:
                Dict of parsed properties.

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
            data["inchi"] = InChI.split("\n")[0]
            data["geometry"] = np.array(geometry) * unit.angstrom
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
        return data

    def _list_files(self, directory: str, extension: str) -> list:
        """
        Returns a list of files in a directory with a given extension.

        Parameters
        ----------
        directory: str, required
            Directory of interest.
        extension: str, required
            Only consider files with this given file extension

        Returns
        -------
        list
            List of files in the given directory with desired extension.

        """

        logger.debug(f"Gathering xyz data files in {directory}.")

        files = []
        for file in os.listdir(directory):
            if file.endswith(extension):
                files.append(file)
        return files

    def process(self, force_download: bool = False, unit_testing: bool = False) -> None:
        """
        Downloads the dataset and extracts relevant information.

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
        > qm9_data = QM9_curation(local_cache_dir='~/datasets/qm9_dataset')
        > qm9_data.process()

        """
        name = self.dataset_description["dataset_filename"]
        url = self.dataset_description["dataset_download_url"]

        self._download(
            url=url,
            name=name,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )

        self._extract(
            file_path=f"{self.local_cache_dir}/{name}",
            cache_directory=self.local_cache_dir,
        )
        files = self._list_files(directory=self.local_cache_dir, extension=".xyz")

        self.data = []
        for i, file in enumerate(tqdm(files, desc="processing", total=len(files))):
            data_temp = self._parse_xyzfile(f"{self.local_cache_dir}/{file}")
            self.data.append(data_temp)
            if unit_testing:
                if i > 10:
                    break
        dict_to_hdf5(self.hdf5_file_name, self.data, id_key="name")

from modelforge.curate import Record, SourceDataset
from modelforge.curate.properties import (
    AtomicNumbers,
    DipoleMomentScalarPerSystem,
    Energies,
    MetaData,
    PropertyBaseModel,
    PartialCharges,
    Polarizability,
    Positions,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration

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

        Examples
        --------
        >>> qm9_data = QM9Curation(hdf5_file_name='qm9_dataset.hdf5',
        >>>                         output_file_dir='~/mf_datasets/hdf5_files',
        >>>                         local_cache_dir='~/mf_datasets/qm9_dataset')
        >>> qm9_data.process()

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
        self.dataset_filename = data_inputs[self.version_select]["dataset_filename"]
        self.dataset_length = data_inputs[self.version_select]["dataset_length"]

        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
        )

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

    def _parse_xyzfile(self, file_name: str) -> Record:
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
            Record:
                Record entry for the molecule

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

        from modelforge.dataset.utils import _ATOMIC_ELEMENT_TO_NUMBER
        from modelforge.utils.misc import str_to_float

        with open(file_name, "r") as file:
            # temporary dictionary to store data for each file

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
                atomic_numbers.append(_ATOMIC_ELEMENT_TO_NUMBER[element])
                temp = [
                    str_to_float(x),
                    str_to_float(y),
                    str_to_float(z),
                ]
                geometry.append(temp)
                charges.append([str_to_float(q)])

            # line 3+n+1: read harmonic_vibrational_frequencies
            hvf_temp = file.readline().split()
            for h in hvf_temp:
                hvf.append(str_to_float(h))

            # line 3+n+2: SMILES string
            smiles = file.readline().split()

            # line 3+n+3: inchi string
            InChI = file.readline()

            # end of file, now parse the inputs

            record_name = file_name.split("/")[-1].split(".")[0]
            record_temp = Record(record_name)

            # first parse metadata
            smiles1 = MetaData(name="smiles_gdb-17", value=smiles[0])
            smiles2 = MetaData(name="smiles_b3lyp", value=smiles[1])
            inchi1 = MetaData(
                name="inchi_corina",
                value=InChI.split("\n")[0].split()[0].replace("InChI=", ""),
            )
            inchi2 = MetaData(
                name="inchi_b3lyp",
                value=InChI.split("\n")[0].split()[1].replace("InChI=", ""),
            )
            idx = MetaData(name="idx", value=properties["idx"])
            tag = MetaData(name="tag", value=properties["tag"])

            record_temp.add_properties([smiles1, smiles2, inchi1, inchi2, idx, tag])

            positions = Positions(
                value=np.array(geometry).reshape(1, -1, 3),
                units="angstrom",
            )

            atomic_numbers = AtomicNumbers(
                value=np.array(atomic_numbers).reshape(-1, 1)
            )

            partial_charges = PartialCharges(
                value=np.array(charges).reshape(1, -1, 1),
                units="elementary_charge",
            )

            isotropic_polarizability = Polarizability(
                value=np.array(properties["isotropic_polarizability"]).reshape(1, 1),
                units="angstrom^3",
            )
            dipole_moment_scalar = DipoleMomentScalarPerSystem(
                value=np.array(properties["dipole_moment"]).reshape(1, 1),
                units="debye",
            )

            dipole_moment = self.compute_dipole_moment(
                atomic_numbers=atomic_numbers,
                partial_charges=partial_charges,
                positions=positions,
                dipole_moment_scalar=dipole_moment_scalar,
            )

            energy_of_homo = Energies(
                name="energy_of_homo",
                value=np.array(properties["energy_of_homo"]).reshape(1, 1),
                units=unit.hartree,
            )
            lumo_homo_gap = Energies(
                name="lumo-homo_gap",
                value=np.array(properties["lumo-homo_gap"]).reshape(1, 1),
                units=unit.hartree,
            )
            zero_point = Energies(
                name="zero_point_vibrational_energy",
                value=np.array(properties["zero_point_vibrational_energy"]).reshape(
                    1, 1
                ),
                units=unit.hartree,
            )
            internal_energy_at_0K = Energies(
                name="internal_energy_at_0K",
                value=np.array(properties["internal_energy_at_0K"]).reshape(1, 1),
                units=unit.hartree,
            )
            internal_energy_at_298K = Energies(
                name="internal_energy_at_298.15K",
                value=np.array(properties["internal_energy_at_298.15K"]).reshape(1, 1),
                units=unit.hartree,
            )
            enthalpy_at_298K = Energies(
                name="enthalpy_at_298.15K",
                value=np.array(properties["enthalpy_at_298.15K"]).reshape(1, 1),
                units=unit.hartree,
            )

            free_energy_at_298K = Energies(
                name="free_energy_at_298.15K",
                value=np.array(properties["free_energy_at_298.15K"]).reshape(1, 1),
                units=unit.hartree,
            )

            rotational_constants = PropertyBaseModel(
                name="rotational_constants",
                value=np.array(
                    [
                        properties["rotational_constant_A"],
                        properties["rotational_constant_B"],
                        properties["rotational_constant_C"],
                    ]
                ).reshape(1, 3),
                units=unit.gigahertz,
                property_type="frequency",
                classification="per_system",
            )

            harmonic_vibrational_frequencies = PropertyBaseModel(
                name="harmonic_vibrational_frequencies",
                value=np.array(hvf).reshape(1, -1),
                units=unit.cm**-1,
                property_type="wavenumber",
                classification="per_system",
            )
            electronic_spatial_extent = PropertyBaseModel(
                name="electronic_spatial_extent",
                value=np.array(properties["electronic_spatial_extent"]).reshape(1, 1),
                units="angstrom^2",
                property_type="area",
                classification="per_system",
            )
            heat_capacity_at_298K = PropertyBaseModel(
                name="heat_capacity_at_298.15K",
                value=np.array(properties["heat_capacity_at_298.15K"]).reshape(1, 1),
                units=unit.calorie_per_mole / unit.kelvin,
                property_type="heat_capacity",
                classification="per_system",
            )
            record_temp.add_properties(
                [
                    positions,
                    atomic_numbers,
                    partial_charges,
                    isotropic_polarizability,
                    dipole_moment,
                    dipole_moment_scalar,
                    energy_of_homo,
                    lumo_homo_gap,
                    zero_point,
                    internal_energy_at_298K,
                    internal_energy_at_0K,
                    enthalpy_at_298K,
                    free_energy_at_298K,
                    heat_capacity_at_298K,
                    rotational_constants,
                    harmonic_vibrational_frequencies,
                    electronic_spatial_extent,
                ]
            )
            record_temp.validate()
            return record_temp

    def _process_downloaded(
        self,
        local_path_dir: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
    ) -> SourceDataset:
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

        dataset = SourceDataset("qm9")
        for i, file in enumerate(
            tqdm(files[0:n_max], desc="processing", total=len(files))
        ):
            record_temp = self._parse_xyzfile(f"{local_path_dir}/{file}")
            dataset.add_record(record_temp)

        return dataset

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

        # untar the dataset
        from modelforge.utils.misc import extract_tarred_file

        # extract the tar.bz2 file into the local_cache_dir
        # creating a directory called qm9_xyz_files to hold the contents
        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.dataset_filename,
            output_path_dir=f"{self.local_cache_dir}/qm9_xyz_files",
            mode="r:bz2",
        )

        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}/qm9_xyz_files",
            max_records,
            max_conformers_per_record,
            total_conformers,
        )

        logger.info(f"writing file {self.hdf5_file_name} to {self.output_file_dir}")
        self.write_hdf5_and_json_files(
            file_name=self.hdf5_file_name, file_path=self.output_file_dir
        )

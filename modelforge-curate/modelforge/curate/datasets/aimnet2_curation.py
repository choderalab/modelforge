from modelforge.curate import (
    Record,
    SourceDataset,
    PartialCharges,
    DipoleMomentPerSystem,
)
from modelforge.curate.properties import (
    AtomicNumbers,
    Positions,
    Energies,
    Forces,
    TotalCharge,
    PartialCharges,
    DipoleMomentPerSystem,
    QuadrupoleMomentPerSystem,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration

from typing import Optional
from loguru import logger
from openff.units import unit
import numpy as np


class Aimnet2Curation(DatasetCuration):
    """
    Routines to fetch and process the Aimnet2 dataset into a curated hdf5 file.

    The datasets contain molecular structures and the properties computed with B97-3c (GGA DFT) or wB97M-def2-TZVPP
    (range-separated hybrid DFT) methods. Each data file contains about 20M structures.
    DFT calculation performed with ORCA 5.0.3 software.

    Properties include energy, forces, atomic charges, and molecular dipole and quadrupole moments.

    Dataset Citation: Zubatiuk, Roman; Isayev, Olexandr; Anstine, Dylan (2024).
            Training datasets for AIMNet2 machine-learned neural network potential.
            Carnegie Mellon University.
            https://doi.org/10.1184/R1/27629937.v2

    DOI for associated publication:
        publisher: https://doi.org/10.1039/D4SC08572H
        ChemRxiv: https://doi.org/10.26434/chemrxiv-2023-296ch-v3

    Parameters
    ----------
    local_cache_dir: str, optional, default='./Aimnet2_dataset'
        Location to save downloaded dataset.
    version_select: str, optional, default='wB97M_v0'
        Version of the dataset to use. Options include B97-3c_v0 and wB97M_v0 which correspond to the
        data calculated with B97-3c (GGA DFT) and wB97M-def2-TZPP respectively.
        The associated yaml defines all versions and their associated download links;
        see this file for a full lists of all available dataset versions.

    Examples
    --------
    >>> aimnet2_data = Aimnet2Curation(dataset_name="aimnet2", local_cache_dir='~/datasets/aimnet2_dataset')
    >>> aimnet2_data.process()
    >>> aimnet2_data.to_hdf5(output_file_dir='~/datasets/aimnet2_dataset', hdf5_file_name='aimnet2_dataset.hdf5')

    """

    def __init__(
        self,
        dataset_name: str,
        local_cache_dir: str = "./",
        version_select: str = "wb97m_v0",
    ):
        """
        Sets input and output parameters.

        Parameters
        ----------
        dataset_name: str, required
            Name of the dataset to curate.
        local_cache_dir: str, optional, default='./qm9_datafiles'
            Location to save downloaded dataset.
        version_select: str, optional, default='latest'
            Version of the dataset to use as defined in the associated yaml file.

        """
        # since we may have different levels of theory in the same yaml file, we want the default to be wb97m_v0,
        # not necessarily "latest" to avoid any ambiguity
        # this is fine because I can be explicit in the curation scripts as well
        super().__init__(dataset_name, local_cache_dir, version_select)

    def _init_dataset_parameters(self) -> None:
        """
        Initializes the dataset parameters.

        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curate.datasets import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "aimnet2_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "aimnet2"

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

    def _convert_quadrupole(self, components: np.ndarray):
        # we need to map the orca components to a symmetric 3x3 matrix
        # with the appropriate scaling

        psi4_array = []
        for i in range(components.shape[0]):
            xx, yy, zz, xy, yz, xz = components[i]

            orca_matrix = 1.5 * np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            psi4_array.append(orca_matrix)

        return np.concatenate(psi4_array)

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        name: str, required
            Name of the raw hdf5 file,


        Examples
        --------
        """
        import h5py
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"
        logger.debug(f"Processing {input_file_name}.")

        conformers_counter = 0

        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )
        with h5py.File(input_file_name, "r") as hf:
            #  The aimnet2 hdf5 file groups molecules by number of atoms
            # we need to break up each of these groups into individual molecules
            mol_counter = 0

            # note openff units do not support eV
            # we will use the conversion factor of
            eV_to_kjmol = 96.485332125

            # it lists
            for key, properties in hf.items():
                species = properties["numbers"][()]
                coordinates = properties["coord"][()]  # in angstrom
                energies = properties["energy"][()]  # eV
                forces = properties["forces"][()]  # eV/Angstrom
                charges = properties["charges"][()]  # e
                charge = properties["charge"][()]  # e
                dipole = properties["dipole"][
                    ()
                ]  # e * angstrom (note, mislabeled in the dataset readme)
                quadrupole = properties["quadrupole"][
                    ()
                ]  # e * angstrom^2 (note, mislabeled in the dataset readme)

                # in the HDF5 file provided for the aimnet data set,  all configurations of the same size are grouped
                # together into a single array, even if they correspond to different molecules.
                # As a reasonable way to break these up, we use atomic numbers array to identify unique molecules,
                # assuming that sequential identical patterns of atomic numbers correspond to conformers
                # of the same system (same assumption was used in the ani2x dataset).
                # This assumes that isomers either have a different order of atomic numbers
                # (if generated from SMILES they will more than likely be in a different order)
                # or do not occur sequentially.

                import numpy as np

                molecules = {}

                last = species[0]

                molecule_name = (
                    f'{np.array2string(species[0], separator="_")}_m{mol_counter}'
                )

                molecules[molecule_name] = []

                # first loop over all entries in the array
                # to figure out how to break them up
                for i in range(species.shape[0]):
                    if np.all(species[i] == last):
                        molecules[molecule_name].append(i)
                    else:
                        mol_counter += 1
                        molecule_name = f'{np.array2string(species[0], separator="_")}_m{mol_counter}'
                        molecules[molecule_name] = [i]
                        last = species[i]

                for molecule_name in tqdm([key for key in molecules.keys()]):

                    record_temp = Record(name=molecule_name)

                    base_index = molecules[molecule_name][0]

                    # a list of the indices from the array to include
                    indices = molecules[molecule_name]

                    n_conformers = len(molecules[molecule_name])
                    n_atoms = species[base_index].shape[0]

                    atomic_numbers = AtomicNumbers(
                        value=species[base_index].reshape(-1, 1)
                    )
                    record_temp.add_property(atomic_numbers)

                    positions_prop = Positions(
                        value=coordinates[indices].reshape(-1, n_atoms, 3),
                        units=unit.angstrom,
                    )
                    record_temp.add_property(positions_prop)

                    energies_prop = Energies(
                        value=energies[indices].reshape(-1, 1) * eV_to_kjmol,
                        units=unit.kilojoules_per_mole,
                    )
                    record_temp.add_property(energies_prop)

                    forces_prop = Forces(
                        value=forces[indices].reshape(-1, n_atoms, 3) * eV_to_kjmol,
                        units=unit.kilojoules_per_mole / unit.angstrom,
                    )

                    record_temp.add_property(forces_prop)

                    total_charge_prop = TotalCharge(
                        value=charge[indices].reshape(-1, 1),
                        units=unit.elementary_charge,
                    )
                    record_temp.add_property(total_charge_prop)

                    partial_charges_prop = PartialCharges(
                        value=charges[indices].reshape(-1, n_atoms, 1),
                        units=unit.elementary_charge,
                    )
                    record_temp.add_property(partial_charges_prop)

                    dipole_moment_prop = DipoleMomentPerSystem(
                        value=dipole[indices].reshape(-1, 3),
                        units=unit.elementary_charge * unit.angstrom,
                    )
                    record_temp.add_property(dipole_moment_prop)

                    quadruopole_moment = self._convert_quadrupole(
                        quadrupole[indices].reshape(-1, 6)
                    )
                    quadrupole_moment_prop = QuadrupoleMomentPerSystem(
                        value=quadruopole_moment.reshape(-1, 3, 3),
                        units=unit.elementary_charge * unit.angstrom * unit.angstrom,
                    )
                    record_temp.add_property(quadrupole_moment_prop)

                    dataset.add_record(record_temp)

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
        >>> aimnet2_data = Aimnet2Curation(local_cache_dir='~/datasets/aimnet2_dataset')
        >>> aimnet2_data.process()

        """

        from modelforge.curate.utils import download_from_figshare

        url = self.dataset_download_url

        # download the dataset
        download_from_figshare(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.dataset_filename,
            force_download=force_download,
        )

        # # untar and uncompress the dataset
        # from modelforge.utils.misc import extract_tarred_file
        #
        # extract_tarred_file(
        #     input_path_dir=self.local_cache_dir,
        #     file_name=self.dataset_filename,
        #     output_path_dir=self.local_cache_dir,
        #     mode="r:gz",
        # )

        # the untarred file will be in a directory named 'final_h5' within the local_cache_dir,
        # hdf5_filename = f"{self.dataset_filename.replace('.tar.gz', '')}.h5"

        # process the rest of the dataset
        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}",
            self.dataset_filename,
        )

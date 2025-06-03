from modelforge.curate import Record, SourceDataset
from modelforge.curate.properties import (
    AtomicNumbers,
    Positions,
    Energies,
    Forces,
    DipoleMomentPerSystem,
    QuadrupoleMomentPerSystem,
    PartialCharges,
    PropertyBaseModel,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration

from typing import Optional
from loguru import logger
from openff.units import unit
import numpy as np


class ANI1xCuration(DatasetCuration):
    """
        Data class for handling ANI1x dataset.

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
        local_cache_dir: str, optional, default='./AN1_dataset'
            Location to save downloaded dataset.

    )

    """

    def _init_dataset_parameters(self) -> None:
        """
        Initializes the dataset parameters.

        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curation import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "ani1x_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "ani1x"

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
        from numpy import newaxis

        input_file_name = f"{local_path_dir}/{name}"
        logger.debug(f"Processing {input_file_name}.")

        conformers_counter = 0

        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )
        with h5py.File(input_file_name, "r") as hf:

            names = list(hf.keys())
            for molecule_name in tqdm(names):

                record_temp = Record(name=molecule_name)

                atomic_numbers = AtomicNumbers(
                    value=hf[molecule_name]["atomic_numbers"][()].reshape(-1, 1)
                )
                record_temp.add_property(atomic_numbers)

                # we can grab the number of atoms just to make it easier to reshape
                n_atoms = record_temp.n_atoms

                positions = Positions(
                    value=hf[molecule_name]["coordinates"][()].reshape(-1, n_atoms, 3),
                    units=unit.angstrom,
                )
                record_temp.add_property(positions)

                wb97x_dz_energy = Energies(
                    name="wb97x_dz_energy",
                    value=hf[molecule_name]["wb97x_dz.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(wb97x_dz_energy)

                wb97x_tz_energy = Energies(
                    name="wb97x_tz_energy",
                    value=hf[molecule_name]["wb97x_tz.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(wb97x_tz_energy)

                ccsd_t_cbs_energy = Energies(
                    name="ccsd(t)_cbs_energy",
                    value=hf[molecule_name]["ccsd(t)_cbs.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(ccsd_t_cbs_energy)

                hf_dz_energy = Energies(
                    name="hf_dz_energy",
                    value=hf[molecule_name]["hf_dz.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(hf_dz_energy)

                hf_tz_energy = Energies(
                    name="hf_tz_energy",
                    value=hf[molecule_name]["hf_tz.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(hf_tz_energy)

                hf_qz_energy = Energies(
                    name="hf_qz_energy",
                    value=hf[molecule_name]["hf_qz.energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(hf_qz_energy)

                npno_ccsd_t_dz_corr_energy = Energies(
                    name="npno_ccsd(t)_dz_corr_energy",
                    value=hf[molecule_name]["npno_ccsd(t)_dz.corr_energy"][()].reshape(
                        -1, 1
                    ),
                    units=unit.hartree,
                )
                record_temp.add_property(npno_ccsd_t_dz_corr_energy)

                npno_ccsd_t_tz_corr_energy = Energies(
                    name="npno_ccsd(t)_tz_corr_energy",
                    value=hf[molecule_name]["npno_ccsd(t)_tz.corr_energy"][()].reshape(
                        -1, 1
                    ),
                    units=unit.hartree,
                )
                record_temp.add_property(npno_ccsd_t_tz_corr_energy)

                tpno_ccsd_t_dz_corr_energy = Energies(
                    name="tpno_ccsd(t)_dz_corr_energy",
                    value=hf[molecule_name]["tpno_ccsd(t)_dz.corr_energy"][()].reshape(
                        -1, 1
                    ),
                    units=unit.hartree,
                )
                record_temp.add_property(tpno_ccsd_t_dz_corr_energy)

                mp2_dz_corr_energy = Energies(
                    name="mp2_dz_corr_energy",
                    value=hf[molecule_name]["mp2_dz.corr_energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(mp2_dz_corr_energy)

                mp2_tz_corr_energy = Energies(
                    name="mp2_tz_corr_energy",
                    value=hf[molecule_name]["mp2_tz.corr_energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(mp2_tz_corr_energy)

                mp2_qz_corr_energy = Energies(
                    name="mp2_qz_corr_energy",
                    value=hf[molecule_name]["mp2_qz.corr_energy"][()].reshape(-1, 1),
                    units=unit.hartree,
                )
                record_temp.add_property(mp2_qz_corr_energy)

                wb97x_dz_forces = Forces(
                    name="wb97x_dz_forces",
                    value=hf[molecule_name]["wb97x_dz.forces"][()].reshape(
                        -1, n_atoms, 3
                    ),
                    units=unit.hartree / unit.angstrom,
                )
                record_temp.add_property(wb97x_dz_forces)

                wb97x_tz_forces = Forces(
                    name="wb97x_tz_forces",
                    value=hf[molecule_name]["wb97x_tz.forces"][()].reshape(
                        -1, n_atoms, 3
                    ),
                    units=unit.hartree / unit.angstrom,
                )
                record_temp.add_property(wb97x_tz_forces)

                wb97x_dz_dipole = DipoleMomentPerSystem(
                    name="wb97x_dz_dipole",
                    value=hf[molecule_name]["wb97x_dz.dipole"][()].reshape(-1, 3),
                    units=unit.elementary_charge * unit.angstrom,
                )
                record_temp.add_property(wb97x_dz_dipole)

                wb97x_tz_dipole = DipoleMomentPerSystem(
                    name="wb97x_tz_dipole",
                    value=hf[molecule_name]["wb97x_tz.dipole"][()].reshape(-1, 3),
                    units=unit.elementary_charge * unit.angstrom,
                )
                record_temp.add_property(wb97x_tz_dipole)

                # Rather than a 3x3 matrix, this just has the 6 unique elements
                # So we need to adjust this to be 3x3 for consistency
                # The 6 unique element output order used by ORCA (used for ani1x) is: xx, yy, zz, xy, xz, yz
                # The 3x3 matrix order is: xx, xy, xz, xy, yy, yz, xz, yz, zz.
                # So we can just create a new array with the correct order.
                # to avoid a loop we just use : and then transpose the array before reshaping.

                qt = hf[molecule_name]["wb97x_dz.quadrupole"][()]
                quadrupole = np.array(
                    [
                        qt[:, 0],
                        qt[:, 3],
                        qt[:, 4],
                        qt[:, 3],
                        qt[:, 1],
                        qt[:, 5],
                        qt[:, 4],
                        qt[:, 5],
                        qt[:, 2],
                    ]
                ).T.reshape(-1, 3, 3)

                wb97x_dz_quadrupole = QuadrupoleMomentPerSystem(
                    name="wb97x_dz_quadrupole",
                    value=quadrupole,
                    units=unit.elementary_charge * unit.angstrom**2,
                )
                record_temp.add_property(wb97x_dz_quadrupole)

                wb97x_dz_cm5_charges = PartialCharges(
                    name="wb97x_dz_cm5_charges",
                    value=hf[molecule_name]["wb97x_dz.cm5_charges"][()].reshape(
                        -1, n_atoms, 1
                    ),
                    units=unit.elementary_charge,
                )
                record_temp.add_property(wb97x_dz_cm5_charges)

                wb97x_dz_hirshfeld_charges = PartialCharges(
                    name="wb97x_dz_hirshfeld_charges",
                    value=hf[molecule_name]["wb97x_dz.hirshfeld_charges"][()].reshape(
                        -1, n_atoms, 1
                    ),
                    units=unit.elementary_charge,
                )
                record_temp.add_property(wb97x_dz_hirshfeld_charges)

                wb97x_tz_mbis_charges = PartialCharges(
                    name="wb97x_tz_mbis_charges",
                    value=hf[molecule_name]["wb97x_tz.mbis_charges"][()].reshape(
                        -1, n_atoms, 1
                    ),
                    units=unit.elementary_charge,
                )
                record_temp.add_property(wb97x_tz_mbis_charges)

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
        >>> ani1_data = ANI1xCuration(local_cache_dir='~/datasets/ani1x_dataset')
        >>> ani1_data.process()

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

        hdf5_filename = self.dataset_filename

        # process the rest of the dataset
        self.dataset = self._process_downloaded(
            self.local_cache_dir,
            hdf5_filename,
        )

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


class tmQMXTBCuration(DatasetCuration):
    """
    Routines to process the tmQM-xtb dataset into a curated hdf5 file.

    This dataset uses  configurations from the original tmQM dataset as starting points
    for semi-emperical calculations using the GFN2-XTB (calculated using TBlite) that perform MD-based sampling.
    MD sampling using the atomic simulation environment (ASE) and was performed using the Langevin thermostat,
    with a time step of 1.0 fs, and friction of 0.01 1/fs.

    v0 of the dataset performings sampling at T=400K, generating 10 snapshots per molecule (the first corresponding
    to the original, energy minimized state in tmQM) with 100 timesteps spacing between each snapshot. The goal
    was to primarily capture the fluctuations around the equilibrium, rather than large scale conformation changes.

    To remove potentially problematic configurations, a filtering criteria is applied during curation:
    - The initial configurations of the molecules undergo bond perception using RDKit, with bond distances recorded.
    - The relative change in bond length is calculated for each snapshot.
    - If the relative change in any bond length is more than 0.09 the snapshot is removed from the dataset.


    The original tmQM dataset contains the geometries and properties of 108,541 (in the 13Aug24 release)
    mononuclear complexes extracted from the Cambridge Structural Database, including Werner, bioinorganic, and
    organometallic complexes based on a large variety of organic ligands and 30 transition metals
    (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    The scripts used to generate the tmQM-xtb dataset are available at:
    https://github.com/chrisiacovella/xtb_Config_gen

    The tmQM-xtb dataset is avialble from zenodo:
     10.5281/zenodo.14894964 (v0)

    Citation to the original tmQM dataset:

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
    >>> tmQM_xtb_data = tmQMXTBCuration(local_cache_dir='~/datasets/tmQM_dataset')
    >>> tmQM_xtb_data.process()
    >>> tmQM_xtb_data.to_hdf5(hdf5_file_name='tmQM_dataset.hdf5', output_file_dir='~/datasets/tmQM_dataset')



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

        yaml_file = resources.files(yaml_files) / "tmqm_xtb_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "tmqm_xtb"

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

        # if convert_units is True, which it is by default
        # we will convert each input unit (key) to the following output units (val)

    def _process_downloaded(
        self,
        local_path_dir: str,
        hdf5_file_name: str,
    ):
        """
        Processes a downloaded dataset: extracts relevant information into a list of dicts.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the .hd5f file.
        hdf5_file_name: str, required
            Name of the hdf5 file that will be read



        Examples
        --------

        """
        from tqdm import tqdm
        from modelforge.utils.misc import OpenWithLock
        import h5py

        dataset = SourceDataset("tmqm_xtb")
        with OpenWithLock(f"{local_path_dir}/{hdf5_file_name}.lockfile", "w") as f:
            with h5py.File(f"{local_path_dir}/{hdf5_file_name}", "r") as f:
                for key in tqdm(f.keys()):
                    # set up a record
                    record = Record(name=key)

                    # extract the atomic numbers
                    atomic_numbers = AtomicNumbers(
                        value=f[key]["atomic_numbers"][()].reshape(-1, 1)
                    )
                    record.add_property(atomic_numbers)
                    n_atoms = atomic_numbers.n_atoms
                    # extract the positions
                    positions = Positions(
                        value=f[key]["geometry"][()].reshape(-1, n_atoms, 3),
                        units=f[key]["geometry"].attrs["u"],
                    )
                    record.add_property(positions)

                    # extract the energies
                    energies = Energies(
                        value=f[key]["energy"][()].reshape(-1, 1),
                        units=f[key]["energy"].attrs["u"],
                    )
                    record.add_property(energies)

                    # extract the forces
                    forces = Forces(
                        value=f[key]["forces"][()].reshape(-1, n_atoms, 3),
                        units=f[key]["forces"].attrs["u"],
                    )
                    record.add_property(forces)

                    # extract the partial charges
                    partial_charges = PartialCharges(
                        value=f[key]["partial_charges"][()].reshape(-1, n_atoms, 1),
                        units=f[key]["partial_charges"].attrs["u"],
                    )
                    record.add_property(partial_charges)

                    # extract the dipole moment
                    dipole_moment = DipoleMomentScalarPerSystem(
                        value=f[key]["dipole_moment"][()].reshape(-1, 1),
                        units=f[key]["dipole_moment"].attrs["u"],
                    )
                    record.add_property(dipole_moment)

                    # extract the total charge
                    total_charge = TotalCharge(
                        value=f[key]["total_charge"][()].reshape(-1, 1),
                        units=f[key]["total_charge"].attrs["u"],
                    )
                    record.add_property(total_charge)

                    # extract spin multiplicities
                    spin_multiplicities = SpinMultiplicities(
                        value=f[key]["spin_multiplicity"][()].reshape(-1, 1),
                    )
                    record.add_property(spin_multiplicities)

                    # metadata for scoichiometry
                    metadata = MetaData(
                        name="stoichiometry",
                        value=f[key]["stoichiometry"][()].decode("utf-8"),
                    )

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
        >>> tmQM_xtb_data = tmQMXTBCuration(local_cache_dir='~/datasets/tmQM_Xtb_dataset')
        >>> tmQM_xtb_data.process()

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

        # unzip the dataset

        from modelforge.utils.misc import ungzip_file

        ungzip_file(
            input_path_dir=f"{self.local_cache_dir}",
            file_name=self.dataset_filename,
            output_path_dir=f"{self.local_cache_dir}",
        )
        unzipped_file_name = self.dataset_filename.replace(".gz", "")
        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}", unzipped_file_name
        )

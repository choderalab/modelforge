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
    DipoleMomentPerSystem,
    SpinMultiplicities,
)
from modelforge.curate.record import infer_bonds, calculate_max_bond_length_change

from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

from modelforge.utils.units import chem_context
import numpy as np

from typing import Optional, List
from loguru import logger
from openff.units import unit


class FeIICuration(DatasetCuration):
    """
    Routines to process the Fe(II) dataset into a curated hdf5 file.

    The Fe(II) dataset includes 28834 total configurations for 383 unique molecules Fe(II) organometallic complexes.
    Specifically, this includes 15568 HS geometries and 13266 LS geometries.
    These complexes originate from the Cambridge Structural Database (CSD) as curated by
    Nandy, et al. (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214), and were filtered
    into “computation-ready” complexes, (those where both oxidation states and charges are already specified
    without hydrogen atoms missing in the structures), following the procedure outlined by Arunachalam, et al.
    (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700)


    The Fe (II)  dataset is available from github:
    https://github.com/Neon8988/Iron_NNPs

    Citation to the original  dataset:

    Modeling Fe(II) Complexes Using Neural Networks
    Hongni Jin and Kenneth M. Merz Jr.
    Journal of Chemical Theory and Computation 2024 20 (6), 2551-2558
    DOI: 10.1021/acs.jctc.4c00063


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
        cutoff: Optional[unit.Quantity] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information into a list of dicts.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the .hd5f file.
        hdf5_file_name: str, required
            Name of the hdf5 file that will be read
        cutoff: unit.Quantity, optional, default=None
            The cutoff value for the relative change in bond length to filter out problematic configurations.



        Examples
        --------

        """
        from tqdm import tqdm
        from modelforge.utils.misc import OpenWithLock
        import h5py

        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )
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
                    dipole_moment = DipoleMomentPerSystem(
                        value=f[key]["dipole_moment"][()].reshape(-1, 3),
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
                    record.add_property(metadata)

                    if cutoff is not None:
                        bonds = infer_bonds(record)
                        max_bond_delta = calculate_max_bond_length_change(record, bonds)
                        configs_to_include = []
                        if len(max_bond_delta) != record.n_configs:
                            raise ValueError(
                                "Number of  max bond lengths does not match number of configurations"
                            )
                        for index, delta in enumerate(max_bond_delta):
                            if delta <= cutoff:
                                configs_to_include.append(index)
                        record_new = record.remove_configs(configs_to_include)
                        dataset.add_record(record_new)
                    else:
                        dataset.add_record(record)

            return dataset

    def process(
        self, force_download: bool = False, cutoff: Optional[unit.Quantity] = None
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        cutoff: unit.Quantity, optional, default=None
            The cutoff value for the relative change in bond length to filter out problematic configurations.



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

        if cutoff is not None:
            assert cutoff.is_compatible_with(unit.angstrom)

        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}", unzipped_file_name, cutoff=cutoff
        )

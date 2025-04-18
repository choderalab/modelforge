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

# pytorch geometric is needed to unpickle the data
# import torch_geometric
# from torch_geometric.data import Data


class FeIICuration(DatasetCuration):
    """
    Routines to process the Fe(II) dataset into a curated hdf5 file.

    The Fe(II) dataset includes 28834 total configurations for 383 unique molecules Fe(II) organometallic complexes.
    Specifically, this includes 15568 HS geometries and 13266 LS geometries.
    These complexes originate from the Cambridge Structural Database (CSD) as curated by
    Nandy, et al. (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214),
    and were filtered into “computation-ready” complexes, (those where both oxidation states and charges are already
    specified without hydrogen atoms missing in the structures), following the procedure outlined by Arunachalam, et al.
    (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700)


    The original Fe (II) dataset is available from github:
    https://github.com/Neon8988/Iron_NNPs

    The code uses a fork of the original dataset, to enable clear versioning (i.e. a release):
    https://github.com/chrisiacovella/Iron_NNPs

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
    >>> fe_II_data = FeIICuration(dataset_name="fe_II", local_cache_dir='~/datasets/fe_II_dataset')
    >>> fe_II_data.process()
    >>> fe_II_data.to_hdf5(hdf5_file_name='fe_II_dataset.hdf5', output_file_dir='~/datasets/fe_II_dataset')



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

        yaml_file = resources.files(yaml_files) / "fe_II_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "fe_II"

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

    # def _pyg2_data_transform(self, data: Data) -> Data:
    #     """
    #     This is taken directly from the IronNNPs repository and is needed after we unpickle the data
    #
    #     if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    #     we need to convert the data to the new format
    #     """
    #     if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
    #         return Data(**{k: v for k, v in data.__dict__.items() if v is not None})
    #
    #     return data

    def _process_downloaded(
        self,
        local_path_dir: str,
        lmdb_files: str,
    ):
        """
        Processes a downloaded dataset: extracts relevant information into a list of dicts.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the lmdb files.
        lmdb_files: str, required
            Name of the hdf5 file that will be read


        Examples
        --------

        """
        from tqdm import tqdm
        from modelforge.utils.misc import OpenWithLock
        import lmdb
        import pickle

        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )
        for lmdb_file in lmdb_files:
            with OpenWithLock(f"{local_path_dir}/{lmdb_file}.lockfile", "w") as f:
                env = lmdb.open(
                    f"{local_path_dir}/{lmdb_file}",
                    readonly=True,
                    lock=False,
                    subdir=False,
                    readahead=False,
                    meminit=False,
                    max_readers=1,
                )

                keys = [f"{j}".encode("ascii") for j in range(env.stat()["entries"])]
                for idx in range(len(keys)):
                    datapoint_pickled = env.begin().get(keys[idx])

                    pickled = pickle.loads(datapoint_pickled)
                    print(pickled["atomic_numbers"])
                    break
                    # pickled = self._pyg2_data_transform(pickle.loads(datapoint_pickled))
        # with OpenWithLock(f"{local_path_dir}/{hdf5_file_name}.lockfile", "w") as f:
        #     with h5py.File(f"{local_path_dir}/{hdf5_file_name}", "r") as f:
        #         for key in tqdm(f.keys()):
        #             # set up a record
        #             record = Record(name=key)
        #
        #             # extract the atomic numbers
        #             atomic_numbers = AtomicNumbers(
        #                 value=f[key]["atomic_numbers"][()].reshape(-1, 1)
        #             )
        #             record.add_property(atomic_numbers)
        #             n_atoms = atomic_numbers.n_atoms
        #             # extract the positions
        #             positions = Positions(
        #                 value=f[key]["geometry"][()].reshape(-1, n_atoms, 3),
        #                 units=f[key]["geometry"].attrs["u"],
        #             )
        #             record.add_property(positions)
        #
        #             # extract the energies
        #             energies = Energies(
        #                 value=f[key]["energy"][()].reshape(-1, 1),
        #                 units=f[key]["energy"].attrs["u"],
        #             )
        #             record.add_property(energies)
        #
        #             # extract the forces
        #             forces = Forces(
        #                 value=f[key]["forces"][()].reshape(-1, n_atoms, 3),
        #                 units=f[key]["forces"].attrs["u"],
        #             )
        #             record.add_property(forces)
        #
        #             # extract the partial charges
        #             partial_charges = PartialCharges(
        #                 value=f[key]["partial_charges"][()].reshape(-1, n_atoms, 1),
        #                 units=f[key]["partial_charges"].attrs["u"],
        #             )
        #             record.add_property(partial_charges)
        #
        #             # extract the dipole moment
        #             dipole_moment = DipoleMomentPerSystem(
        #                 value=f[key]["dipole_moment"][()].reshape(-1, 3),
        #                 units=f[key]["dipole_moment"].attrs["u"],
        #             )
        #             record.add_property(dipole_moment)
        #
        #             # extract the total charge
        #             total_charge = TotalCharge(
        #                 value=f[key]["total_charge"][()].reshape(-1, 1),
        #                 units=f[key]["total_charge"].attrs["u"],
        #             )
        #             record.add_property(total_charge)
        #
        #             # extract spin multiplicities
        #             spin_multiplicities = SpinMultiplicities(
        #                 value=f[key]["spin_multiplicity"][()].reshape(-1, 1),
        #             )
        #             record.add_property(spin_multiplicities)
        #
        #             # metadata for scoichiometry
        #             metadata = MetaData(
        #                 name="stoichiometry",
        #                 value=f[key]["stoichiometry"][()].decode("utf-8"),
        #             )
        #             record.add_property(metadata)
        #
        #             if cutoff is not None:
        #                 bonds = infer_bonds(record)
        #                 max_bond_delta = calculate_max_bond_length_change(record, bonds)
        #                 configs_to_include = []
        #                 if len(max_bond_delta) != record.n_configs:
        #                     raise ValueError(
        #                         "Number of  max bond lengths does not match number of configurations"
        #                     )
        #                 for index, delta in enumerate(max_bond_delta):
        #                     if delta <= cutoff:
        #                         configs_to_include.append(index)
        #                 record_new = record.remove_configs(configs_to_include)
        #                 dataset.add_record(record_new)
        #             else:
        #                 dataset.add_record(record)
        #
        #     return dataset

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
        cutoff: unit.Quantity, optional, default=None
            The cutoff value for the relative change in bond length to filter out problematic configurations.



        Examples
        --------
        >>> fe_II_data = FeIICuration(dataset_name = "fe_II", local_cache_dir='~/datasets/tmQM_Xtb_dataset')
        >>> fe_II_data.process()

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

        from modelforge.utils.misc import extract_tarred_file, list_files

        # extract the tar.bz2 file into the local_cache_dir
        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.dataset_filename,
            output_path_dir=f"{self.local_cache_dir}/fe_II_files",
            mode="r:gz",
        )

        lmdb_files = list_files(
            directory=f"{self.local_cache_dir}/fe_II_files/{self.extracted_filepath}",
            extension=".lmdb",
        )
        print(lmdb_files)

        self.dataset = self._process_downloaded(
            local_path_dir=f"{self.local_cache_dir}/fe_II_files/{self.extracted_filepath}",
            lmdb_files=lmdb_files,
        )

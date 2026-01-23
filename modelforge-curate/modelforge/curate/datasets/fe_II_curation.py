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
    SpinMultiplicitiesPerSystem,
)
from modelforge.curate.record import (
    infer_bonds,
    calculate_max_bond_length_change,
    map_configurations,
)

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
    def _fast_map(
        self, atomic_numbers_ref: np.ndarray, atomic_numbers_test: np.ndarray
    ) -> List[int]:
        index_map1 = [(val, idx) for idx, val in enumerate(atomic_numbers_test)]

        mapping = []
        for i in range(len(atomic_numbers_ref)):
            for j in range(len(index_map1)):
                if atomic_numbers_ref[i] == index_map1[j][0]:
                    mapping.append((index_map1[j][1]))
                    index_map1.pop(j)
                    break
        return mapping

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
            name=self.dataset_name,
            local_db_dir=self.local_cache_dir,
            append_property=True,
        )
        ev_to_kj_mol = 96.485
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
                for idx in tqdm(range(len(keys))):
                    datapoint_pickled = env.begin().get(keys[idx])

                    pickled = pickle.loads(datapoint_pickled)
                    atomic_numbers_tmp = (
                        pickled["atomic_numbers"].numpy().reshape(-1, 1)
                    )

                    atomic_numbers = AtomicNumbers(value=atomic_numbers_tmp.astype(int))

                    # in all cases, we have only a single snapshot
                    positions_tmp = pickled["pos"].numpy().reshape(1, -1, 3)
                    positions = Positions(value=positions_tmp, units="angstrom")

                    energies_tmp = np.array(pickled["y"]).reshape(1, 1) * ev_to_kj_mol
                    energies = Energies(value=energies_tmp, units="kilojoule_per_mole")

                    forces_tmp = (
                        pickled["force"].numpy().reshape(1, -1, 3) * ev_to_kj_mol
                    )
                    forces = Forces(
                        value=forces_tmp, units=unit.kilojoule_per_mole / unit.angstrom
                    )

                    charge_tmp = np.array([pickled["charge"]]).reshape(1, 1)
                    charge = TotalCharge(value=charge_tmp, units="elementary_charge")

                    spin_tmp = np.array(pickled["spin"]).reshape(1, 1)
                    spin = SpinMultiplicitiesPerSystem(value=spin_tmp)

                    sid = pickled["sid"]
                    molid = sid.split("_")[0]
                    molecule_name = MetaData(name="mol_id", value=molid)
                    record = Record(name=molid, append_property=True)
                    record.add_properties(
                        [
                            atomic_numbers,
                            positions,
                            energies,
                            forces,
                            charge,
                            spin,
                            molecule_name,
                        ]
                    )
                    # if the molecule is not in the dataset, add it
                    if not molid in dataset.records.keys():
                        dataset.add_record(record)
                    # if the molecule is in the dataset, we need to check if the atomic numbers match
                    else:
                        # fetch the existing record
                        record_existing = dataset.get_record(molid)

                        # compare the atomic numbers, if they do not match, we need to reorder
                        if not np.all(
                            record_existing.atomic_numbers.value
                            == record.atomic_numbers.value
                        ):
                            # if they don't match, we need to get the mapping and reorder
                            # this uses RDKIT under the hood, using GetBestAlignmentTransform.
                            # It is not very efficient, because of many steps, but since curation is  namely
                            # a one time cost, it is not a big deal.
                            # The steps involve: find the orientation that minimizes the RMSD, then
                            # find the correspondence between atoms in each molecule based on element and distance.
                            # This is likely better than just reordering atoms naively,
                            # since it ensures that, e.g., the 3rd carbon atom refers to the same carbon atom
                            # in both molecules. This is especially useful for visualization or any application that
                            # requires a fixed topology.
                            # However it shouldn't actually matter to the NNP claculation, since 1) we don't have bonds
                            # 2) interacting pairs are determined uniquely for each configuration during training.
                            # 65 configurations cdfail the first method, so even still should only be a small fraction
                            try:
                                mapping = map_configurations(record_existing, record)

                            except:
                                # if the rdkit mapping fails, we need to use the fast mapping
                                # this does not try to orient the molecules and then create
                                # a mapping based on the closest atoms of the same species,
                                # instead it just ensure that the atomic numbers are the same
                                # order, which should be fine
                                mapping = self._fast_map(
                                    record_existing.atomic_numbers.value,
                                    record.atomic_numbers.value,
                                )
                            record.reorder(mapping)

                            if not np.all(
                                record.atomic_numbers.value
                                == record_existing.atomic_numbers.value
                            ):
                                raise ValueError(
                                    "Atomic numbers do not match after reordering"
                                )

                        # add the properties to the existing record
                        dataset.add_properties(
                            name=molid,
                            properties=[
                                record.get_property("positions"),
                                record.get_property("energies"),
                                record.get_property("forces"),
                                record.get_property("total_charge"),
                                record.get_property("spin_multiplicities"),
                            ],
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
        status = download_from_url(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.dataset_filename,
            length=self.dataset_length,
            force_download=force_download,
        )
        if status == False:
            logger.info(f"Could not download file from {url}")
            raise Exception("Failed to download file")

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

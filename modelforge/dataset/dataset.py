"""
This module contains classes and functions for managing datasets.
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as log
from openff.units import Quantity
from torch.utils.data import DataLoader

from modelforge.dataset.utils import RandomRecordSplittingStrategy, SplittingStrategy
from modelforge.utils.misc import lock_with_attribute
from modelforge.utils.prop import PropertyNames
from modelforge.utils.prop import BatchData
from modelforge.utils.prop import NNPInput
from modelforge.utils.prop import Metadata

if TYPE_CHECKING:
    from modelforge.potential.processing import AtomicSelfEnergies

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class CaseInsensitiveEnum(str, Enum):
    """
    Enum class that allows case-insensitive comparison of its members.
    """

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class DataSetName(CaseInsensitiveEnum):
    QM9 = "QM9"
    ANI1X = "ANI1X"
    ANI2X = "ANI2X"
    SPICE1 = "SPICE1"
    SPICE2 = "SPICE2"
    SPICE1_OPENFF = "SPICE1_OPENFF"
    PHALKETHOH = "PhAlkEthOH"


class DatasetParameters(BaseModel):
    """
    Class to hold the dataset parameters.

    Attributes
    ----------
    dataset_name : DataSetName
        The name of the dataset.
    version_select : str
        The version of the dataset to use.
    num_workers : int
        The number of workers to use for the DataLoader.
    pin_memory : bool
        Whether to pin memory for the DataLoader.
    regenerate_processed_cache : bool
        Whether to regenerate the processed cache.
    """

    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    dataset_name: DataSetName
    version_select: str
    num_workers: int = Field(gt=0)
    pin_memory: bool
    regenerate_processed_cache: bool = False


# Define the input class
class TorchDataset(torch.utils.data.Dataset[BatchData]):
    def __init__(
        self,
        dataset: np.lib.npyio.NpzFile,
        property_name: PropertyNames,
        preloaded: bool = False,
    ):
        """
        Wraps a numpy dataset to make it compatible with PyTorch DataLoader.

        Parameters
        ----------
        dataset : np.lib.npyio.NpzFile
            The underlying numpy dataset.
        property_name : PropertyNames
            Names of the properties to extract from the dataset.
        preloaded : bool, optional
            If True, converts properties to PyTorch tensors ahead of time. Default
            is False.
        """
        super().__init__()
        self.preloaded = preloaded
        self.properties_of_interest = self._load_properties(dataset, property_name)

        self.number_of_records = len(dataset["atomic_subsystem_counts"])
        self.number_of_atoms = len(dataset["atomic_numbers"])
        self.length = len(self.properties_of_interest["E"])

        # Prepare indices for atom and conformer data
        self._prepare_indices(dataset)

    def _prepare_indices(self, dataset: np.lib.npyio.NpzFile):
        """Prepare indices for atom and conformer data."""
        single_atom_start_idxs_by_rec = np.concatenate(
            [np.array([0]), np.cumsum(dataset["atomic_subsystem_counts"])]
        )
        self.series_mol_start_idxs_by_rec = np.concatenate(
            [np.array([0]), np.cumsum(dataset["n_confs"])]
        )

        self.single_atom_start_idxs_by_conf = np.repeat(
            single_atom_start_idxs_by_rec[: self.number_of_records], dataset["n_confs"]
        )
        self.single_atom_end_idxs_by_conf = np.repeat(
            single_atom_start_idxs_by_rec[1 : self.number_of_records + 1],
            dataset["n_confs"],
        )

        self.series_atom_start_idxs_by_conf = np.concatenate(
            [
                np.array([0]),
                np.cumsum(
                    np.repeat(dataset["atomic_subsystem_counts"], dataset["n_confs"])
                ),
            ]
        )

    def _load_properties(
        self, dataset: np.lib.npyio.NpzFile, property_name: PropertyNames
    ) -> Dict[str, torch.Tensor]:
        """Load properties from the dataset."""
        properties = {
            "atomic_numbers": torch.from_numpy(
                dataset[property_name.atomic_numbers].flatten()
            ).to(torch.int32),
            "positions": torch.from_numpy(dataset[property_name.positions]).to(
                torch.float32
            ),
            "E": torch.from_numpy(dataset[property_name.E]).to(torch.float64),
        }

        properties["total_charge"] = (
            torch.from_numpy(dataset[property_name.total_charge])
            .to(torch.int32)
            .unsqueeze(-1)
            if property_name.total_charge is not None
            and False  # FIXME: as soon as I figured out how to make this to a per data point property
            else torch.zeros((dataset[property_name.E].shape[0], 1), dtype=torch.int32)
        )

        properties["F"] = (
            torch.from_numpy(dataset[property_name.F])
            if property_name.F is not None
            else torch.zeros_like(properties["positions"])
        )

        properties["dipole_moment"] = (
            torch.from_numpy(dataset[property_name.dipole_moment])
            if property_name.dipole_moment is not None
            else torch.zeros(
                (dataset[property_name.E].shape[0], 3), dtype=torch.float32
            )
        )

        properties["pair_list"] = None  # Placeholder for pair list

        return properties

    def __len__(self) -> int:
        """
        Return the number of data points in the dataset.

        Returns:
        --------
        int
            Total number of data points available in the dataset.
        """
        return self.length

    def record_len(self) -> int:
        """
        Return the number of records in the TorchDataset.
        """
        return self.number_of_records

    def get_series_mol_idxs(self, record_idx: int) -> List[int]:
        """
        Return the indices of the conformers for a given record.
        """
        return list(
            range(
                self.series_mol_start_idxs_by_rec[record_idx],
                self.series_mol_start_idxs_by_rec[record_idx + 1],
            )
        )

    def __setitem__(self, idx: int, value: Dict[str, torch.Tensor]) -> None:
        """
        Set the value of a property for a given conformer index.

        Parameters
        ----------
        idx : int
            Index of the conformer to set the value for.
        value : Dict[str, torch.Tensor]
            Dictionary containing the property name and value to set.

        """
        for key, val in value.items():
            self.properties_of_interest[key][idx] = val

    def _set_pairlist(self, idx: int):
        # pairlist is set here (instead of l279) because it is not a default property
        if self.properties_of_interest["pair_list"] is None:
            pair_list = None
        else:
            pair_list_indices_start = self.properties_of_interest["number_of_pairs"][
                idx
            ]
            pair_list_indices_end = self.properties_of_interest["number_of_pairs"][
                idx + 1
            ]
            pair_list = self.properties_of_interest["pair_list"][
                :, pair_list_indices_start:pair_list_indices_end
            ]
        return pair_list

    def __getitem__(self, idx: int) -> BatchData:
        """Fetch data for a given conformer index."""
        series_atom_start_idx = self.series_atom_start_idxs_by_conf[idx]
        series_atom_end_idx = self.series_atom_start_idxs_by_conf[idx + 1]
        single_atom_start_idx = self.single_atom_start_idxs_by_conf[idx]
        single_atom_end_idx = self.single_atom_end_idxs_by_conf[idx]

        atomic_numbers = self.properties_of_interest["atomic_numbers"][
            single_atom_start_idx:single_atom_end_idx
        ]

        # get properties (Note that default properties are set in l279)
        positions = self.properties_of_interest["positions"][
            series_atom_start_idx:series_atom_end_idx
        ]
        E = self.properties_of_interest["E"][idx]
        F = self.properties_of_interest["F"][series_atom_start_idx:series_atom_end_idx]
        total_charge = self.properties_of_interest["total_charge"][idx]
        number_of_atoms = len(atomic_numbers)
        dipole_moment = self.properties_of_interest["dipole_moment"][idx]

        nnp_input = NNPInput(
            atomic_numbers=atomic_numbers,
            positions=positions,
            pair_list=self._set_pairlist(idx),
            per_system_total_charge=total_charge,
            atomic_subsystem_indices=torch.zeros(number_of_atoms, dtype=torch.int32),
        )
        metadata = Metadata(
            per_system_energy=E,
            per_atom_force=F,
            atomic_subsystem_counts=torch.tensor([number_of_atoms], dtype=torch.int32),
            atomic_subsystem_indices_referencing_dataset=torch.repeat_interleave(
                torch.tensor([idx], dtype=torch.int32), number_of_atoms
            ),
            number_of_atoms=number_of_atoms,
            per_system_dipole_moment=dipole_moment,
        )

        return BatchData(nnp_input, metadata)


class HDF5Dataset:
    """
    Manages data stored in HDF5 format, supporting processing and interaction.

    Attributes
    ----------
    raw_data_file : str
        Path to the raw HDF5 data file.
    processed_data_file : str
        Path to the processed data file, typically a .npz file for efficiency.
    """

    def __init__(
        self,
        url: str,
        gz_data_file: Dict[str, str],
        hdf5_data_file: Dict[str, str],
        processed_data_file: Dict[str, str],
        local_cache_dir: str,
        force_download: bool = False,
        regenerate_cache: bool = False,
    ):
        """
        Initializes the HDF5Dataset with paths to raw and processed data files.

        Parameters
        ----------
        url : str
            URL of the hdf5.gz data file.
        gz_data_file : Dict[str, str]
            Name of the gzipped data file (name) and checksum (md5).
        hdf5_data_file : Dict[str, str]
            Name of the hdf5 data file (name) and checksum (md5).
        processed_data_file : Dict[str, str]
            Name of the processed npz data file (name) and checksum (md5).
        local_cache_dir : str
            Directory to store the files.
        force_download : bool, optional
            If set to True, the data will be downloaded even if it already exists. Default is False.
        regenerate_cache : bool, optional
            If set to True, the cache file will be regenerated even if it already exists. Default is False.
        """
        self.url = url
        self.gz_data_file = gz_data_file
        self.hdf5_data_file = hdf5_data_file
        self.processed_data_file = processed_data_file
        import os

        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.force_download = force_download
        self.regenerate_cache = regenerate_cache

        self.hdf5data: Optional[Dict[str, List[np.ndarray]]] = None
        self.numpy_data: Optional[np.ndarray] = None

    def _ungzip_hdf5(self) -> None:
        """
        Unzips an HDF5.gz file.

        Examples
        -------
        """
        import gzip
        import shutil

        with gzip.open(
            f"{self.local_cache_dir}/{self.gz_data_file['name']}", "rb"
        ) as gz_file:
            from modelforge.utils.misc import OpenWithLock

            # rather than locking the file we are writing, we will create a lockfile.  the _from_hdf5 function will
            # try to open the same lockfile before reading, so this should prevent issues
            # The use of a lockfile is necessary because h5py will exit immediately if it tries to open a file that is
            # locked by another process.
            with OpenWithLock(
                f"{self.local_cache_dir}/{self.hdf5_data_file['name']}.lockfile", "w"
            ) as lock_file:
                with open(
                    f"{self.local_cache_dir}/{self.hdf5_data_file['name']}", "wb"
                ) as out_file:
                    shutil.copyfileobj(gz_file, out_file)

            # now that the file is written we can safely remove the lockfile
            import os

            os.remove(f"{self.local_cache_dir}/{self.hdf5_data_file['name']}.lockfile")

    def _check_lists(self, list_1: List, list_2: List) -> bool:
        """
        Check to see if all elements in the lists match and the length is the same.

        Note the order of the lists do not matter.

        Parameters
        ----------
        list_1 : List
            First list to compare
        list_2 : List
            Second list to compare

        Returns
        -------
        bool
            True if all elements of sub_list are in containing_list, False otherwise
        """
        if len(list_1) != len(list_2):
            return False
        for a in list_1:
            if a not in list_2:
                return False
        return True

    def _metadata_validation(self, file_name: str, file_path: str) -> bool:
        """
        Validates the metadata file for the npz file.

        Parameters
        ----------
        file_name : str
            Name of the metadata file.
        file_path : str
            Path to the metadata file.

        Returns
        -------
        bool
            True if the metadata file exists, False otherwise.
        """
        if not os.path.exists(f"{file_path}/{file_name}"):
            log.debug(f"Metadata file {file_path}/{file_name} does not exist.")
            return False
        else:
            import json

            from modelforge.utils.misc import OpenWithLock

            with OpenWithLock(f"{file_path}/{file_name}.lockfile", "w") as fl:
                with open(f"{file_path}/{file_name}", "r") as f:
                    self._npz_metadata = json.load(f)

                    if not self._check_lists(
                        self._npz_metadata["data_keys"], self.properties_of_interest
                    ):
                        log.warning(
                            f"Data keys used to generate {file_path}/{file_name} ({self._npz_metadata['data_keys']})"
                        )
                        log.warning(
                            f"do not match data loader ({self.properties_of_interest})."
                        )
                        return False

                    if (
                        self._npz_metadata["hdf5_checksum"]
                        != self.hdf5_data_file["md5"]
                    ):
                        log.warning(
                            f"Checksum for hdf5 file used to generate npz file does not match current file in dataloader."
                        )
                        return False
            os.remove(f"{file_path}/{file_name}.lockfile")
        return True

    def _file_validation(
        self, file_name: str, file_path: str, checksum: str = None
    ) -> bool:
        """
        Validates if the file exists, and if the calculated checksum matches the expected checksum.

        Parameters
        ----------
        file_name : str
            Name of the file to validate.
        file_path : str
            Path to the file.
        checksum : str
            Expected checksum of the file. Default=None
            If None, checksum will not be validated.

        Returns
        -------
        bool
            True if the file exists and the checksum matches, False otherwise.
        """
        full_file_path = f"{file_path}/{file_name}"
        if not os.path.exists(full_file_path):
            log.debug(f"File {full_file_path} does not exist.")
            return False
        elif checksum is not None:
            from modelforge.utils.remote import calculate_md5_checksum

            calculated_checksum = calculate_md5_checksum(file_name, file_path)
            if calculated_checksum != checksum:
                log.warning(
                    f"Checksum mismatch for file {file_path}/{file_name}. Expected {calculated_checksum}, found {checksum}."
                )
                return False
            return True
        else:
            return True

    def _from_hdf5(self) -> None:
        """
        Processes and extracts data from an hdf5 file.

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> processed_data = hdf5_data._from_hdf5()

        """
        from collections import OrderedDict

        import h5py
        import tqdm

        # this will create an unzipped file which we can then load in
        # this is substantially faster than passing gz_file directly to h5py.File()
        # by avoiding data chunking issues.

        temp_hdf5_file = f"{self.local_cache_dir}/{self.hdf5_data_file['name']}"

        if self._file_validation(
            self.hdf5_data_file["name"],
            self.local_cache_dir,
            self.hdf5_data_file["md5"],
        ):
            log.debug(f"Loading unzipped hdf5 file from {temp_hdf5_file}")
        else:
            from modelforge.utils.remote import calculate_md5_checksum

            checksum = calculate_md5_checksum(
                self.hdf5_data_file["name"], self.local_cache_dir
            )
            raise ValueError(
                f"Checksum mismatch for unzipped data file {temp_hdf5_file}. Found {checksum}, Expected {self.hdf5_data_file['md5']}"
            )
        from modelforge.utils.misc import OpenWithLock

        # h5py does file locking internally, but will exit immediately if the file is locked by another program
        # let us create a simple lockfile to prevent this, as OpenWithLock will just wait until the lockfile is unlocked
        # before proceeding
        with OpenWithLock(
            f"{self.local_cache_dir}/{self.hdf5_data_file['name']}.lockfile", "w"
        ) as lock_file:
            with h5py.File(temp_hdf5_file, "r") as hf:
                # create dicts to store data for each format type
                single_rec_data: Dict[str, List[np.ndarray]] = OrderedDict()
                # value shapes: (*)
                single_atom_data: Dict[str, List[np.ndarray]] = OrderedDict()
                # value shapes: (n_atoms, *)
                single_mol_data: Dict[str, List[np.ndarray]] = OrderedDict()
                # value_shapes: (*)
                series_mol_data: Dict[str, List[np.ndarray]] = OrderedDict()
                # value shapes: (n_confs, *)
                series_atom_data: Dict[str, List[np.ndarray]] = OrderedDict()
                # value shapes: (n_confs, n_atoms, *)

                # initialize each relevant value in data dicts to empty list
                for value in self.properties_of_interest:
                    value_format = hf[next(iter(hf.keys()))][value].attrs["format"]
                    if value_format == "single_rec":
                        single_rec_data[value] = []
                    elif value_format == "single_atom":
                        single_atom_data[value] = []
                    elif value_format == "series_mol":
                        series_mol_data[value] = []
                    elif value_format == "series_atom":
                        series_atom_data[value] = []
                    else:
                        raise ValueError(
                            f"Unknown format type {value_format} for property {value}"
                        )
                self.atomic_subsystem_counts = []  # number of atoms in each record
                self.n_confs = []  # number of conformers in each record

                # loop over all records in the hdf5 file and add property arrays to the appropriate dict

                log.debug(f"n_entries: {len(hf.keys())}")

                for record in tqdm.tqdm(list(hf.keys())):
                    # if we have a record with no conformers, we'll skip it to avoid failures
                    if hf[record]["n_configs"][()] != 0:
                        # There may be cases where a specific property of interest
                        # has not been computed for a given record
                        # in that case, we'll want to just skip over that entry
                        property_found = [
                            value in hf[record].keys()
                            for value in self.properties_of_interest
                        ]
                        if all(property_found):
                            # we want to exclude conformers with NaN values for any property of interest
                            configs_nan_by_prop: Dict[str, np.ndarray] = (
                                OrderedDict()
                            )  # ndarray.size (n_configs, )
                            for value in list(series_mol_data.keys()) + list(
                                series_atom_data.keys()
                            ):
                                record_array = hf[record][value][()]
                                configs_nan_by_prop[value] = np.isnan(record_array).any(
                                    axis=tuple(range(1, record_array.ndim))
                                )
                            # check that all values have the same number of conformers

                            if (
                                len(
                                    set(
                                        [
                                            value.shape
                                            for value in configs_nan_by_prop.values()
                                        ]
                                    )
                                )
                                != 1
                            ):
                                val_temp = [
                                    value.shape
                                    for value in configs_nan_by_prop.values()
                                ]
                                raise ValueError(
                                    f"Number of conformers is inconsistent across properties for record {record}: values {val_temp}"
                                )

                            configs_nan = np.logical_or.reduce(
                                list(configs_nan_by_prop.values())
                            )  # boolean array of size (n_configsself.properties_of_interest, )
                            n_confs_rec = sum(~configs_nan)

                            atomic_subsystem_counts_rec = hf[record][
                                next(iter(single_atom_data.keys()))
                            ].shape[0]
                            # all single and series atom properties should have the same number of atoms as the first property

                            self.n_confs.append(n_confs_rec)
                            self.atomic_subsystem_counts.append(
                                atomic_subsystem_counts_rec
                            )

                            for value in single_atom_data.keys():
                                record_array = hf[record][value][()]
                                if record_array.shape[0] != atomic_subsystem_counts_rec:
                                    raise ValueError(
                                        f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                                    )
                                else:
                                    single_atom_data[value].append(record_array)

                            for value in series_atom_data.keys():
                                record_array = hf[record][value][()][~configs_nan]
                                try:
                                    if (
                                        record_array.shape[1]
                                        != atomic_subsystem_counts_rec
                                    ):
                                        raise ValueError(
                                            f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                                        )
                                    else:
                                        series_atom_data[value].append(
                                            record_array.reshape(
                                                n_confs_rec
                                                * atomic_subsystem_counts_rec,
                                                -1,
                                            )
                                        )
                                except IndexError:
                                    log.warning(
                                        f"Property {value} has an index error for record {record}."
                                    )
                                    log.warning(
                                        record_array.shape, atomic_subsystem_counts_rec
                                    )

                            for value in series_mol_data.keys():
                                record_array = hf[record][value][()][~configs_nan]
                                series_mol_data[value].append(record_array)

                            for value in single_rec_data.keys():
                                record_array = hf[record][value][()]
                                single_rec_data[value].append(record_array)

                        else:
                            log.warning(
                                f"Skipping record {record} as not all properties of interest are present."
                            )
                # convert lists of arrays to single arrays

                data = OrderedDict()
                for value in single_atom_data.keys():
                    data[value] = np.concatenate(single_atom_data[value], axis=0)
                for value in single_mol_data.keys():
                    data[value] = np.concatenate(single_mol_data[value], axis=0)
                for value in series_mol_data.keys():
                    data[value] = np.concatenate(series_mol_data[value], axis=0)
                for value in series_atom_data.keys():
                    data[value] = np.concatenate(series_atom_data[value], axis=0)
                for value in single_rec_data.keys():
                    data[value] = np.stack(single_rec_data[value], axis=0)

            self.hdf5data = data
        # we can safely remove the lockfile now that we have read the file
        import os

        os.remove(f"{self.local_cache_dir}/{self.hdf5_data_file['name']}.lockfile")

    def _from_file_cache(self) -> None:
        """
        Loads the processed data from cache.

        Examples
        --------
        """
        # skip validating the checksum, as the npz file checksum of otherwise identical data differs between python 3.11 and 3.9/10
        # we have a metadatafile we validate separately instead
        if self._file_validation(
            self.processed_data_file["name"], self.local_cache_dir, checksum=None
        ):
            if self._metadata_validation(
                self.processed_data_file["name"].replace(".npz", ".json"),
                self.local_cache_dir,
            ):
                log.debug(
                    f"Loading processed data from {self.local_cache_dir}/{self.processed_data_file['name']} generated on {self._npz_metadata['date_generated']}"
                )
                log.debug(
                    f"Properties of Interest in .npz file: {self._npz_metadata['data_keys']}"
                )

                from modelforge.utils.misc import OpenWithLock

                # this will check check for the existence of the lock file and wait until it is unlocked
                # we will just open it as write, since we do not need to read it in; this ensure that we don't have an issue
                # where we have deleted the lock file from a separate, prior process
                with OpenWithLock(
                    f"{self.local_cache_dir}/{self.processed_data_file['name']}.lockfile",
                    "w",
                ) as f:
                    self.numpy_data = np.load(
                        f"{self.local_cache_dir}/{self.processed_data_file['name']}"
                    )
                # we can safely remove the lockfile
                import os

                os.remove(
                    f"{self.local_cache_dir}/{self.processed_data_file['name']}.lockfile"
                )
        else:
            raise ValueError(
                f"Processed data file {self.local_cache_dir}/{self.processed_data_file['name']} not found."
            )

    def _to_file_cache(
        self,
    ) -> None:
        """
                Save processed data to a numpy (.npz) file.
                Parameters
                ----------
        )
                Examples
                --------
                >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
                >>> hdf5_data._to_file_cache()
        """
        log.debug(
            f"Writing npz file to {self.local_cache_dir}/{self.processed_data_file['name']}"
        )
        from modelforge.utils.misc import OpenWithLock

        # we will create a separate lock file that we will check for in the load function to ensure we aren't
        # reading the npz file from a separate process while still writing

        with OpenWithLock(
            f"{self.local_cache_dir}/{self.processed_data_file['name']}.lockfile", "w"
        ) as f:
            np.savez(
                f"{self.local_cache_dir}/{self.processed_data_file['name']}",
                atomic_subsystem_counts=self.atomic_subsystem_counts,
                n_confs=self.n_confs,
                **self.hdf5data,
            )
        # we can safely remove the lockfile
        import os

        os.remove(f"{self.local_cache_dir}/{self.processed_data_file['name']}.lockfile")
        import datetime

        # we will generate a simple metadata file to list which data keys were used to generate the npz file
        # and the checksum of the hdf5 file used to create the npz
        # we can also add in the date of generation so we can report on when the datafile was generated when we load the npz
        metadata = {
            "data_keys": list(self.hdf5data.keys()),
            "hdf5_checksum": self.hdf5_data_file["md5"],
            "hdf5_gz_checkusm": self.gz_data_file["md5"],
            "date_generated": str(datetime.datetime.now()),
        }
        import json

        json_file_path = f"{self.local_cache_dir}/{self.processed_data_file['name'].replace('.npz', '.json')}"
        with OpenWithLock(f"{json_file_path}.lockfile", "w") as fl:
            with open(
                json_file_path,
                "w",
            ) as f:
                json.dump(metadata, f)

        del self.hdf5data


class DatasetFactory:
    """
    Factory for creating TorchDataset instances from HDF5 data.

    Methods are provided to load or process data as needed, handling caching to improve efficiency.
    """

    @staticmethod
    def _load_or_process_data(
        data: HDF5Dataset,
    ) -> None:
        """
        Loads the dataset from cache if available, otherwise processes and caches the data.

        Parameters
        ----------
        data : HDF5Dataset
            The HDF5 dataset instance to use.
        """

        # For efficiency purposes, we first want to see if there is an npz file available before reprocessing the hdf5
        # file, expanding the gzziped archive or download the file.
        # Saving to cache will create an npz file and metadata file.
        # The metadata file will contain the keys used to generate the npz file, the checksum of the hdf5 and gz
        # file used to generate the npz file.  We will look at the metadata file and compare this to the
        # variables saved in the HDF5Dataset class to determine if the npz file is valid.
        # It is important to check the keys used to generate the npz file, as these are allowed to be changed by the user.

        if data._file_validation(
            data.processed_data_file["name"],
            data.local_cache_dir,
        ) and (
            data._metadata_validation(
                data.processed_data_file["name"].replace(".npz", ".json"),
                data.local_cache_dir,
            )
            and not data.force_download
            and not data.regenerate_cache
        ):
            data._from_file_cache()
        # check to see if the hdf5 file exists and the checksum matches
        elif (
            data._file_validation(
                data.hdf5_data_file["name"],
                data.local_cache_dir,
                data.hdf5_data_file["md5"],
            )
            and not data.force_download
        ):
            data._from_hdf5()
            data._to_file_cache()
            data._from_file_cache()
        # if the npz or hdf5 files don't exist/match checksums, call download
        # download will check if the gz file exists and matches the checksum
        # or will use force_download.
        else:
            data._download()
            data._ungzip_hdf5()
            data._from_hdf5()
            data._to_file_cache()
            data._from_file_cache()

    @staticmethod
    def create_dataset(
        data: HDF5Dataset,
    ) -> TorchDataset:
        """
        Creates a TorchDataset from an HDF5Dataset, applying optional transformations.

        Parameters
        ----------
        data : HDF5Dataset
            The HDF5 dataset to convert.

        Returns
        -------
        TorchDataset
            The resulting PyTorch-compatible dataset.
        """

        log.info(f"Creating dataset from {data.url}")
        DatasetFactory._load_or_process_data(data)
        return TorchDataset(data.numpy_data, data._property_names)


from openff.units import unit
from modelforge.custom_types import DatasetType


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: DatasetType,
        splitting_strategy: SplittingStrategy = RandomRecordSplittingStrategy(),
        batch_size: int = 64,
        remove_self_energies: bool = True,
        shift_center_of_mass_to_origin: bool = False,
        atomic_self_energies: Optional[Dict[str, float]] = None,
        regression_ase: bool = False,
        force_download: bool = False,
        version_select: str = "latest",
        local_cache_dir: str = "./",
        regenerate_cache: bool = False,
        regenerate_dataset_statistic: bool = False,
        regenerate_processed_cache: bool = True,
    ):
        """
        Initializes adData module for PyTorch Lightning handling data preparation and loading object with the specified configuration.
        If `remove_self_energies` is `True` and:
        - `self_energies` are passed as a dictionary, these will be used
        - `self_energies` are `None`, `self._ase` will be used
        - `regression_ase` is `True`, self_energies will be calculated

        Parameters
        ---------
            name: Literal["QM9", "ANI1X", "ANI2X", "SPICE1", "SPICE2", "SPICE1_OPENFF"]
                The name of the dataset to use.
            splitting_strategy : SplittingStrategy, defaults to RandomRecordSplittingStrategy
                The strategy to use for splitting the dataset into train, test, and validation sets. .
            batch_size : int, defaults to 64.
                The batch size to use for the dataset.
            remove_self_energies : bool, defaults to True
                Whether to remove the self energies from the dataset.
            shift_center_of_mass_to_origin: bool, defaults to False
                Whether to shift the center of mass of the molecule to the origin. This is necessary if using the
                dipole moment in the loss function.
            atomic_self_energies : Optional[Dict[str, float]]
                A dictionary mapping element names to their self energies. If not provided, the self energies will be calculated.
            regression_ase: bool, defaults to False
                Whether to use the calculated self energies for regression.
            force_download : bool,  defaults to False
                Whether to force the dataset to be downloaded, even if it is already cached.
            version_select : str, defaults to "latest"
                Select the version of the dataset to use. If "latest", the latest version will be used.
                "latest_test" will use the latest test version. Specific versions can be selected by passing the version name
                as defined in the yaml files associated with each dataset.
            local_cache_dir : str, defaults to "./"
                Directory to store the files.
            regenerate_cache : bool, defaults to False
                Whether to regenerate the cache.
        """
        from modelforge.potential.neighbors import Pairlist
        import os

        super().__init__()

        self.name = name
        self.batch_size = batch_size
        self.splitting_strategy = splitting_strategy
        self.remove_self_energies = remove_self_energies
        self.shift_center_of_mass_to_origin = shift_center_of_mass_to_origin
        self.dict_atomic_self_energies = (
            atomic_self_energies  # element name (e.g., 'H') maps to self energies
        )
        self.regression_ase = regression_ase
        self.force_download = force_download
        self.version_select = version_select
        self.regenerate_dataset_statistic = regenerate_dataset_statistic
        self.train_dataset: Optional[TorchDataset] = None
        self.val_dataset: Optional[TorchDataset] = None
        self.test_dataset: Optional[TorchDataset] = None

        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        # create the local cache directory if it does not exist
        os.makedirs(self.local_cache_dir, exist_ok=True)
        self.regenerate_cache = regenerate_cache
        # Use a logical OR to ensure regenerate_processed_cache is True when
        # regenerate_cache is True
        self.regenerate_processed_cache = (
            regenerate_processed_cache or self.regenerate_cache
        )

        self.pairlist = Pairlist()
        self.dataset_statistic_filename = (
            f"{self.local_cache_dir}/{self.name}_dataset_statistic.toml"
        )
        self.cache_processed_dataset_filename = (
            f"{self.local_cache_dir}/{self.name}_{self.version_select}_processed.pt"
        )
        self.lock_file = f"{self.cache_processed_dataset_filename}.lockfile"

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # move all tensors  to the device
        return batch.to_device(device)

    @lock_with_attribute("lock_file")
    def prepare_data(
        self,
    ) -> None:
        """
        Prepares the dataset for use. This method is responsible for the initial
        processing of the data such as calculating self energies, atomic energy
        statistics, and splitting. It is executed only once per node.
        """
        # check if there is a filelock present, if so, wait until it is removed

        # if the dataset has already been processed, skip this step
        if (
            os.path.exists(self.cache_processed_dataset_filename)
            and not self.regenerate_processed_cache
        ):
            if not os.path.exists(self.dataset_statistic_filename):
                raise FileNotFoundError(
                    f"Dataset statistics file {self.dataset_statistic_filename} not found. Please regenerate the cache."
                )
            log.info('Processed dataset already exists. Skipping "prepare_data" step.')
            return None

        # if the dataset is not already processed, process it
        from modelforge.dataset import _ImplementedDatasets

        dataset_class = _ImplementedDatasets.get_dataset_class(str(self.name))
        dataset = dataset_class(
            force_download=self.force_download,
            version_select=self.version_select,
            local_cache_dir=self.local_cache_dir,
            regenerate_cache=self.regenerate_cache,
        )
        torch_dataset = self._create_torch_dataset(dataset)
        # if dataset statistics is present load it from disk
        if (
            os.path.exists(self.dataset_statistic_filename)
            and self.regenerate_dataset_statistic is False
        ):
            log.info(
                f"Loading dataset statistics from disk: {self.dataset_statistic_filename}"
            )
            atomic_self_energies = self._read_atomic_self_energies()
            training_dataset_statistics = self._read_atomic_energies_stats()
        else:
            atomic_self_energies = None
            training_dataset_statistics = None
            # obtain the atomic self energies from the dataset
            dataset_ase = dataset.atomic_self_energies.energies

            # depending on the control flow that is set in the __init__,
            # either use the atomic self energies of the dataset, recalculate
            # it on the fly  or use the provided dictionary
            atomic_self_energies = self._calculate_atomic_self_energies(
                torch_dataset, dataset_ase
            )

        # wrap them in the AtomicSelfEnergy class for processing the dataset
        from modelforge.potential.processing import AtomicSelfEnergies

        log.debug("Process dataset ...")
        self._process_dataset(torch_dataset, AtomicSelfEnergies(atomic_self_energies))

        # calculate the dataset statistic of the dataset
        # This is done __after__ self energies are removed (if requested)
        if training_dataset_statistics is None:
            from modelforge.dataset.utils import calculate_mean_and_variance

            training_dataset_statistics = calculate_mean_and_variance(torch_dataset)
            # wrap everything in a dictionary and save it to disk
            dataset_statistic = {
                "atomic_self_energies": atomic_self_energies,
                "training_dataset_statistics": training_dataset_statistics,
            }

            if atomic_self_energies and training_dataset_statistics:
                log.info(dataset_statistic)
                # save dataset_statistic dictionary to disk as yaml files
                self._log_dataset_statistic(dataset_statistic)
            else:
                raise RuntimeError(
                    "Atomic self energies or atomic energies statistics are missing."
                )

        # Save processed dataset and statistics for later use in setup
        self._cache_dataset(torch_dataset)

    def _log_dataset_statistic(self, dataset_statistic):
        """Save the dataset statistics to a file with units"""
        import toml

        # cast units to string
        atomic_self_energies = {
            key: str(value) if isinstance(value, unit.Quantity) else value
            for key, value in dataset_statistic["atomic_self_energies"].items()
        }
        # cast float and kJ/mol on pytorch tensors and then convert to string
        training_dataset_statistics = {
            key: (
                str(unit.Quantity(value.item(), unit.kilojoule_per_mole))
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in dataset_statistic["training_dataset_statistics"].items()
        }

        dataset_statistic = {
            "atomic_self_energies": atomic_self_energies,
            "training_dataset_statistics": training_dataset_statistics,
        }
        toml.dump(
            dataset_statistic,
            open(
                self.dataset_statistic_filename,
                "w",
            ),
        )
        log.info(
            f"Saving dataset statistics to disk: {self.dataset_statistic_filename}"
        )

    def _read_atomic_self_energies(self) -> Dict[str, Quantity]:
        """Read the atomic self energies from a file."""
        from modelforge.potential.processing import load_atomic_self_energies

        return load_atomic_self_energies(self.dataset_statistic_filename)

    def _read_atomic_energies_stats(self) -> Dict[str, torch.Tensor]:
        """Read the atomic energies statistics from a file."""
        from modelforge.potential.processing import load_dataset_energy_statistics

        return load_dataset_energy_statistics(self.dataset_statistic_filename)

    def _create_torch_dataset(self, dataset):
        """Create a PyTorch dataset from the provided dataset instance."""
        return DatasetFactory().create_dataset(dataset)

    def _process_dataset(
        self, torch_dataset: TorchDataset, atomic_self_energies: "AtomicSelfEnergies"
    ) -> None:
        """Calculate and subtract self energies from the dataset."""
        from modelforge.potential.processing import AtomicSelfEnergies

        if self.regression_ase and not self.dict_atomic_self_energies:
            atomic_self_energies = AtomicSelfEnergies(
                self.calculate_self_energies(torch_dataset)
            )
        self._per_datapoint_operations(torch_dataset, atomic_self_energies)

    def _calculate_atomic_self_energies(
        self, torch_dataset, dataset_ase
    ) -> Dict[str, float]:
        # Use provided ase dictionary
        if self.dict_atomic_self_energies:
            log.info("Using atomic self energies from the provided dictionary.")
            return self.dict_atomic_self_energies

        # Use regression to calculate ase
        elif self.dict_atomic_self_energies is None and self.regression_ase is True:
            log.info("Calculating atomic self energies using regression.")
            return self.calculate_self_energies(torch_dataset)

        # use self energies provided by the dataset (this should be the DEFAULT option)
        elif self.dict_atomic_self_energies is None and self.regression_ase is False:
            log.info("Using atomic self energies provided by the dataset.")
            return dataset_ase
        else:
            raise RuntimeError()

    def _cache_dataset(self, torch_dataset):
        """Cache the dataset and its statistics using PyTorch's serialization."""
        torch.save(torch_dataset, self.cache_processed_dataset_filename)
        # sleep for 5 second to make sure that the dataset was written to disk
        import time

        time.sleep(5)

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up datasets for the train, validation, and test stages based on the stage argument."""

        self.torch_dataset = torch.load(self.cache_processed_dataset_filename)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.splitting_strategy.split(self.torch_dataset)

    def calculate_self_energies(
        self, torch_dataset: TorchDataset, collate: bool = True
    ) -> Dict[str, float]:
        """
        Calculates the self energies for each atomic number in the dataset by performing a least squares regression.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset from which to calculate self energies.
        collate : bool, optional
            If True, uses a custom collate function to gather batch data. Defaults to True.

        Returns
        -------
        Dict[int, float]
            A dictionary mapping atomic numbers to their calculated self energies.
        """
        log.info("Computing self energies for elements in the dataset.")
        from modelforge.dataset.utils import _calculate_self_energies

        # Define the collate function based on the `collate` parameter
        collate_fn = collate_conformers if collate else None
        return _calculate_self_energies(
            torch_dataset=torch_dataset, collate_fn=collate_fn
        )

    def _per_datapoint_operations(
        self, dataset, self_energies: "AtomicSelfEnergies"
    ) -> None:
        """
        Removes the self energies from the total energies for each molecule in the dataset .

        Parameters
        ----------
        dataset: torch.Dataset
            The dataset from which to remove the self energies.
        self_energies : AtomicSelfEnergies
        """

        from tqdm import tqdm

        # remove the self energies if requested
        log.info("Performing per datapoint operations in the dataset dataset")
        if self.remove_self_energies:
            log.info("Removing self energies from the dataset")

            for i in tqdm(range(len(dataset)), desc="Process dataset"):
                start_idx = dataset.single_atom_start_idxs_by_conf[i]
                end_idx = dataset.single_atom_end_idxs_by_conf[i]
                energy = torch.sum(
                    self_energies.ase_tensor_for_indexing[
                        dataset.properties_of_interest["atomic_numbers"][
                            start_idx:end_idx
                        ]
                    ]
                )

                dataset[i] = {"E": dataset.properties_of_interest["E"][i] - energy}

        if self.shift_center_of_mass_to_origin:
            log.info("Shifting the center of mass of each molecule to the origin.")
            from openff.units.elements import MASSES

            for i in tqdm(range(len(dataset)), desc="Process dataset"):
                start_idx = dataset.single_atom_start_idxs_by_conf[i]
                end_idx = dataset.single_atom_end_idxs_by_conf[i]

                atomic_masses = torch.Tensor(
                    [
                        MASSES[atomic_number].m
                        for atomic_number in dataset.properties_of_interest[
                            "atomic_numbers"
                        ][start_idx:end_idx].tolist()
                    ]
                )
                molecule_mass = torch.sum(atomic_masses)

                positions = dataset.properties_of_interest["positions"][
                    start_idx:end_idx
                ]
                center_of_mass = (
                    torch.einsum("i, ij->j", atomic_masses, positions) / molecule_mass
                )
                dataset.properties_of_interest["positions"][
                    start_idx:end_idx
                ] -= center_of_mass

        from torch.utils.data import DataLoader

        all_pairs = []
        n_pairs_per_system_list = [torch.tensor([0], dtype=torch.int16)]

        for batch in tqdm(
            DataLoader(
                dataset,
                batch_size=500,
                collate_fn=collate_conformers,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
                persistent_workers=False,
            ),
            desc="Calculating pairlist for dataset",
        ):
            (
                pairs_batch,
                n_pairs_batch,
            ) = self.pairlist.construct_initial_pairlist_using_numpy(
                batch.nnp_input.atomic_subsystem_indices.to("cpu")
            )
            all_pairs.append(torch.from_numpy(pairs_batch))
            n_pairs_per_system_list.append(torch.from_numpy(n_pairs_batch))

        nr_of_pairs = torch.cat(n_pairs_per_system_list, dim=0)
        nr_of_pairs_in_dataset = torch.cumsum(nr_of_pairs, dim=0, dtype=torch.int64)

        # Determine N (number of tensors) and K (maximum M)
        dataset.properties_of_interest["pair_list"] = torch.cat(all_pairs, dim=1)
        dataset.properties_of_interest["number_of_pairs"] = nr_of_pairs_in_dataset

    def train_dataloader(
        self, num_workers: int = 4, shuffle: bool = True, pin_memory: bool = False
    ) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_conformers,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )

    def val_dataloader(self, num_workers: int = 4) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_conformers,
            num_workers=num_workers,
        )

    def test_dataloader(self, num_workers: int = 4) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_conformers,
            num_workers=num_workers,
        )


def collate_conformers(conf_list: List[BatchData]) -> BatchData:
    """
    Collate a list of BatchData instances into a single BatchData instance.

    Parameters
    ----------
    conf_list : List[BatchData]
        List of BatchData instances.

    Returns
    -------
    BatchData
        Collated batch data.
    """
    atomic_numbers_list = []
    positions_list = []
    total_charge_list = []
    E_list = []  # total energy
    F_list = []  # forces
    ij_list = []
    dipole_moment_list = []
    atomic_subsystem_counts_list = []
    atomic_subsystem_indices_referencing_dataset_list = []

    offset = torch.tensor([0], dtype=torch.int32)
    pair_list_present = (
        True
        if hasattr(conf_list[0].nnp_input, "pair_list")
        and isinstance(conf_list[0].nnp_input.pair_list, torch.Tensor)
        else False
    )

    for conf in conf_list:
        if pair_list_present:
            ## pairlist
            # generate pairlist without padded values
            pair_list = conf.nnp_input.pair_list.to(dtype=torch.int32) + offset
            # update offset (for making sure the pair_list indices are pointing to the correct molecule)
            offset += conf.nnp_input.atomic_numbers.shape[0]
            ij_list.append(pair_list)

        atomic_numbers_list.append(conf.nnp_input.atomic_numbers)
        positions_list.append(conf.nnp_input.positions)
        total_charge_list.append(conf.nnp_input.per_system_total_charge)
        dipole_moment_list.append(conf.metadata.per_system_dipole_moment)
        E_list.append(conf.metadata.per_system_energy)
        F_list.append(conf.metadata.per_atom_force)
        atomic_subsystem_counts_list.append(conf.metadata.atomic_subsystem_counts)
        atomic_subsystem_indices_referencing_dataset_list.append(
            conf.metadata.atomic_subsystem_indices_referencing_dataset
        )

    atomic_subsystem_counts = torch.cat(atomic_subsystem_counts_list)
    atomic_subsystem_indices = torch.repeat_interleave(
        torch.arange(len(conf_list), dtype=torch.int32), atomic_subsystem_counts
    )
    atomic_subsystem_indices_referencing_dataset = torch.cat(
        atomic_subsystem_indices_referencing_dataset_list
    )
    atomic_numbers = torch.cat(atomic_numbers_list)
    total_charge = torch.stack(total_charge_list)
    positions = torch.cat(positions_list).requires_grad_(True)
    F = torch.cat(F_list).to(torch.float64)
    dipole_moment = torch.stack(dipole_moment_list).to(torch.float64)
    E = torch.stack(E_list)
    if pair_list_present:
        IJ_cat = torch.cat(ij_list, dim=1).to(torch.int64)
    else:
        IJ_cat = None

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        per_system_total_charge=total_charge,
        atomic_subsystem_indices=atomic_subsystem_indices,
        pair_list=IJ_cat,
    )
    metadata = Metadata(
        per_system_energy=E,
        per_atom_force=F,
        atomic_subsystem_counts=atomic_subsystem_counts,
        atomic_subsystem_indices_referencing_dataset=atomic_subsystem_indices_referencing_dataset,
        number_of_atoms=atomic_numbers.numel(),
        per_system_dipole_moment=dipole_moment,
    )
    return BatchData(nnp_input, metadata)


from modelforge.dataset.dataset import DatasetFactory
from modelforge.dataset.utils import (
    FirstComeFirstServeSplittingStrategy,
    SplittingStrategy,
)


def initialize_datamodule(
    dataset_name: str,
    version_select: str = "nc_1000_v0",
    batch_size: int = 64,
    splitting_strategy: SplittingStrategy = FirstComeFirstServeSplittingStrategy(),
    remove_self_energies: bool = True,
    shift_center_of_mass_to_origin: bool = False,
    regression_ase: bool = False,
    regenerate_dataset_statistic: bool = False,
    local_cache_dir="./",
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """

    data_module = DataModule(
        dataset_name,
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        version_select=version_select,
        remove_self_energies=remove_self_energies,
        shift_center_of_mass_to_origin=shift_center_of_mass_to_origin,
        regression_ase=regression_ase,
        regenerate_dataset_statistic=regenerate_dataset_statistic,
        local_cache_dir=local_cache_dir,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def single_batch(batch_size: int = 64, dataset_name="QM9", local_cache_dir="./"):
    """
    Utility function to create a single batch of data for testing.
    """
    data_module = initialize_datamodule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        version_select="nc_1000_v0",
        local_cache_dir=local_cache_dir,
    )
    return next(iter(data_module.train_dataloader(shuffle=False)))


def initialize_dataset(
    dataset_name: str,
    local_cache_dir: str,
    versions_select: str = "nc_1000_v0",
    force_download: bool = False,
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """
    from modelforge.dataset import _ImplementedDatasets

    factory = DatasetFactory()
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        local_cache_dir=local_cache_dir,
        version_select=versions_select,
        force_download=force_download,
    )
    dataset = factory.create_dataset(data)

    return dataset

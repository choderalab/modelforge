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

from modelforge.dataset.parameters import DatasetParameters


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

        # since total_charge, force, dipole_moment and spin multiplicity are optional
        # we will set them to zero if they are not present
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
        properties["S"] = (
            torch.from_numpy(dataset[property_name.spin_multiplicity])
            if property_name.spin_multiplicity is not None
            else torch.zeros((dataset[property_name.E].shape[0], 1), dtype=torch.int32)
        )

        properties["partial_charges"] = (
            torch.from_numpy(dataset[property_name.partial_charges])
            if property_name.partial_charges is not None
            else torch.zeros(
                (properties["positions"].shape[0], 1),
                dtype=torch.float32,
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
        spin_state = self.properties_of_interest["S"][idx]
        per_atom_charge = self.properties_of_interest["partial_charges"][
            series_atom_start_idx:series_atom_end_idx
        ]

        nnp_input = NNPInput(
            atomic_numbers=atomic_numbers,
            positions=positions,
            pair_list=self._set_pairlist(idx),
            per_system_total_charge=total_charge,
            per_system_spin_state=spin_state,
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
            per_atom_charge=per_atom_charge,
        )

        return BatchData(nnp_input, metadata)


from abc import ABC, abstractmethod


class HDF5Dataset:

    def __init__(
        self,
        dataset_name: str,
        dataset_cache_dir: str,
        local_cache_dir: str,
        properties_of_interest: List[str],
        properties_assignment: Dict[str, str],
        version_select: str = "latest",
        force_download: bool = False,
        regenerate_processed_dataset: bool = False,
        element_filter: List[tuple] = None,
        local_yaml_file: Optional[str] = None,
    ):
        """
        Initializes the HDF5Dataset class.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        dataset_cache_dir : str
            Directory to store the files.
        """
        import os

        self.dataset_name = dataset_name.lower()
        # make sure we can handle a path with a ~ in it
        self.dataset_cache_dir = os.path.expanduser(dataset_cache_dir)
        self.local_cache_dir = os.path.expanduser(local_cache_dir)

        # make the directories if they don't exist
        os.makedirs(self.dataset_cache_dir, exist_ok=True)
        os.makedirs(self.local_cache_dir, exist_ok=True)

        self.force_download = force_download
        self.regenerate_processed_dataset = regenerate_processed_dataset
        self.element_filter = element_filter
        self.version_select = version_select
        self.local_yaml_file = local_yaml_file

        self._properties_of_interest = properties_of_interest

        self.cache_regenerated = False
        from modelforge.utils import PropertyNames

        self._properties_assignment = PropertyNames(**properties_assignment)

        from loguru import logger

        import yaml

        # if we did not specify a local yaml file, we will look in the yaml_files directory in datasets
        if self.local_yaml_file is None:

            from modelforge.utils.io import get_path_string
            from modelforge.dataset import yaml_files

            yaml_file = get_path_string(yaml_files) + f"/{dataset_name.lower()}.yaml"

            # check to ensure the yaml file exists
            if not os.path.exists(yaml_file):
                raise FileNotFoundError(
                    f"Dataset yaml file {yaml_file} not found. Please check the dataset name."
                )
            logger.debug(f"Loading config data from {yaml_file}")

        # if we have specified a local yaml file, we will use that instead
        if self.local_yaml_file is not None:
            yaml_file = self.local_yaml_file
            # make sure the file exists
            # make sure the file exists
            if not os.path.exists(yaml_file):
                raise FileNotFoundError(
                    f"Local dataset yaml file {yaml_file} not found."
                )
            logger.debug(f"Loading config data from user specified file: {yaml_file}")

        # actually open the yaml file
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        # make sure we have the correct yaml file
        assert data_inputs["dataset"].lower() == self.dataset_name.lower()

        # load the atomic self energies  from the data file if they exist
        if "atomic_self_energies" in data_inputs:

            for key, val in data_inputs["atomic_self_energies"].items():
                data_inputs["atomic_self_energies"][key] = Quantity(val)

            self._ase = data_inputs["atomic_self_energies"]
        else:
            log.warning("No atomic self energies found in the dataset yaml file.")
            self._ase = None

        if self.version_select == "latest":
            # in the yaml file, the entry latest will define the name of the version to use
            dataset_version = data_inputs["latest"]
            logger.info(f"Using the latest dataset: {dataset_version}")
        elif self.version_select == "latest_test":
            dataset_version = data_inputs["latest_test"]
            logger.info(f"Using the latest test dataset: {dataset_version}")
        else:
            dataset_version = self.version_select
            logger.info(f"Using dataset version {dataset_version}")

        self._available_properties = data_inputs[dataset_version][
            "available_properties"
        ]
        if dataset_version not in data_inputs:
            raise ValueError(
                f"Dataset version {dataset_version} not found in {yaml_file}"
            )

        if local_yaml_file is None:

            self.url = data_inputs[dataset_version]["remote_dataset"]["url"]

            # fetch the dictionaries that defined the size, md5 checksums (if provided) and filenames of the data files
            self.gz_data_file_dict = data_inputs[dataset_version]["remote_dataset"][
                "gz_data_file"
            ]
            # dict for the hdf5 file contains checksum and filename
            self.hdf5_data_file_dict = data_inputs[dataset_version]["remote_dataset"][
                "hdf5_data_file"
            ]
        else:
            # if we are using a local yaml file, we will not download
            # or need to unzip, we just need to load the hdf5 file
            # note this assumes that the file name has the path to the file defined
            self.hdf5_data_file_dict = data_inputs[dataset_version]["local_dataset"][
                "hdf5_data_file"
            ]

        self.processed_data_file = self.dataset_name.lower() + ".npz"

    def _acquire_dataset(self) -> None:
        """
        Function to acquire the dataset.

        Note, this wraps logic to check the dataset_cache_dir for appropriate files, to avoid
        downloading the dataset if it already exists (or extracting the .hdf5 file from the .hdf5.gz file).

        This also handles file validation for any local datasets defined in a local_yaml_file.

        """
        # Right now this function needs to be defined for each dataset.
        # once all datasets are moved to zenodo, we should only need a single function defined in the base class
        from modelforge.utils.remote import download_from_url
        from modelforge.utils.misc import OpenWithLock

        # we will lock this while we are checking the file so it doesn't change on us
        with OpenWithLock(
            f"{self.local_cache_dir}/{self.processed_data_file}_checking.lockfile", "w"
        ) as lock_file:
            if (
                os.path.exists(f"{self.local_cache_dir}/{self.processed_data_file}")
                and not self.force_download
                and not self.regenerate_processed_dataset
            ):
                if self._metadata_validation(
                    self.processed_data_file.replace(".npz", ".json"),
                    self.local_cache_dir,
                ):
                    log.debug(
                        f"Unzipped npz file {self.processed_data_file} already exists in {self.local_cache_dir}"
                    )
                    self._from_file_cache()

                    return

        # if we do not use the cached file, we will mark the cache as regenerated
        self.cache_regenerated = True

        # If we are not using a local_yaml_file,
        # (1) we will check the dataset_cache_dir  for the appropriate hdf5 file
        # (2) if that is not available, we will check for the gz file
        # (3) if that is not available, we will download the gz file
        if self.local_yaml_file is None:
            # first check if the appropriate .hdf5 file exists in the dataset_cache_dir and the checksum matches
            if (
                self._file_validation(
                    file_name=self.hdf5_data_file_dict["file_name"],
                    file_path=self.dataset_cache_dir,
                    checksum=self.hdf5_data_file_dict["md5"],
                )
                and not self.force_download
            ):
                log.debug(
                    f"Unzipped hdf5 file {self.hdf5_data_file_dict['file_name']} already exists in {self.dataset_cache_dir}"
                )
                self._from_hdf5()
                self._to_file_cache()
                self._from_file_cache()
            # If the .hdf5 file didn't exist or the checksum didn't match, we will next see if the .gz file exists
            # If it doesn't, we will download it. Fortuitously, download_from_url will do both these functions for us
            # so we can just use a single else statement.
            else:
                log.debug(
                    f"hdf5 file {self.hdf5_data_file_dict['file_name']} not found."
                )
                download_from_url(
                    url=self.url,
                    md5_checksum=self.gz_data_file_dict["md5"],
                    output_path=self.dataset_cache_dir,
                    output_filename=self.gz_data_file_dict["file_name"],
                    length=self.gz_data_file_dict["length"],
                    force_download=self.force_download,
                )
                self._ungzip_hdf5()
                self._from_hdf5()
                self._to_file_cache()
                self._from_file_cache()

        # if we are using a local yaml file, we will not download
        # we will just check if the hdf5 file exists and has the right checksum
        # Note: I think we still should require the checksum to be defined in the local yaml to ensure that the
        # user is actually using the correct file.
        else:
            log.debug("Using local yaml file for dataset.")
            print(self.hdf5_data_file_dict)
            if not self._file_validation(
                file_name=self.hdf5_data_file_dict["file_name"],
                checksum=self.hdf5_data_file_dict["md5"],
            ):
                raise ValueError(
                    f"File {self.hdf5_data_file_dict['file_name']} does not exist in {self.dataset_cache_dir} or the checksum does not match."
                )
            self._from_hdf5()
            self._to_file_cache()
            self._from_file_cache()

    @property
    def atomic_self_energies(self):
        from modelforge.potential.processing import AtomicSelfEnergies

        if self._ase is None:
            return None

        return AtomicSelfEnergies(energies=self._ase)

    @property
    def properties_of_interest(self) -> List[str]:
        """
        Getter for the properties of interest.
        The order of this list determines also the order provided in the __getitem__ call
        from the PytorchDataset.

        Returns
        -------
        List[str]
            List of properties of interest.

        """
        return self._properties_of_interest

    # add setter for properties of interest
    @properties_of_interest.setter
    def properties_of_interest(self, properties_of_interest: List[str]) -> None:
        """
        Setter for the properties of interest.
        The order of this list determines also the order provided in the __getitem__ call
        from the PytorchDataset.

        Parameters
        ----------
        properties_of_interest : List[str]
            List of properties of interest.

        """
        # first check to ensure the properties of interest are in the available properties
        for prop in properties_of_interest:
            if prop not in self._available_properties:
                raise ValueError(
                    f"Property {prop} is not available in the dataset. Available properties are: {self._available_properties}"
                )

        self._properties_of_interest = properties_of_interest

    @property
    def properties_assignment(self) -> "PropertyNames":
        """
        Getter for the properties assignment.

        Returns
        -------
        Dict[str, str]
            Dictionary of properties assignment.

        """
        return self._properties_assignment

    @property
    def available_properties(self) -> List[str]:
        """
        List of available properties in the dataset.

        Returns
        -------
        List[str]
            List of available properties in the dataset.

        Examples
        --------
        >>> data = HDF5Dataset()
        >>> data.available_properties
        ['geometry', 'atomic_numbers', 'return_energy']
        """
        return self._available_properties

    def _ungzip_hdf5(self) -> None:
        """
        Unzips an HDF5.gz file.

        Examples
        -------
        """
        import gzip
        import shutil
        from modelforge.utils.misc import OpenWithLock
        import os

        with OpenWithLock(
            f"{self.dataset_cache_dir}/{self.gz_data_file_dict['file_name']}.lockfile",
            "w",
        ) as lock_file_gz:
            with gzip.open(
                f"{self.dataset_cache_dir}/{self.gz_data_file_dict['file_name']}", "rb"
            ) as gz_file:

                # rather than locking the file we are writing, we will create a lockfile.  the _from_hdf5 function will
                # try to open the same lockfile before reading, so this should prevent issues
                # The use of a lockfile is necessary because h5py will exit immediately if it tries to open a file that is
                # locked by another process.
                with OpenWithLock(
                    f"{self.dataset_cache_dir}/{self.hdf5_data_file_dict['file_name']}.lockfile",
                    "w",
                ) as lock_file:
                    with open(
                        f"{self.dataset_cache_dir}/{self.hdf5_data_file_dict['file_name']}",
                        "wb",
                    ) as out_file:
                        shutil.copyfileobj(gz_file, out_file)

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

                    if self._npz_metadata["element_filter"] != str(self.element_filter):
                        log.warning(
                            "Element filter for hdf5 file used to generate npz file does not match current file in dataloader."
                        )
                        return False
                    if (
                        self._npz_metadata["hdf5_checksum"]
                        != self.hdf5_data_file_dict["md5"]
                    ):
                        log.warning(
                            f"Checksum for hdf5 file used to generate npz file does not match current file in dataloader."
                        )
                        return False
        return True

    @staticmethod
    def _file_validation(
        file_name: str, file_path: str = None, checksum: str = None
    ) -> bool:
        """
        Validates if the file exists, and if the calculated checksum matches the expected checksum.

        Parameters
        ----------
        file_name : str
            Name of the file to validate.
        file_path : str
            Path to the file to validate. Default = None.
            If None, the code will assume the file_path is defined as part of file_name
            If define, the file_path will be prepended to the file_name, i.e., {file_path}/{file_name}
        checksum : str
            Expected checksum of the file. Default=None
            If None, checksum will not be validated.

        Returns
        -------
        bool
            True if the file exists and the checksum matches, False otherwise.
        """
        from modelforge.utils.misc import OpenWithLock

        if file_path is not None:
            file_path = os.path.expanduser(file_path)
            full_file_path = f"{file_path}/{file_name}"
        else:
            full_file_path = os.path.expanduser(file_name)

        print(f"Validating file {full_file_path}")
        with OpenWithLock(f"{full_file_path}.lockfile", "w") as lock_file:
            if not os.path.exists(full_file_path):
                log.debug(f"File {full_file_path} does not exist.")
                return False
            elif checksum is not None:
                from modelforge.utils.remote import calculate_md5_checksum

                calculated_checksum = calculate_md5_checksum(file_name, file_path)
                if calculated_checksum != checksum:
                    log.warning(
                        f"Checksum mismatch for file {file_path}/{file_name}. Expected {checksum}, found {calculated_checksum}."
                    )
                    return False
                return True
            else:
                return True

    def _satisfy_element_filter(self, data):
        result = True
        if self.element_filter is None:
            pass
        else:
            for each_filter in self.element_filter:
                result = True
                try:
                    for each_element in each_filter:
                        if each_element > 0:
                            result = result and np.isin(each_element, data)
                        elif each_element < 0:
                            result = result and not np.isin(-each_element, data)
                        else:
                            raise ValueError(
                                f"Invalid atomic number input: {each_element}! "
                                f"Please input a valid atomic number."
                            )
                except TypeError:
                    raise TypeError(
                        "Please use atomic number to refer to element types!"
                    )
                # If any element filters are true,
                # then we include because sub-lists comparison is an OR operator
                if result:
                    result = bool(result)
                    break

        return result

    def _from_hdf5(self) -> None:
        """
        Processes and extracts data from an hdf5 file.

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> hdf5_data._from_hdf5()

        """
        from collections import OrderedDict
        from modelforge.utils.units import GlobalUnitSystem, chem_context

        import h5py
        import tqdm

        #
        if self.local_yaml_file is None:
            temp_hdf5_file = (
                f"{self.dataset_cache_dir}/{self.hdf5_data_file_dict['file_name']}"
            )

        else:
            # if a local yaml file is provided, the "name" field provides the full path to the file
            # we will expand the path to handle ~ if provided
            temp_hdf5_file = os.path.expanduser(self.hdf5_data_file_dict["file_name"])

        from modelforge.utils.misc import OpenWithLock

        log.debug(f"Reading data from {temp_hdf5_file}")
        log.debug(f"element filter: {self.element_filter}")
        # h5py does file locking internally, but will exit immediately if the file is locked by another program
        # we create a simple lockfile to prevent this, as OpenWithLock will just wait until the lockfile is unlocked
        # before proceeding
        with OpenWithLock(f"{temp_hdf5_file}.lockfile", "w") as lock_file:
            with h5py.File(temp_hdf5_file, "r") as hf:
                # create dicts to store data for each format type

                # value shapes: (n_atoms, *)
                atomic_numbers_data: Dict[str, List[np.ndarray]] = OrderedDict()

                # value_shapes: (n_confs, *)
                per_system_data: Dict[str, List[np.ndarray]] = OrderedDict()

                # value shapes: (n_confs, n_atoms, *)
                per_atom_data: Dict[str, List[np.ndarray]] = OrderedDict()

                # initialize each relevant value in data dicts to empty list

                for value in self.properties_of_interest:
                    value_format = hf[next(iter(hf.keys()))][value].attrs["format"]
                    if value_format == "atomic_numbers":
                        atomic_numbers_data[value] = []
                    elif value_format == "per_system":
                        per_system_data[value] = []
                    elif value_format == "per_atom":
                        per_atom_data[value] = []
                    else:
                        raise ValueError(
                            f"Unknown format type {value_format} for property {value}"
                        )

                log.debug(f"Properties of Interest: {self.properties_of_interest}")
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

                        # filter by elements
                        satisfy_element_filter = self._satisfy_element_filter(
                            hf[record]["atomic_numbers"]
                        )
                        # we want to exclude a record if the element filter is not satisfied
                        # or if we don't have all properties of interest (i.e., an incomplete record)

                        if all(property_found) and satisfy_element_filter:

                            # we want to exclude configurations with NaN values for any property of interest
                            configs_nan_by_prop: Dict[str, np.ndarray] = (
                                OrderedDict()
                            )  # ndarray.size (n_configs, )

                            # loop over the properties in the per_system_data and per_atom_data dicts
                            for value in list(per_system_data.keys()) + list(
                                per_atom_data.keys()
                            ):
                                # fetch the array from the hdf5 file
                                record_array = hf[record][value][()]

                                # This will generate a boolean array of size (n_configs, )
                                # # where True indicates that the property has NaN values
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
                            )  # boolean array of size (n_config, self.properties_of_interest, )
                            n_confs_rec = sum(~configs_nan)

                            atomic_subsystem_counts_rec = hf[record][
                                next(iter(atomic_numbers_data.keys()))
                            ].shape[0]

                            self.n_confs.append(n_confs_rec)
                            self.atomic_subsystem_counts.append(
                                atomic_subsystem_counts_rec
                            )

                            for value in atomic_numbers_data.keys():
                                record_array = hf[record][value][()]

                                if record_array.shape[0] != atomic_subsystem_counts_rec:
                                    raise ValueError(
                                        f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                                    )
                                else:
                                    atomic_numbers_data[value].append(record_array)

                            for value in per_atom_data.keys():
                                record_array = hf[record][value][()][~configs_nan]
                                if "u" in hf[record][value].attrs:
                                    units = hf[record][value].attrs["u"]
                                    property_type = hf[record][value].attrs[
                                        "property_type"
                                    ]

                                    if units != "dimensionless":
                                        record_array = Quantity(record_array, units).to(
                                            GlobalUnitSystem.get_units(property_type),
                                            "chem",
                                        )
                                        record_array = record_array.magnitude

                                try:
                                    if (
                                        record_array.shape[1]
                                        != atomic_subsystem_counts_rec
                                    ):
                                        raise ValueError(
                                            f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                                        )
                                    else:
                                        per_atom_data[value].append(
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
                                        record_array.shape,
                                        atomic_subsystem_counts_rec,
                                    )

                            for value in per_system_data.keys():

                                record_array = hf[record][value][()][~configs_nan]
                                if "u" in hf[record][value].attrs:
                                    units = hf[record][value].attrs["u"]
                                    property_type = hf[record][value].attrs[
                                        "property_type"
                                    ]

                                    if units != "dimensionless":
                                        record_array = Quantity(record_array, units).to(
                                            GlobalUnitSystem.get_units(property_type),
                                            "chem",
                                        )
                                        record_array = record_array.magnitude
                                per_system_data[value].append(record_array)

                        else:
                            log.warning(
                                f"Skipping record {record} as not all properties of interest are present."
                            )
                # convert lists of arrays to single arrays

                data = OrderedDict()
                for value in atomic_numbers_data.keys():
                    data[value] = np.concatenate(atomic_numbers_data[value], axis=0)
                for value in per_system_data.keys():
                    data[value] = np.concatenate(per_system_data[value], axis=0)
                for value in per_atom_data.keys():
                    data[value] = np.concatenate(per_atom_data[value], axis=0)

            self.hdf5data = data

    def _from_file_cache(self) -> None:
        """
        Loads the processed data from cache.

        Examples
        --------
        """
        # skip validating the checksum, as the npz file checksum of otherwise identical data differs between python 3.11 and 3.9/10
        # we have a metadata file we validate separately instead
        if self._file_validation(
            self.processed_data_file, self.local_cache_dir, checksum=None
        ):
            if self._metadata_validation(
                self.processed_data_file.replace(".npz", ".json"),
                self.local_cache_dir,
            ):
                log.debug(
                    f"Loading processed data from {self.local_cache_dir}/{self.processed_data_file} generated on {self._npz_metadata['date_generated']}"
                )
                log.debug(
                    f"Properties of Interest in .npz file: {self._npz_metadata['data_keys']}"
                )

                from modelforge.utils.misc import OpenWithLock

                # this will check check for the existence of the lock file and wait until it is unlocked
                # we will just open it as write, since we do not need to read it in; this ensure that we don't have an issue
                # where we have deleted the lock file from a separate, prior process
                with OpenWithLock(
                    f"{self.local_cache_dir}/{self.processed_data_file}.lockfile",
                    "w",
                ) as f:
                    self.numpy_data = np.load(
                        f"{self.local_cache_dir}/{self.processed_data_file}"
                    )
                    log.debug("Loaded processed data from cache.")

        else:
            raise ValueError(
                f"Processed data file {self.local_cache_dir}/{self.processed_data_file} not found."
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
            f"Writing npz file to {self.local_cache_dir}/{self.processed_data_file}"
        )
        from modelforge.utils.misc import OpenWithLock

        # we will create a separate lock file that we will check for in the load function to ensure we aren't
        # reading the npz file from a separate process while still writing

        with OpenWithLock(
            f"{self.local_cache_dir}/{self.processed_data_file}.lockfile",
            "w",
        ) as f:
            np.savez(
                f"{self.local_cache_dir}/{self.processed_data_file}",
                atomic_subsystem_counts=self.atomic_subsystem_counts,
                n_confs=self.n_confs,
                **self.hdf5data,
            )
        # we can safely remove the lockfile
        import os

        import datetime

        # we will generate a simple metadata file to list which data keys were used to generate the npz file
        # and the checksum of the hdf5 file used to create the npz
        # we can also add in the date of generation so we can report on when the datafile was generated when we load the npz

        metadata = {
            "data_keys": list(self.hdf5data.keys()),
            "element_filter": str(self.element_filter),
            "hdf5_checksum": self.hdf5_data_file_dict["md5"],
            "date_generated": str(datetime.datetime.now()),
        }
        import json

        json_file_path = f"{self.local_cache_dir}/{self.processed_data_file.replace('.npz', '.json')}"
        with OpenWithLock(f"{json_file_path}.lockfile", "w") as fl:
            with open(
                json_file_path,
                "w",
            ) as f:
                json.dump(metadata, f)

        del self.hdf5data


from openff.units import unit
from modelforge.custom_types import DatasetType


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        name: str,
        properties_of_interest: List[str],
        properties_assignment: Dict[str, str],
        splitting_strategy: SplittingStrategy = RandomRecordSplittingStrategy(),
        batch_size: int = 64,
        remove_self_energies: bool = True,
        shift_center_of_mass_to_origin: bool = False,
        atomic_self_energies: Optional[Dict[str, float]] = None,
        regression_ase: bool = False,
        force_download: bool = False,
        regenerate_processed_dataset: bool = False,
        version_select: str = "latest",
        local_cache_dir: str = "./",
        dataset_cache_dir: str = "./",
        element_filter: Optional[List[tuple]] = None,
        local_yaml_file: Optional[str] = None,
    ):
        """
        Initializes a DataModule for PyTorch Lightning handling data preparation and loading object with the specified configuration.
        If `remove_self_energies` is `True` and:
        - If `atomic_self_energies` are passed as a dictionary, these will be used
        - If we have not provided a dict for `atomic_self_energies` and  regression_ase is 'True', self energies will be calculated via regression
        - Otherwise, ASE values provided with the dataset will be used; if they are not provided in the dataset yaml, an error will be raised
        If `remove_self_energies` is `False`, the self energies from the dataset will be queried and written to the dataset statistics file

        Parameters
        ---------
            name: Literal["QM9", "ANI1X", "ANI2X", "SPICE1", "SPICE2", "SPICE1_OPENFF"]
                The name of the dataset to use.
            properties_of_interest : List[str]
                The properties to include in the dataset.
            properties_assignment : Dict[str, str]
                The properties of interest from the hdf5 dataset to associate with internal properties with the code.
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
                If True, calculate atomic self energies (ASE) using regression, rather than provided values or those in the dataset.
            force_download : bool,  defaults to False
                Whether to force the dataset to be downloaded, even if it is already cached.
            regenerate_processed_dataset : bool, defaults to False
                Whether to regenerate the .npz file cached in local_cache_dir.
            version_select : str, defaults to "latest"
                Select the version of the dataset to use. If "latest", the latest version will be used.
                "latest_test" will use the latest test version. Specific versions can be selected by passing the version name
                as defined in the yaml files associated with each dataset.
            local_cache_dir : str, defaults to "./"
                Directory to store the files associated/specific with a given dataset/training run.
            dataset_cache_dir : str, defaults to "./"
                Directory to store the dataset files,

            local_yaml_file : Optional[str]
                Path to the local yaml file to use for the dataset. If not provided, the code will search for a
                yaml file associated with the dataset name in the modelforge.dataset.yaml_files directory.
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
        self.regenerate_processed_dataset = regenerate_processed_dataset
        self.version_select = version_select
        self.train_dataset: Optional[TorchDataset] = None
        self.val_dataset: Optional[TorchDataset] = None
        self.test_dataset: Optional[TorchDataset] = None

        self.properties_of_interest = properties_of_interest
        self.properties_assignment = properties_assignment
        self.element_filter = element_filter
        self.local_yaml_file = local_yaml_file

        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.dataset_cache_dir = os.path.expanduser(dataset_cache_dir)

        # create the local cache directory if it does not exist
        os.makedirs(self.local_cache_dir, exist_ok=True)
        os.makedirs(self.dataset_cache_dir, exist_ok=True)

        self.pairlist = Pairlist()
        self.dataset_statistic_filename = (
            f"{self.local_cache_dir}/{self.name}_dataset_statistic.toml"
        )
        self.cache_processed_dataset_filename = (
            f"{self.local_cache_dir}/{self.name}_{self.version_select}_processed.pt"
        )
        self.lock_file = f"{self.cache_processed_dataset_filename}.lockfile"
        self.torch_dataset = None

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

        if self.properties_of_interest is None:
            raise ValueError(
                "Properties of interest must be provided. Please set properties_of_interest."
            )
        if self.properties_assignment is None:
            raise ValueError(
                "Properties assignment must be provided. Please set properties_assignment."
            )

        dataset = HDF5Dataset(
            dataset_name=self.name,
            force_download=self.force_download,
            version_select=self.version_select,
            properties_of_interest=self.properties_of_interest,
            properties_assignment=self.properties_assignment,
            local_cache_dir=self.local_cache_dir,
            dataset_cache_dir=self.dataset_cache_dir,
            element_filter=self.element_filter,
            local_yaml_file=self.local_yaml_file,
            regenerate_processed_dataset=self.regenerate_processed_dataset,
        )

        torch_dataset = self._create_torch_dataset(dataset)

        if self.remove_self_energies:
            # all the logic described in the doc string above regarding
            # whether to regress , user provided, or from the dataset
            # are handled by this function
            # note this will return an instance of the AtomicSelfEnergies class
            # not a dictionary
            atomic_self_energies = self._calculate_atomic_self_energies(
                torch_dataset, dataset.atomic_self_energies
            )

        else:
            # if we are not removing self energies, we do not actually use these values
            # but we still need something for the dataset statistic writer
            atomic_self_energies = dataset.atomic_self_energies

        # perform the operations to process the dataset which may include:
        # 1. removing self energies
        # 2. shifting the center of mass to the origin
        # 3. calculating the pairlist
        log.debug("Process dataset ...")
        self._per_datapoint_operations(torch_dataset, atomic_self_energies)

        # calculate the dataset statistic of the dataset
        # This is done __after__ self energies are removed (if requested)
        # note, this is a fast operation and thus we will just recalculate this every time to ensure
        # there are no issues with an old statistic file hanging around in a directory
        from modelforge.dataset.utils import calculate_mean_and_variance

        training_dataset_statistics = calculate_mean_and_variance(torch_dataset)

        # wrap everything in a dictionary and save it to disk

        dataset_statistic = {
            "atomic_self_energies": atomic_self_energies.energies,
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
        self.torch_dataset = torch_dataset

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
        dataset._acquire_dataset()
        return TorchDataset(dataset.numpy_data, dataset.properties_assignment)

    def _calculate_atomic_self_energies(
        self, torch_dataset, dataset_ase
    ) -> "AtomicSelfEnergies":
        """
        This function wraps the logic for the source of the atomic self energies,
        i.e., from a provided dictionary, from the dataset, or calculated via regression.
        """
        from modelforge.potential.processing import AtomicSelfEnergies

        # if a dictionary of atomic self energies is provided, we will use that
        if self.dict_atomic_self_energies and self.regression_ase is False:
            log.info("Using atomic self energies from the provided dictionary.")
            atomic_self_energies = AtomicSelfEnergies(self.dict_atomic_self_energies)
        # if we have not been provided ASE, but regression_ase is True, we will calculate them
        elif self.regression_ase is True:
            log.info("Calculating atomic self energies using regression.")
            atomic_self_energies = AtomicSelfEnergies(
                self.calculate_self_energies(torch_dataset)
            )
        # if we have not been provided ASE and regression_ase is False, we will use the ASE values provided by the dataset
        else:
            log.info("Using atomic self energies provided by the dataset.")
            atomic_self_energies = dataset_ase
            if atomic_self_energies is None:
                raise ValueError(
                    "Atomic self energies are required. Provide them or set regression_ase = True."
                )

        return atomic_self_energies

    def _cache_dataset(self, torch_dataset):
        """Cache the dataset and its statistics using PyTorch's serialization."""
        torch.save(torch_dataset, self.cache_processed_dataset_filename)
        # sleep for 5 second to make sure that the dataset was written to disk
        import time

        time.sleep(5)

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up datasets for the train, validation, and test stages based on the stage argument."""

        if self.torch_dataset is None:
            self.torch_dataset = torch.load(
                self.cache_processed_dataset_filename, weights_only=False
            )

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

                start_idx_mol = dataset.series_atom_start_idxs_by_conf[i]
                end_idx_mol = dataset.series_atom_start_idxs_by_conf[i + 1]

                positions = dataset.properties_of_interest["positions"][
                    start_idx_mol:end_idx_mol
                ]
                center_of_mass = (
                    torch.einsum("i, ij->j", atomic_masses, positions) / molecule_mass
                )
                dataset.properties_of_interest["positions"][
                    start_idx_mol:end_idx_mol
                ] -= center_of_mass

        from torch.utils.data import DataLoader

        all_pairs = []
        n_pairs_per_system_list = [torch.tensor([0], dtype=torch.int16)]

        for batch in tqdm(
            DataLoader(
                dataset,
                batch_size=200,
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
    S_list = []
    E_list = []  # total energy
    F_list = []  # forces
    ij_list = []
    dipole_moment_list = []
    per_atom_charge_list = []
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
        per_atom_charge_list.append(conf.metadata.per_atom_charge)

        E_list.append(conf.metadata.per_system_energy)
        F_list.append(conf.metadata.per_atom_force)
        S_list.append(conf.nnp_input.per_system_spin_state)
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
    total_charge = torch.stack(total_charge_list).to(torch.float32)
    positions = torch.cat(positions_list).requires_grad_(True)
    F = torch.cat(F_list).to(torch.float64)
    dipole_moment = torch.stack(dipole_moment_list).to(torch.float32)
    per_atom_charge = torch.cat(per_atom_charge_list).to(torch.float32)
    E = torch.stack(E_list)
    spin_multiplicity = torch.cat(S_list).to(torch.float32)
    if pair_list_present:
        IJ_cat = torch.cat(ij_list, dim=1).to(torch.int64)
    else:
        IJ_cat = None

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        per_system_total_charge=total_charge,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_spin_state=spin_multiplicity,
        pair_list=IJ_cat,
    )
    metadata = Metadata(
        per_system_energy=E,
        per_atom_force=F,
        atomic_subsystem_counts=atomic_subsystem_counts,
        atomic_subsystem_indices_referencing_dataset=atomic_subsystem_indices_referencing_dataset,
        number_of_atoms=atomic_numbers.numel(),
        per_system_dipole_moment=dipole_moment,
        per_atom_charge=per_atom_charge,
    )

    return BatchData(nnp_input, metadata)


# from modelforge.dataset.dataset import DatasetFactory
from modelforge.dataset.utils import (
    FirstComeFirstServeSplittingStrategy,
    SplittingStrategy,
)


def initialize_datamodule(
    dataset_name: str,
    version_select: str,
    batch_size: int = 64,
    splitting_strategy: SplittingStrategy = FirstComeFirstServeSplittingStrategy(),
    remove_self_energies: bool = True,
    shift_center_of_mass_to_origin: bool = False,
    regression_ase: bool = False,
    local_cache_dir="./",
    dataset_cache_dir="./",
    properties_of_interest: Optional[PropertyNames] = None,
    properties_assignment: Optional[Dict[str, str]] = None,
    element_filter: Optional[List[tuple]] = None,
    local_yaml_file: Optional[str] = None,
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """

    data_module = DataModule(
        name=dataset_name,
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        version_select=version_select,
        remove_self_energies=remove_self_energies,
        shift_center_of_mass_to_origin=shift_center_of_mass_to_origin,
        regression_ase=regression_ase,
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        properties_of_interest=properties_of_interest,
        properties_assignment=properties_assignment,
        element_filter=element_filter,
        local_yaml_file=local_yaml_file,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


def single_batch(
    batch_size: int = 64,
    dataset_name="QM9",
    local_cache_dir="./",
    dataset_cache_dir="./",
    version_select="nc_1000_v1.1",
    properties_of_interest=[
        "atomic_numbers",
        "positions",
        "internal_energy_at_0K",
        "dipole_moment_per_system",
    ],
    properties_assignment={
        "atomic_numbers": "atomic_numbers",
        "positions": "positions",
        "E": "internal_energy_at_0K",
        "dipole_moment": "dipole_moment_per_system",
    },
    shift_center_of_mass_to_origin: bool = False,
):
    """
    Utility function to create a single batch of data for testing (default returns qm9)

    Parameters
    ----------
    batch_size : int
        The size of the batch to create.
    dataset_name : str
        The name of the dataset to use. default is "QM9".
    local_cache_dir : str
        Local cache directory for the processed dataset.
    dataset_cache_dir : str
        Directory for the raw dataset files.
    version_select : str
        The version of the dataset to use. default is "nc_1000_v1.1".
    properties_of_interest : list
        List of properties to include in the dataset.
        Default is ["atomic_numbers", "positions", "internal_energy_at_0K", "dipole_moment_per_system"] for qm9.
    properties_assignment : dict
        Dictionary mapping properties of interest to their corresponding keys in the dataset.
        Default is {"atomic_numbers": "atomic_numbers", "positions": "positions", "E": "internal_energy_at_0K", "dipole_moment": "dipole_moment_per_system"} for qm9.
    shift_center_of_mass_to_origin: bool
        Whether to shift the center of mass of the molecule to the origin. This is necessary if computing the dipole moment or quadrupole moment.
    """
    data_module = initialize_datamodule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        version_select=version_select,
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        properties_of_interest=properties_of_interest,
        properties_assignment=properties_assignment,
        shift_center_of_mass_to_origin=shift_center_of_mass_to_origin,
    )
    return next(iter(data_module.train_dataloader(shuffle=False)))


def initialize_dataset(
    dataset_name: str,
    local_cache_dir: str,
    dataset_cache_dir: str,
    version_select: str,
    properties_of_interest=List[str],
    properties_assignment=Dict[str, str],
    force_download: bool = False,
    local_yaml_file: Optional[str] = None,
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """

    dataset = HDF5Dataset(
        dataset_name=dataset_name,
        force_download=force_download,
        version_select=version_select,
        properties_of_interest=properties_of_interest,
        properties_assignment=properties_assignment,
        local_cache_dir=local_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
        local_yaml_file=local_yaml_file,
    )

    return dataset

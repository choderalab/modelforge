import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union, NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as log
from openff.units import Quantity
from torch.utils.data import DataLoader

from modelforge.dataset.utils import RandomRecordSplittingStrategy, SplittingStrategy
from modelforge.utils.prop import PropertyNames

if TYPE_CHECKING:
    from modelforge.potential.processing import AtomicSelfEnergies


@dataclass
class DatasetStatistics:
    E_i_mean: float
    E_i_stddev: float
    atomic_self_energies: "AtomicSelfEnergies"
    atomic_self_energies_removed: bool = False


@dataclass(frozen=False)
class Metadata:
    """
    A NamedTuple to structure the inputs for neural network potentials.

    Parameters
    ----------
    """

    E: torch.Tensor
    atomic_subsystem_counts: torch.Tensor
    atomic_subsystem_indices_referencing_dataset: torch.Tensor
    number_of_atoms: int
    F: torch.Tensor = torch.tensor([], dtype=torch.float32)

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """Move all tensors in this instance to the specified device."""
        if device:
            self.E = self.E.to(device)
            self.F = self.F.to(device)
            self.atomic_subsystem_counts = self.atomic_subsystem_counts.to(device)
            self.atomic_subsystem_indices_referencing_dataset = (
                self.atomic_subsystem_indices_referencing_dataset.to(device)
            )
        if dtype:
            self.E = self.E.to(dtype)
            self.F = self.F.to(dtype)
        return self


@dataclass
class NNPInput:
    """
    A dataclass to structure the inputs for neural network potentials.

    Attributes
    ----------
    atomic_numbers : torch.Tensor
        A 1D tensor containing atomic numbers for each atom in the system(s).
        Shape: [num_atoms], where `num_atoms` is the total number of atoms across all systems.
    positions : torch.Tensor
        A 2D tensor of shape [num_atoms, 3], representing the XYZ coordinates of each atom.
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule.
        This allows for calculations involving multiple molecules or subsystems within the same batch.
        Shape: [num_atoms].
    total_charge : torch.Tensor
        A tensor with the total charge of molecule.
        Shape: [num_systems], where `num_systems` is the number of molecules.
    """

    atomic_numbers: torch.Tensor
    positions: Union[torch.Tensor, Quantity]
    atomic_subsystem_indices: torch.Tensor
    total_charge: torch.Tensor
    pair_list: Optional[torch.Tensor] = None

    def to(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move all tensors in this instance to the specified device/dtype."""

        if device:
            self.atomic_numbers = self.atomic_numbers.to(device)
            self.positions = self.positions.to(device)
            self.atomic_subsystem_indices = self.atomic_subsystem_indices.to(device)
            self.total_charge = self.total_charge.to(device)
            if self.pair_list is not None:
                self.pair_list = self.pair_list.to(device)
        if dtype:
            self.positions = self.positions.to(dtype)
        return self

    def __post_init__(self):
        # Set dtype and convert units if necessary
        self.atomic_numbers = self.atomic_numbers.to(torch.int32)
        self.atomic_subsystem_indices = self.atomic_subsystem_indices.to(torch.int32)
        self.total_charge = self.total_charge.to(torch.int32)

        # Unit conversion for positions
        if isinstance(self.positions, Quantity):
            positions = self.positions.to(unit.nanometer).m
            self.positions = torch.tensor(
                positions, dtype=torch.float32, requires_grad=True
            )

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        if self.atomic_numbers.dim() != 1:
            raise ValueError("atomic_numbers must be a 1D tensor")
        if self.positions.dim() != 2 or self.positions.size(1) != 3:
            raise ValueError("positions must be a 2D tensor with shape [num_atoms, 3]")
        if self.atomic_subsystem_indices.dim() != 1:
            raise ValueError("atomic_subsystem_indices must be a 1D tensor")
        if self.total_charge.dim() != 1:
            raise ValueError("total_charge must be a 1D tensor")

        # Optionally, check that the lengths match if required
        if len(self.positions) != len(self.atomic_numbers):
            raise ValueError(
                "The size of atomic_numbers and the first dimension of positions must match"
            )
        if len(self.positions) != len(self.atomic_subsystem_indices):
            raise ValueError(
                "The size of atomic_subsystem_indices and the first dimension of positions must match"
            )

    def as_namedtuple(self) -> NamedTuple:
        """Export the dataclass fields and values as a named tuple."""

        import collections
        from dataclasses import dataclass, fields

        NNPInputTuple = collections.namedtuple(
            "NNPInputTuple", [field.name for field in fields(self)]
        )
        return NNPInputTuple(*[getattr(self, field.name) for field in fields(self)])

    def as_jax_namedtuple(self) -> NamedTuple:
        """Export the dataclass fields and values as a named tuple.
        Convert pytorch tensors to jax arrays."""

        from dataclasses import dataclass, fields
        import collections
        from pytorch2jax.pytorch2jax import convert_to_jax

        NNPInputTuple = collections.namedtuple(
            "NNPInputTuple", [field.name for field in fields(self)]
        )
        return NNPInputTuple(
            *[convert_to_jax(getattr(self, field.name)) for field in fields(self)]
        )


@dataclass
class BatchData:
    nnp_input: NNPInput
    metadata: Metadata

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.nnp_input = self.nnp_input.to(device=device, dtype=dtype)
        self.metadata = self.metadata.to(device=device, dtype=dtype)
        return self


class TorchDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    """
    Wraps a numpy dataset to make it compatible with PyTorch DataLoader.

    Parameters
    ----------
    dataset : np.lib.npyio.NpzFile
        The underlying numpy dataset.
    property_name : PropertyNames
        Names of the properties to extract from the dataset.
    preloaded : bool, optional
        If True, converts properties to PyTorch tensors ahead of time. Default is False.

    """

    def __init__(
        self,
        dataset: np.lib.npyio.NpzFile,
        property_name: PropertyNames,
        preloaded: bool = False,
    ):
        """
        Initializes the TorchDataset with a numpy dataset and property names.

        Parameters
        ----------
        dataset : np.lib.npyio.NpzFile
            The numpy dataset to wrap.
        property_name : PropertyNames
            The property names to extract from the dataset for use in PyTorch.
        preloaded : bool, optional
            If set to True, properties are preloaded as PyTorch tensors. Default is False.
        """

        self.properties_of_interest = {}

        self.properties_of_interest["atomic_numbers"] = torch.from_numpy(
            dataset[property_name.Z].flatten()
        ).to(torch.int32)
        self.properties_of_interest["positions"] = torch.from_numpy(
            dataset[property_name.R]
        ).to(torch.float32)
        self.properties_of_interest["E"] = torch.from_numpy(
            dataset[property_name.E]
        ).to(torch.float64)

        if property_name.Q is not None:
            self.properties_of_interest["Q"] = torch.from_numpy(
                dataset[property_name.Q]
            ).to(torch.int32)
        else:
            # this is a per atom property, so it will match the first dimension of the geometry
            self.properties_of_interest["Q"] = torch.zeros(
                (dataset[property_name.R].shape[0], 1)
            ).to(torch.int32)

        if property_name.F is not None:
            self.properties_of_interest["F"] = torch.from_numpy(
                dataset[property_name.F]
            )
        else:
            # a per atom property in each direction, so it will match geometry
            self.properties_of_interest["F"] = torch.zeros(
                dataset[property_name.R].shape
            )

        self.number_of_records = len(dataset["atomic_subsystem_counts"])
        self.properties_of_interest["pair_list"] = None
        self.number_of_atoms = len(dataset["atomic_numbers"])

        single_atom_start_idxs_by_rec = np.concatenate(
            [np.array([0]), np.cumsum(dataset["atomic_subsystem_counts"])]
        )
        # length: n_records + 1

        self.series_mol_start_idxs_by_rec = np.concatenate(
            [np.array([0]), np.cumsum(dataset["n_confs"])]
        )
        # length: n_records + 1

        if len(single_atom_start_idxs_by_rec) != len(self.series_mol_start_idxs_by_rec):
            raise ValueError(
                "Number of records in `atomic_subsystem_counts` and `n_confs` do not match."
            )

        self.single_atom_start_idxs_by_conf = np.repeat(
            single_atom_start_idxs_by_rec[: self.number_of_records], dataset["n_confs"]
        )
        self.single_atom_end_idxs_by_conf = np.repeat(
            single_atom_start_idxs_by_rec[1 : self.number_of_records + 1],
            dataset["n_confs"],
        )
        # length: n_conformers

        self.series_atom_start_idxs_by_conf = np.concatenate(
            [
                np.array([0]),
                np.cumsum(
                    np.repeat(dataset["atomic_subsystem_counts"], dataset["n_confs"])
                ),
            ]
        )
        # length: n_conformers + 1

        self.length = len(self.properties_of_interest["E"])
        self.preloaded = preloaded

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

    def __getitem__(self, idx: int) -> BatchData:
        """
        Fetch a dictionary of the values for the properties of interest for a given conformer index.

        Parameters
        ----------
        idx : int
            Index of the molecule to fetch data for.

        Returns
        -------
        BatchData instance representing the data for one conformer.
        """
        series_atom_start_idx = self.series_atom_start_idxs_by_conf[idx]
        series_atom_end_idx = self.series_atom_start_idxs_by_conf[idx + 1]
        single_atom_start_idx = self.single_atom_start_idxs_by_conf[idx]
        single_atom_end_idx = self.single_atom_end_idxs_by_conf[idx]
        atomic_numbers = self.properties_of_interest["atomic_numbers"][
            single_atom_start_idx:single_atom_end_idx
        ]
        positions = self.properties_of_interest["positions"][
            series_atom_start_idx:series_atom_end_idx
        ]
        E = self.properties_of_interest["E"][idx]
        F = self.properties_of_interest["F"][series_atom_start_idx:series_atom_end_idx]
        total_charge = self.properties_of_interest["Q"][idx]
        number_of_atoms = len(atomic_numbers)
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

        nnp_input = NNPInput(
            atomic_numbers=atomic_numbers,
            positions=positions,
            pair_list=pair_list,
            total_charge=total_charge,
            atomic_subsystem_indices=torch.zeros(number_of_atoms, dtype=torch.int32),
        )

        metadata = Metadata(
            E=E,
            F=F,
            atomic_subsystem_counts=torch.tensor([number_of_atoms], dtype=torch.int32),
            atomic_subsystem_indices_referencing_dataset=torch.repeat_interleave(
                torch.tensor([idx], dtype=torch.int32), number_of_atoms
            ),
            number_of_atoms=number_of_atoms,
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
            with open(
                f"{self.local_cache_dir}/{self.hdf5_data_file['name']}", "wb"
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

            with open(f"{file_path}/{file_name}", "r") as f:
                self._npz_metadata = json.load(f)

                if not self._check_lists(
                    self._npz_metadata["data_keys"], self.properties_of_interest
                ):
                    log.warning(
                        f"Data keys used to generate {file_path}/{file_name} ({self._npz_metadata['data_keys']}) do not match data loader ({self.properties_of_interest}) ."
                    )
                    return False

                if self._npz_metadata["hdf5_checksum"] != self.hdf5_data_file["md5"]:
                    log.warning(
                        f"Checksum for hdf5 file used to generate npz file does not match current file in dataloader."
                    )
                    return False

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

        with h5py.File(temp_hdf5_file, "r") as hf:
            # create dicts to store data for each format type
            single_rec_data: Dict[str, List[np.ndarray]] = OrderedDict()
            # value shapes: (*)
            single_atom_data: Dict[str, List[np.ndarray]] = OrderedDict()
            # value shapes: (n_atoms, *)
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
                            raise ValueError(
                                f"Number of conformers is inconsistent across properties for record {record}"
                            )

                        configs_nan = np.logical_or.reduce(
                            list(configs_nan_by_prop.values())
                        )  # boolean array of size (n_configs, )
                        n_confs_rec = sum(~configs_nan)

                        atomic_subsystem_counts_rec = hf[record][
                            next(iter(single_atom_data.keys()))
                        ].shape[0]
                        # all single and series atom properties should have the same number of atoms as the first property

                        self.n_confs.append(n_confs_rec)
                        self.atomic_subsystem_counts.append(atomic_subsystem_counts_rec)

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
                                if record_array.shape[1] != atomic_subsystem_counts_rec:
                                    raise ValueError(
                                        f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                                    )
                                else:
                                    series_atom_data[value].append(
                                        record_array.reshape(
                                            n_confs_rec * atomic_subsystem_counts_rec,
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

            # convert lists of arrays to single arrays

            data = OrderedDict()
            for value in single_atom_data.keys():
                data[value] = np.concatenate(single_atom_data[value], axis=0)
            for value in series_mol_data.keys():
                data[value] = np.concatenate(series_mol_data[value], axis=0)
            for value in series_atom_data.keys():
                data[value] = np.concatenate(series_atom_data[value], axis=0)
            for value in single_rec_data.keys():
                data[value] = np.stack(single_rec_data[value], axis=0)

        self.hdf5data = data

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
                self.numpy_data = np.load(
                    f"{self.local_cache_dir}/{self.processed_data_file['name']}"
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

        np.savez(
            f"{self.local_cache_dir}/{self.processed_data_file['name']}",
            atomic_subsystem_counts=self.atomic_subsystem_counts,
            n_confs=self.n_confs,
            **self.hdf5data,
        )
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

        with open(
            f"{self.local_cache_dir}/{self.processed_data_file['name'].replace('.npz', '.json')}",
            "w",
        ) as f:
            json.dump(metadata, f)

        del self.hdf5data


class DatasetFactory:
    """
    Factory for creating TorchDataset instances from HDF5 data.

    Methods are provided to load or process data as needed, handling caching to improve efficiency.

    Examples
    --------
    >>> factory = DatasetFactory()
    >>> qm9_data = QM9Data()
    >>> torch_dataset = factory.create_dataset(qm9_data)
    """

    def __init__(
        self,
    ) -> None:
        pass

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

        log.info(f"Creating {data.dataset_name} dataset")
        DatasetFactory._load_or_process_data(data)
        return TorchDataset(data.numpy_data, data._property_names)


from torch import nn
from openff.units import unit


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        name: Literal[
            "QM9",
            "ANI1X",
            "ANI2X",
            "SPICE114",
            "SPICE2",
            "SPICE114_OPENFF",
            "PhAlkEthOH",
        ],
        splitting_strategy: SplittingStrategy = RandomRecordSplittingStrategy(),
        batch_size: int = 64,
        remove_self_energies: bool = True,
        atomic_self_energies: Optional[Dict[str, float]] = None,
        regression_ase: bool = False,
        force_download: bool = False,
        version_select: str = "latest",
        local_cache_dir: str = "./",
        regenerate_cache: bool = False,
    ):
        """
        Initializes adData module for PyTorch Lightning handling data preparation and loading object with the specified configuration.
        If `remove_self_energies` is `True` and:
        - `self_energies` are passed as a dictionary, these will be used
        - `self_energies` are `None`, `self._ase` will be used
        - `regression_ase` is `True`, self_energies will be calculated

        Parameters
        ---------
            name: Literal["QM9", "ANI1X", "ANI2X", "SPICE114", "SPICE2", "SPICE114_OPENFF"]
                The name of the dataset to use.
            splitting_strategy : SplittingStrategy, defaults to RandomRecordSplittingStrategy
                The strategy to use for splitting the dataset into train, test, and validation sets. .
            batch_size : int, defaults to 64.
                The batch size to use for the dataset.
            remove_self_energies : bool, defaults to True
                Whether to remove the self energies from the dataset.
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
        super().__init__()

        self.name = name
        self.batch_size = batch_size
        self.splitting_strategy = splitting_strategy
        self.dataset_statistics: Optional[DatasetStatistics] = None
        self.remove_self_energies = remove_self_energies
        self.dict_atomic_self_energies = (
            atomic_self_energies  # element name (e.g., 'H') maps to self energies
        )
        self.regression_ase = regression_ase
        self.force_download = force_download
        self.version_select = version_select
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        import os

        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.regenerate_cache = regenerate_cache
        from modelforge.potential.models import Pairlist

        self.pairlist = Pairlist()
        self.dataset_statistics_filename = (
            f"{self.local_cache_dir}/{self.name}_dataset_statistics.toml"
        )

    def prepare_data(
        self,
    ) -> None:
        """
        Prepares the dataset for use. This method is responsible for the initial processing of the data such as calculating self energies, atomic energy statistics, and splitting. It is executed only once per node.
        """
        from modelforge.dataset import _ImplementedDatasets
        import toml

        dataset_class = _ImplementedDatasets.get_dataset_class(self.name)
        dataset = dataset_class(
            force_download=self.force_download,
            version_select=self.version_select,
            local_cache_dir=self.local_cache_dir,
            regenerate_cache=self.regenerate_cache,
        )
        torch_dataset = self._create_torch_dataset(dataset)

        # if dataset statistics is present load it from disk
        if os.path.exists(self.dataset_statistics_filename):
            atomic_self_energies = self._read_atomic_self_energies()
            atomic_energies_stats = self._read_atomic_energies_stats()
        else:
            atomic_self_energies = None
            atomic_energies_stats = None
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
        if atomic_energies_stats is None:
            from modelforge.dataset.utils import calculate_mean_and_variance

            atomic_energies_stats = calculate_mean_and_variance(torch_dataset)
            # wrap everything in a dictionary and save it to disk
            dataset_statistics = {
                "atomic_self_energies": atomic_self_energies,
                "atomic_energies_stats": atomic_energies_stats,
            }

            if atomic_self_energies and atomic_energies_stats:
                print(dataset_statistics)
                # save dataset_statistics dictionary to disk as yaml files
                self._log_dataset_statistics(dataset_statistics)
            else:
                raise RuntimeError(
                    "Atomic self energies or atomic energies statistics are missing."
                )

        # Save processed dataset and statistics for later use in setup
        self._cache_dataset(torch_dataset)

    def _log_dataset_statistics(self, dataset_statistics):
        """Save the dataset statistics to a file."""
        import toml

        # remove units from the statistics
        atomic_self_energies = {
            key: value.magnitude if isinstance(value, unit.Quantity) else value
            for key, value in dataset_statistics["atomic_self_energies"].items()
        }
        # cast float on pytorch tensors
        atomic_energies_stats = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in dataset_statistics["atomic_energies_stats"].items()
        }

        dataset_statistics = {
            "atomic_self_energies": atomic_self_energies,
            "atomic_energies_stats": atomic_energies_stats,
        }
        toml.dump(
            dataset_statistics,
            open(
                self.dataset_statistics_filename,
                "w",
            ),
        )

    def _read_atomic_self_energies(self) -> Dict[str, Quantity]:
        """Read the atomic self energies from a file."""
        import toml

        unitless_energy_statistic = toml.load(
            open(self.dataset_statistics_filename, "r")
        )

        # attach kJ/mol units
        atomic_self_energies = {
            key: float(value) * unit.kilojoule_per_mole
            for key, value in unitless_energy_statistic["atomic_self_energies"].items()
        }

        return atomic_self_energies

    def _read_atomic_energies_stats(self) -> Dict[str, torch.Tensor]:
        """Read the atomic energies statistics from a file."""
        import toml

        unitless_energy_statistic = toml.load(
            open(self.dataset_statistics_filename, "r")
        )
        # convert values to tensor
        atomic_energies_stats = {
            key: torch.tensor(value) for key, value in unitless_energy_statistic['atomic_energies_stats'].items()
        }
        return atomic_energies_stats

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

        from modelforge.dataset.utils import calculate_mean_and_variance

        # Use provided ase dictionary
        if self.dict_atomic_self_energies:
            log.info(
                "Using provided atomic self energies from the provided dictionary."
            )
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
        torch.save(torch_dataset, "torch_dataset.pt")
        # sleep for 1 second to make sure that the dataset was written to disk
        import time

        time.sleep(1)

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up datasets for the train, validation, and test stages based on the stage argument."""

        self.torch_dataset = torch.load("torch_dataset.pt")
        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.splitting_strategy.split(self.torch_dataset)
        )

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
        log.info("Precalculating pairlist for dataset")
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

        from torch.utils.data import DataLoader

        all_pairs = []
        n_pairs_per_molecule_list = [torch.tensor([0], dtype=torch.int16)]

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
            pairs_batch, n_pairs_batch = (
                self.pairlist.construct_initial_pairlist_using_numpy(
                    batch.nnp_input.atomic_subsystem_indices.to("cpu")
                )
            )
            all_pairs.append(torch.from_numpy(pairs_batch))
            n_pairs_per_molecule_list.append(torch.from_numpy(n_pairs_batch))

        nr_of_pairs = torch.cat(n_pairs_per_molecule_list, dim=0)
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

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the test dataset.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, collate_fn=collate_conformers
        )


from typing import Tuple


def collate_conformers(conf_list: List[BatchData]) -> BatchData:
    """Collate a list of BatchData instances with one conformer each into a single BatchData instance."""
    atomic_numbers_list = []
    positions_list = []
    total_charge_list = []
    E_list = []  # total energy
    F_list = []  # forces
    ij_list = []
    atomic_subsystem_counts_list = []
    atomic_subsystem_indices_referencing_dataset_list = []

    offset = torch.tensor([0], dtype=torch.int32)
    pair_list_present = (
        True
        if hasattr(conf_list[0].nnp_input, "pair_list")
        and isinstance(conf_list[0].nnp_input.pair_list, torch.Tensor)
        else False
    )

    for idx, conf in enumerate(conf_list):
        if pair_list_present:
            ## pairlist
            # generate pairlist without padded values
            pair_list = conf.nnp_input.pair_list.to(dtype=torch.int32) + offset
            # update offset (for making sure the pair_list indices are pointing to the correct molecule)
            offset += conf.nnp_input.atomic_numbers.shape[0]
            ij_list.append(pair_list)

        atomic_numbers_list.append(conf.nnp_input.atomic_numbers)
        positions_list.append(conf.nnp_input.positions)
        total_charge_list.append(conf.nnp_input.total_charge)
        E_list.append(conf.metadata.E)
        F_list.append(conf.metadata.F)
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
    total_charge = torch.cat(total_charge_list)
    positions = torch.cat(positions_list).requires_grad_(True)
    F = torch.cat(F_list).to(torch.float64)
    E = torch.stack(E_list)
    if pair_list_present:
        IJ_cat = torch.cat(ij_list, dim=1).to(torch.int64)
    else:
        IJ_cat = None

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        total_charge=total_charge,
        atomic_subsystem_indices=atomic_subsystem_indices,
        pair_list=IJ_cat,
    )
    metadata = Metadata(
        E=E,
        F=F,
        atomic_subsystem_counts=atomic_subsystem_counts,
        atomic_subsystem_indices_referencing_dataset=atomic_subsystem_indices_referencing_dataset,
        number_of_atoms=atomic_numbers.numel(),
    )
    return BatchData(nnp_input, metadata)

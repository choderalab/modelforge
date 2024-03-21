import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as log
from torch.utils.data import DataLoader

from modelforge.utils.prop import PropertyNames

from modelforge.dataset.utils import RandomRecordSplittingStrategy, SplittingStrategy
from dataclasses import dataclass

if TYPE_CHECKING:
    from modelforge.potential import BatchData


@dataclass
class DatasetStatistics:
    scaling_mean: float
    scaling_stddev: float
    atomic_self_energies: Dict[str, float]


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

    # TODO: add support for general properties with given formats

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

        self.properties_of_interest = {
            "atomic_numbers": torch.from_numpy(dataset[property_name.Z]),
            "positions": torch.from_numpy(dataset[property_name.R]),
            "E": torch.from_numpy(dataset[property_name.E]),
            "Q": torch.from_numpy(dataset[property_name.Q]),
        }

        self.number_of_records = len(dataset["atomic_subsystem_counts"])
        self.number_of_atoms = len(dataset["atomic_numbers"])
        single_atom_start_idxs_by_rec = np.concatenate(
            [[0], np.cumsum(dataset["atomic_subsystem_counts"])]
        )
        # length: n_records + 1

        self.series_mol_start_idxs_by_rec = np.concatenate(
            [[0], np.cumsum(dataset["n_confs"])]
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
                [0],
                np.cumsum(
                    np.repeat(dataset["atomic_subsystem_counts"], dataset["n_confs"])
                ),
            ]
        )
        # length: n_conformers + 1

        self.length = len(self.single_atom_start_idxs_by_conf)
        self.preloaded = preloaded

    def __len__(self) -> int:
        """
        Return the number of conformers in the dataset.

        Returns:
        --------
        int
            Total number of conformers available in the dataset.
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetch a dictionary of the values for the properties of interest for a given conformer index.

        Parameters
        ----------
        idx : int
            Index of the molecule to fetch data for.

        Returns
        -------
        Dictionary, contains:
            - atomic_numbers: torch.Tensor, shape [n_atoms]
                Atomic numbers for each atom in the molecule.
            - positions: torch.Tensor, shape [n_atoms, 3]
                Coordinates for each atom in the molecule.
            - E: torch.Tensor, shape []
                Scalar energy value for the molecule.
            - idx: int
                Index of the conformer in the dataset.
            - atomic_subsystem_counts: torch.Tensor, shape [1]
                Number of atoms in the conformer. Length one if __getitem__ is called with a single index, length batch_size if collate_conformers is used with DataLoader
        """
        from modelforge.potential.utils import ATOMIC_NUMBER_TO_INDEX_MAP

        series_atom_start_idx = self.series_atom_start_idxs_by_conf[idx]
        series_atom_end_idx = self.series_atom_start_idxs_by_conf[idx + 1]
        single_atom_start_idx = self.single_atom_start_idxs_by_conf[idx]
        single_atom_end_idx = self.single_atom_end_idxs_by_conf[idx]
        atomic_numbers = (
            self.properties_of_interest["atomic_numbers"][
                single_atom_start_idx:single_atom_end_idx
            ]
            .clone()
            .detach()
            .squeeze(1)
        ).to(torch.int64)
        positions = (
            self.properties_of_interest["positions"][
                series_atom_start_idx:series_atom_end_idx
            ]
            .clone()
            .detach()
        ).to(torch.float32)
        E = (
            (self.properties_of_interest["E"][idx]).clone().detach().to(torch.float64)
        )  # NOTE: upgrading to float64 to avoid precision issues
        Q = (self.properties_of_interest["Q"][idx]).clone().detach().to(torch.int32)

        return {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "total_charge": Q,
            "E": E,
            "atomic_subsystem_counts": torch.tensor([atomic_numbers.shape[0]]),
            "idx": idx,
            "atomic_index": torch.tensor(
                [
                    ATOMIC_NUMBER_TO_INDEX_MAP.get(atomic_number, -1)
                    for atomic_number in list(atomic_numbers.numpy())
                ]
            ),
        }


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
        self, raw_data_file: str, processed_data_file: str, local_cache_dir: str
    ):
        """
        Initializes the HDF5Dataset with paths to raw and processed data files.

        Parameters
        ----------
        raw_data_file : str
            Path to the raw HDF5 data file.
        processed_data_file : str
            Path to the processed data file.
        local_cache_dir : str
            Directory to store temporary processing files.
        """

        self.raw_data_file = raw_data_file
        self.processed_data_file = processed_data_file
        self.hdf5data: Optional[Dict[str, List[np.ndarray]]] = None
        self.numpy_data: Optional[np.ndarray] = None
        self.local_cache_dir = local_cache_dir

    def _from_hdf5(self) -> None:
        """
        Processes and extracts data from an hdf5 file.

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> processed_data = hdf5_data._from_hdf5()

        """
        import gzip
        from collections import OrderedDict
        import h5py
        import tqdm
        import shutil

        log.debug(f"Processing and extracting data from {self.raw_data_file}")

        # this will create an unzipped file which we can then load in
        # this is substantially faster than passing gz_file directly to h5py.File()
        # by avoiding data chunking issues.

        temp_hdf5_file = f"{self.local_cache_dir}/temp_unzipped.hdf5"
        with gzip.open(self.raw_data_file, "rb") as gz_file:
            with open(temp_hdf5_file, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)

        log.debug("Reading in and processing hdf5 file ...")

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

            # intialize each relevant value in data dicts to empty list
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
                # There may be cases where a specific property of interest
                # has not been computed for a given record
                # in that case, we'll want to just skip over that entry
                property_found = [
                    value in hf[record].keys() for value in self.properties_of_interest
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
                            set([value.shape for value in configs_nan_by_prop.values()])
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
                        if record_array.shape[1] != atomic_subsystem_counts_rec:
                            raise ValueError(
                                f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                            )
                        else:
                            series_atom_data[value].append(
                                record_array.reshape(
                                    n_confs_rec * atomic_subsystem_counts_rec, -1
                                )
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
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> processed_data = hdf5_data._from_file_cache()
        """
        log.debug(f"Loading processed data from {self.processed_data_file}")
        self.numpy_data = np.load(self.processed_data_file)

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
        log.debug(f"Writing data cache to {self.processed_data_file}")

        np.savez(
            self.processed_data_file,
            atomic_subsystem_counts=self.atomic_subsystem_counts,
            n_confs=self.n_confs,
            **self.hdf5data,
        )
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

        # if not cached, download and process
        if not os.path.exists(data.processed_data_file):
            if not os.path.exists(data.raw_data_file):
                data._download()
            # load from hdf5 and process
            data._from_hdf5()
            # save to cache
            data._to_file_cache()
        # load from cache
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


class TorchDataModule(pl.LightningDataModule):
    """
    Data module for PyTorch Lightning, handling data preparation and loading.

    Parameters
    ----------
    data : HDF5Dataset
        The dataset to use.
    split : SplittingStrategy, optional
        The strategy for splitting data into training,
        validation, and test sets. Defaults to RandomRecordSplittingStrategy.
    batch_size : int,

    Examples
    --------
    >>> data = QM9Dataset()
    >>> data_module = TorchDataModule(data)
    >>> data_module.prepare_data()
    >>> train_loader = data_module.train_dataloader()
    """

    def __init__(
        self,
        data: HDF5Dataset,
        split: SplittingStrategy = RandomRecordSplittingStrategy(),
        batch_size: int = 64,
        split_file: Optional[str] = None,
        transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.split_idxs = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.transform = transform
        self.split = split
        self.split_file = split_file
        self.dataset_statistics: DatasetStatistics = None
        self._ase = data.atomic_self_energies  # atomic self energies

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
        from modelforge.dataset.utils import calculate_self_energies

        # Define the collate function based on the `collate` parameter
        collate_fn = collate_conformers if collate else None
        return calculate_self_energies(
            torch_dataset=torch_dataset, collate_fn=collate_fn
        )

    def prepare_data(
        self,
        remove_self_energies: bool = True,
        normalize: bool = True,
        self_energies: Dict[str, float] = None,
        regression_ase: bool = False,
    ) -> None:
        """
        Prepares the dataset for use by calculating self energies, normalizing data, and splitting the dataset.

        Parameters
        ----------
        remove_self_energies : bool, optional
            Whether to remove self energies from the dataset. Defaults to True.
        normalize : bool, optional
            Whether to normalize the dataset. Defaults to True.
        self_energies : Dict[str, float], optional
            A dictionary mapping atomic numbers to their calculated self energies. Defaults to None.
        regression_ase : bool, optional
            If regression_ase is True, atomic self energies are calculated using linear regression

        if remove_self_energies is True and
            - self_energies are passed as dictionary, these will be used
            - self_energies are None, self._ase will be used
            - regression_ase is True, self_energies will be calculated
        """
        if self.split_file and os.path.exists(self.split_file):
            self.split_idxs = np.load(self.split_file, allow_pickle=True)
            log.debug(f"Loaded split indices from {self.split_file}")
        else:
            log.debug("Splitting strategy will be applied")

        # generate dataset
        factory = DatasetFactory()
        torch_dataset = factory.create_dataset(self.data)
        dataset_statistics = {
            "scaling_stddev": 1,
            "scaling_mean": 0,
            "atomic_self_energies": {},
        }

        if remove_self_energies:
            # calculate self energies, and then remove them from the dataset

            log.debug("Self energies are removed ...")
            if not self_energies:
                if regression_ase:
                    log.debug(
                        "Using linear regression to calculate atomic self energies..."
                    )
                    self_energies = self.calculate_self_energies(
                        torch_dataset=torch_dataset
                    )
                else:
                    log.debug("Using atomic self energies from the dataset...")
                    self_energies = self._ase

            # remove self energies
            self.subtract_self_energies(torch_dataset, self_energies)
            # write the self energies that are removed from the dataset to disk
            import toml

            with open("ase.toml", "w") as f:
                self_energies_ = {str(idx): energy for (idx, energy) in self._ase}
                toml.dump(self_energies_, f)

            # store self energies
            dataset_statistics["atomic_self_energies"] = self_energies

        if normalize:
            # calculate average and variance
            if not remove_self_energies:
                raise RuntimeError(
                    "Cannot normalize the dataset if self energies are not removed."
                )

            log.debug("Normalizing energies...")
            from modelforge.dataset.utils import (
                calculate_mean_and_variance,
                normalize_energies,
            )

            stats = calculate_mean_and_variance(torch_dataset)
            normalize_energies(torch_dataset, stats)
            dataset_statistics["scaling_stddev"] = stats["stddev"]
            dataset_statistics["scaling_mean"] = stats["mean"]

        self.dataset_statistics = DatasetStatistics(**dataset_statistics)
        self.setup(torch_dataset)

    def subtract_self_energies(self, dataset, self_energies: Dict[str, float]) -> None:
        """
        Removes the self energies from the total energies for each molecule in the dataset .

        Parameters
        ----------
        dataset: torch.Dataset
            The dataset from which to remove the self energies.
        self_energies : Dict[str, float]
            Dictionary containing the self energies for each element in the dataset.
        """
        from tqdm import tqdm

        log.info("Removing self energies from the dataset")
        for i in tqdm(range(len(dataset)), desc="Removing Self Energies"):
            atomic_numbers = list(dataset[i]["atomic_numbers"])
            E = dataset[i]["E"]
            for Z in atomic_numbers:
                E -= self_energies[int(Z)]
            dataset[i] = {"E": E}

        return dataset

    @classmethod
    def preprocess_dataset(
        cls,
        dataset: TorchDataset,
        normalize: bool,
        dataset_mean: float,
        dataset_std: Optional[float] = None,
    ) -> None:
        from tqdm import tqdm

        log.info(f"Adjusting energies by subtracting global mean {dataset_mean}")

        if normalize:
            log.info("Normalizing energies using computed mean and std")
        for i in tqdm(range(len(dataset)), desc="Adjusting Energies"):
            energy = dataset[i]["E"]
            if normalize:
                # Normalize using the computed mean and std
                modified_energy = (energy - dataset_mean) / dataset_std
            else:
                # Only adjust by subtracting the mean
                modified_energy = energy - dataset_mean
            dataset[i] = {"E": modified_energy}

        return dataset

    def setup(self, torch_dataset) -> None:
        """
        Sets up datasets for the train, validation, and test stages.

        """
        from torch.utils.data import Subset

        # saving the raw dataset
        self.torch_dataset = torch_dataset  # FIXME: is this necessary?
        # Create subsets for training, validation, and testing
        if self.split_idxs and os.path.exists(self.split_file):
            self.train_dataset = Subset(torch_dataset, self.split_idxs["train_idx"])
            self.val_dataset = Subset(torch_dataset, self.split_idxs["val_idx"])
            self.test_dataset = Subset(torch_dataset, self.split_idxs["test_idx"])
        else:
            # Fallback to manual splitting if split_idxs is not available
            self.train_dataset, self.val_dataset, self.test_dataset = self.split.split(
                torch_dataset
            )

    def train_dataloader(self) -> DataLoader:
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
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
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
            num_workers=4,
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


def collate_conformers(conf_list: List[Dict[str, torch.Tensor]]) -> "BatchData":
    # TODO: once TorchDataset is reimplemented for general properties, reimplement this function using formats too.
    """Concatenate the Z, R, and E tensors from a list of molecules into a single tensor each, and return a new dictionary with the concatenated tensors."""
    from modelforge.potential.utils import NNPInput, BatchData, Metadata

    Z_list = []  # nuclear charges/atomic numbers
    R_list = []  # positions
    E_list = []  # total energy
    Q_list = []  # total charge
    atomic_subsystem_counts = []
    atomic_subsystem_indices = []
    atomic_subsystem_indices_referencing_dataset = []
    for idx, conf in enumerate(conf_list):
        Z_list.append(conf["atomic_numbers"])
        R_list.append(conf["positions"])
        E_list.append(conf["E"])
        Q_list.append(conf["total_charge"])
        atomic_subsystem_counts.extend(conf["atomic_subsystem_counts"])
        atomic_subsystem_indices.extend([idx] * conf["atomic_subsystem_counts"][0])
        atomic_subsystem_indices_referencing_dataset.extend(
            [conf["idx"]] * conf["atomic_subsystem_counts"][0]
        )
    atomic_numbers_cat = torch.cat(Z_list)
    total_charge_cat = torch.cat(Q_list)
    positions_cat = torch.cat(R_list).requires_grad_(True)
    E_stack = torch.stack(E_list)
    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers_cat,
        positions=positions_cat,
        total_charge=total_charge_cat,
        atomic_subsystem_indices=torch.tensor(
            atomic_subsystem_indices, dtype=torch.int32
        ),
    )
    metadata = Metadata(
        E=E_stack,
        atomic_subsystem_counts=torch.tensor(
            atomic_subsystem_counts, dtype=torch.int32
        ),
        atomic_subsystem_indices_referencing_dataset=torch.tensor(
            atomic_subsystem_indices_referencing_dataset, dtype=torch.int32
        ),
        number_of_atoms=atomic_numbers_cat.numel(),
    )
    return BatchData(nnp_input, metadata)

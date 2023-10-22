import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from modelforge.utils.prop import PropertyNames

from modelforge.dataset.utils import RandomSplittingStrategy, SplittingStrategy


class TorchDataset(torch.utils.data.Dataset):
    """
    A custom dataset class to wrap numpy datasets for PyTorch.

    Parameters
    ----------
    dataset : np.lib.npyio.NpzFile
        The underlying numpy dataset.
    property_name : PropertyNames
        Property names to extract from the dataset.
    preloaded : bool, optional
        If True, preconverts the properties to PyTorch tensors to save time during item fetching.
        Default is False.

    """

    # TODO: add support for general properties with given formats

    def __init__(
            self,
            dataset: np.lib.npyio.NpzFile,
            property_name: PropertyNames,
            preloaded: bool = False,
    ):
        self.properties_of_interest = {
            "Z": dataset[property_name.Z],
            "R": dataset[property_name.R],
            "E": dataset[property_name.E],
        }

        n_records = len(dataset["n_atoms"])
        single_atom_start_idxs_by_rec = np.concatenate([[0], np.cumsum(dataset["n_atoms"])])
        # length: n_records + 1

        self.single_atom_start_idxs = np.repeat(
                single_atom_start_idxs_by_rec[:n_records], dataset["n_confs"]
        )
        self.single_atom_end_idxs = np.repeat(
                single_atom_start_idxs_by_rec[1:n_records + 1], dataset["n_confs"]
        )
        # length: n_conformers

        self.series_atom_start_idxs = np.concatenate(
            [[0], np.cumsum(np.repeat(dataset["n_atoms"], dataset["n_confs"]))]
        )
        # length: n_conformers + 1

        self.length = len(self.single_atom_start_idxs)
        self.preloaded = preloaded

    def __len__(self) -> int:
        """
        Return the number of datapoints in the dataset.

        Returns:
        --------
        int
            Total number of datapoints available in the dataset.
        """
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetch a tuple of the values for the properties of interest for a given conformer index.

        Parameters
        ----------
        idx : int
            Index of the molecule to fetch data for.

        Returns
        -------
        dict, contains:
            - 'Z': torch.Tensor, shape [n_atoms]
                Atomic numbers for each atom in the conformer.
            - 'R': torch.Tensor, shape [n_atoms, 3]
                Coordinates for each atom in the conformer.
            - 'E': torch.Tensor, shape [[1, 1]]
                Scalar energy value for the conformer.
            - 'n_atoms': List[int]
                Number of atoms in the conformer. Length one if __getitem__ is called with a single index, length batch_size if collate_conformers is used with DataLoader
        """
        series_atom_start_idx = self.series_atom_start_idxs[idx]
        series_atom_end_idx = self.series_atom_start_idxs[idx + 1]
        single_atom_start_idx = self.single_atom_start_idxs[idx]
        single_atom_end_idx = self.single_atom_end_idxs[idx]
        Z = torch.tensor(self.properties_of_interest["Z"][single_atom_start_idx:single_atom_end_idx], dtype=torch.int64)
        R = torch.tensor(self.properties_of_interest["R"][series_atom_start_idx:series_atom_end_idx],
                         dtype=torch.float32)
        E = torch.tensor(self.properties_of_interest["E"][idx], dtype=torch.float32)

        return {"Z": Z, "R": R, "E": E, "n_atoms": [Z.shape[0]]}


class HDF5Dataset:
    """
    Base class for data stored in HDF5 format.

    Provides methods for processing and interacting with the data stored in HDF5 format.

    Attributes
    ----------
    raw_data_file : str
        Path to the raw data file.
    processed_data_file : str
        Path to the processed data file.
    """

    def __init__(
            self, raw_data_file: str, processed_data_file: str, local_cache_dir: str
    ):
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

        logger.debug(f"Processing and extracting data from {self.raw_data_file}")

        # this will create an unzipped file which we can then load in
        # this is substantially faster than passing gz_file directly to h5py.File()
        # by avoiding data chunking issues.

        temp_hdf5_file = f"{self.local_cache_dir}/temp_unzipped.hdf5"
        with gzip.open(self.raw_data_file, "rb") as gz_file:
            with open(temp_hdf5_file, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)

        logger.debug("Reading in and processing hdf5 file ...")

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

            self.n_atoms = []  # number of atoms in each record
            self.n_confs = []  # number of conformers in each record

            # loop over all records in the hdf5 file and add property arrays to the appropriate dict

            logger.debug(f"n_entries: {len(hf.keys())}")

            for record in tqdm.tqdm(list(hf.keys())):

                # There may be cases where a specific property of interest
                # has not been computed for a given record
                # in that case, we'll want to just skip over that entry
                property_found = [value in hf[record].keys() for value in self.properties_of_interest]

                if all(property_found):

                    # we want to exclude conformers with NaN values for any property of interest
                    configs_nan_by_prop: Dict[str, np.ndarray] = OrderedDict()  # ndarray.size (n_configs, )
                    for value in list(series_mol_data.keys()) + list(series_atom_data.keys()):
                        record_array = hf[record][value][()]
                        configs_nan_by_prop[value] = np.isnan(record_array).any(axis=tuple(range(1, record_array.ndim)))
                    # check that all values have the same number of conformers

                    if len(set([value.shape for value in configs_nan_by_prop.values()])) != 1:
                        raise ValueError(
                            f"Number of conformers is inconsistent across properties for record {record}"
                        )

                    configs_nan = np.logical_or.reduce(
                        list(configs_nan_by_prop.values()))  # boolean array of size (n_configs, )
                    n_confs_rec = sum(~configs_nan)

                    n_atoms_rec = hf[record][next(iter(single_atom_data.keys()))].shape[0]
                    # all single and series atom properties should have the same number of atoms as the first property

                    self.n_confs.append(n_confs_rec)
                    self.n_atoms.append(n_atoms_rec)

                    for value in single_atom_data.keys():
                        record_array = hf[record][value][()]
                        if record_array.shape[0] != n_atoms_rec:
                            raise ValueError(
                                f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                            )
                        else:
                            single_atom_data[value].append(record_array)

                    for value in series_atom_data.keys():
                        record_array = hf[record][value][()][~configs_nan]
                        if record_array.shape[1] != n_atoms_rec:
                            raise ValueError(
                                f"Number of atoms for property {value} is inconsistent with other properties for record {record}"
                            )
                        else:
                            series_atom_data[value].append(record_array.reshape(n_confs_rec * n_atoms_rec, -1))

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
        logger.debug(f"Loading processed data from {self.processed_data_file}")
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
        logger.debug(f"Writing data cache to {self.processed_data_file}")

        np.savez(
            self.processed_data_file,
            n_atoms=self.n_atoms,
            n_confs=self.n_confs,
            **self.hdf5data
        )
        del self.hdf5data


class DatasetFactory:
    """
    Factory class for creating Dataset instances.

    Provides utilities for processing and caching data.

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
            label_transform: Optional[Dict[str, Callable]],
            transform: Optional[Dict[str, Callable]],
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
            label_transform: Optional[Dict[str, Callable]] = None,
            transform: Optional[Dict[str, Callable]] = None,
    ) -> TorchDataset:
        """
        Creates a Dataset instance given an HDF5Dataset.

        Parameters
        ----------
        data : HDF5Dataset
            The HDF5 data to use.
        transform : Optional[Dict[str, Callable]], optional
            Transformation to apply to the data based on the property name, by default None
        label_transform : Optional[Dict[str, Callable]], optional
            Transformation to apply to the labels based on the property name, by default default_transformation
        Returns
        -------
        TorchDataset
            Dataset instance wrapped for PyTorch.
        """

        logger.info(f"Creating {data.dataset_name} dataset")
        DatasetFactory._load_or_process_data(data, label_transform, transform)
        return TorchDataset(data.numpy_data, data._property_names)


class TorchDataModule(pl.LightningDataModule):
    """
    A custom data module class to handle data loading and preparation for PyTorch Lightning training.

    Parameters
    ----------
    data : HDF5Dataset
        The underlying dataset.
    SplittingStrategy : SplittingStrategy, optional
        Strategy used to split the data into training, validation, and test sets.
        Default is RandomSplittingStrategy.
    batch_size : int, optional
        Batch size for data loading. Default is 64.

    Examples
    --------
    >>> data = QM9Dataset()
    >>> data_module = TorchDataModule(data)
    >>> data_module.prepare_data()
    >>> data_module.setup("fit")
    >>> train_loader = data_module.train_dataloader()
    """

    def __init__(
            self,
            data: HDF5Dataset,
            split: SplittingStrategy = RandomSplittingStrategy(),
            batch_size: int = 64,
            split_file: Optional[str] = None,
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.split_idxs: Optional[str] = None
        if split_file:
            import numpy as np

            logger.debug(f"Loading split indices from {split_file}")
            self.split_idxs = np.load(split_file)
        else:
            logger.debug(f"Using splitting strategy {split}")
            self.split = split

    def prepare_data(self) -> None:
        """
        Prepares the data by creating a dataset instance.
        """

        factory = DatasetFactory()
        self.dataset = factory.create_dataset(self.data)

    def setup(self, stage: str) -> None:
        """
        Splits the data into training, validation, and test sets based on the stage.

        Parameters
        ----------
        stage : str
            Either "fit" for training/validation split or "test" for test split.
        """
        from torch.utils.data import Subset

        if stage == "fit":
            if self.split_idxs:
                train_idx, val_idx = (
                    self.split_idxs["train_idx"],
                    self.split_idxs["val_idx"],
                )
                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
            else:
                train_dataset, val_dataset, _ = self.split.split(self.dataset)
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            if self.split_idxs:
                test_idx = self.split_idxs["test_idx"]
                self.test_dataset = Subset(self.dataset, test_idx)
            else:
                _, _, test_dataset = self.split.split(self.dataset)
                self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_conformers)

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_conformers)

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_conformers)


def collate_conformers(conf_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # TODO: once TorchDataset is reimplemented for general properties, reimplement this function using formats too.
    """Concatenate the Z, R, and E tensors from a list of molecules into a single tensor each, and return a new dictionary with the concatenated tensors."""
    Z_list = []
    R_list = []
    E_list = []
    n_atoms = []
    for conf in conf_list:
        Z_list.append(conf['Z'])
        R_list.append(conf['R'])
        E_list.append(conf['E'])
        n_atoms.extend(conf['n_atoms'])
    Z_cat = torch.cat(Z_list)
    R_cat = torch.cat(R_list)
    E_cat = torch.cat(E_list)
    return {"Z": Z_cat, "R": R_cat, "E": E_cat, "n_atoms": n_atoms}

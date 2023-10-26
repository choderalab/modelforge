import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from modelforge.utils.prop import PropertyNames

from .transformation import default_transformation
from modelforge.dataset.utils import RandomSplittingStrategy, SplittingStrategy


class TorchDataset(torch.utils.data.Dataset):
    """
    A custom dataset class to wrap numpy datasets for PyTorch.

    Parameters
    ----------
    dataset : np.ndarray
        The underlying numpy dataset.
    property_name : PropertyNames
        Property names to extract from the dataset.
    preloaded : bool, optional
        If True, preconverts the properties to PyTorch tensors to save time during item fetching.
        Default is False.

    """

    def __init__(
        self,
        dataset: np.ndarray,
        property_name: PropertyNames,
        preloaded: bool = False,
    ):
        self.properties_of_interest = {
            "atomic_numbers": dataset[property_name.Z],
            "positions": dataset[property_name.R],
            "E_label": dataset[property_name.E],
        }

        self.length = len(self.properties_of_interest["atomic_numbers"])
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
        Fetch a tuple of the values for the properties of interest for a given molecule index.

        Parameters
        ----------
        idx : int
            Index of the molecule to fetch data for.

        Returns
        -------
        dict, contains:
            - 'atomic_numbers': torch.Tensor, shape [n_atoms]
                Atomic numbers for each atom in the molecule.
            - 'positions': torch.Tensor, shape [n_atoms, 3]
                Coordinates for each atom in the molecule.
            - 'E_label': torch.Tensor, shape []
                Scalar energy value for the molecule.
            - 'idx': int
                Index of the molecule in the dataset.
        """
        atomic_numbers = torch.tensor(
            self.properties_of_interest["atomic_numbers"][idx], dtype=torch.int64
        )
        positions = torch.tensor(
            self.properties_of_interest["positions"][idx], dtype=torch.float32
        )
        E_label = torch.tensor(
            self.properties_of_interest["E_label"][idx], dtype=torch.float32
        )
        return {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "E_label": E_label,
            "idx": idx,
        }


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
        self.hdf5data: Optional[Dict[str, List]] = None
        self.numpy_data: Optional[np.ndarray] = None
        self.local_cache_dir = local_cache_dir

    def _from_hdf5(self) -> None:
        """
        Processes and extracts data from an hdf5 file.

        Returns
        -------
        OrderedDict[str, List]
            Processed data from the hdf5 file.

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

        logger.debug("Reading in and processing hdf5 file ...")
        # initialize dict with empty lists
        data = OrderedDict()
        for value in self.properties_of_interest:
            data[value] = []

        # molecule_id will contain an integer that is unique to molecules
        # i.e., conformers of the same molecule will have the same id.
        data["molecule_id"] = []
        logger.debug(f"Processing and extracting data from {self.raw_data_file}")

        # this will create an unzipped file which we can then load in
        # this is substantially faster than passing gz_file directly to h5py.File()
        # by avoiding data chunking issues.

        temp_hdf5_file = f"{self.local_cache_dir}/temp_unzipped.hdf5"
        with gzip.open(self.raw_data_file, "rb") as gz_file:
            with open(temp_hdf5_file, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)

        with h5py.File(temp_hdf5_file, "r") as hf:
            logger.debug(f"n_entries: {len(hf.keys())}")
            molecule_id = 0
            for mol in tqdm.tqdm(list(hf.keys())):
                n_configs = hf[mol]["n_configs"][()]
                temp_data = {}
                is_series = {}
                is_atom_mol = {}

                # There may be cases where a specific property of interest
                # has not been computed for a given molecule
                # in that case, we'll want to just skip over that entry
                property_found = True
                property_keys = list(hf[mol].keys())
                for value in self.properties_of_interest:
                    if not value in property_keys:
                        property_found = False

                if property_found:
                    for value in self.properties_of_interest:
                        # First grab all the data of interest;
                        # indexing into a local np array is much faster
                        # than indexing into the array in the hdf5 file
                        temp_data[value] = hf[mol][value][()]
                        is_series[value] = hf[mol][value].attrs["format"].split("_")[0]
                        is_atom_mol[value] = (
                            hf[mol][value].attrs["format"].split("_")[1]
                        )

                    for n in range(n_configs):
                        not_nan = True
                        temp_data_cut = {}
                        for value in self.properties_of_interest:
                            if is_series[value] == "series":
                                # Note: this doesn't treat per atom or per molecules quantities differently
                                # Buy I've put the logic to differentiate here anyway as an example
                                if is_atom_mol[value] == "mol":
                                    temp_data_cut[value] = temp_data[value][n][0]
                                if is_atom_mol[value] == "atom":
                                    temp_data_cut[value] = temp_data[value][n]
                                if np.any(np.isnan(temp_data_cut[value])):
                                    not_nan = False
                                    break
                            else:
                                if value == "atomic_numbers":
                                    temp_data_cut[value] = temp_data[value].reshape(-1)
                                else:
                                    temp_data_cut[value] = temp_data[value]
                                if np.any(np.isnan(temp_data_cut[value])):
                                    not_nan = False
                                    break
                        if not_nan:
                            for value in self.properties_of_interest:
                                data[value].append(temp_data_cut[value])

                            # keep track of the name of the molecule and configuration number
                            # may be needed for splitting
                            data["molecule_id"].append(molecule_id)
                    molecule_id += 1

        self.hdf5data = data

    def _from_file_cache(self) -> Dict[str, List]:
        """
        Loads the processed data from cache.

        Returns
        -------
        OrderedDict[str, List]
            Processed data from the cache file.

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> processed_data = hdf5_data._from_file_cache()
        """
        logger.debug(f"Loading processed data from {self.processed_data_file}")
        self.numpy_data = np.load(self.processed_data_file)

    def _perform_transformations(
        self,
        label_transform: Optional[Dict[str, Callable]],
        transforms: Dict[str, Callable],
    ) -> None:
        for prop_key in self.hdf5data:
            if prop_key not in self.hdf5data:
                raise ValueError(f"Property {prop_key} not found in data")
            if transforms and prop_key in transforms:
                logger.debug(f"Transformation applied to : {prop_key}")
                self.hdf5data[prop_key] = transforms[prop_key](self.hdf5data[prop_key])
            elif label_transform and prop_key in label_transform:
                logger.debug(f"Transformation applied to : {prop_key}")
                self.hdf5data[prop_key] = transforms[prop_key](self.hdf5data[prop_key])
            else:
                logger.debug(f"NO Transformation applied to : {prop_key}")
                self.hdf5data[prop_key] = transforms["all"](self.hdf5data[prop_key])

    def _to_file_cache(
        self,
        label_transform: Optional[Dict[str, Callable]],
        transforms: Optional[Dict[str, Callable]],
    ) -> None:
        """
        Save processed data to a numpy (.npz) file.
        Parameters
        ----------
        data : OrderedDict[str, List[np.ndarray]]
            Dictionary containing processed data to be saved.
        processed_dataset_file : str
            Path to save the processed dataset.
        label_transform : Optional[Dict[str, Callable]], optional
            transformations to apply to the labels
        transforms : Dict[str, Callable], default=default_transformation
            transformations to apply to the data

        Examples
        --------
        >>> hdf5_data = HDF5Dataset("raw_data.hdf5", "processed_data.npz")
        >>> hdf5_data._to_file_cache()
        """
        logger.debug(f"Processing data ...")
        if transforms:
            logger.debug(f"Applying transforms to {transforms.keys()}...")
            self._perform_transformations(label_transform, transforms)

        logger.debug(f"Writing data cache to {self.processed_data_file}")

        np.savez(
            self.processed_data_file,
            **self.hdf5data,
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
        dataset : HDF5Dataset
            The HDF5 dataset instance to use.
        """

        # if not cached, download and process
        if not os.path.exists(data.processed_data_file):
            if not os.path.exists(data.raw_data_file):
                data._download()
            # load from hdf5 and process
            data._from_hdf5()
            # save to cache
            data._to_file_cache(label_transform, transform)
        # load from cache
        data._from_file_cache()

    @staticmethod
    def create_dataset(
        data: HDF5Dataset,
        label_transform: Optional[Dict[str, Callable]] = None,
        transform: Optional[Dict[str, Callable]] = default_transformation,
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
                _, _, test_dataset = self.SplittingStrategy().split(self.dataset)
                self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            DataLoader containing the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

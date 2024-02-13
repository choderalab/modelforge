import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger as log
from torch.utils.data import DataLoader

from modelforge.utils.prop import PropertyNames

from modelforge.dataset.utils import RandomRecordSplittingStrategy, SplittingStrategy


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
            "E_label": torch.from_numpy(dataset[property_name.E]),
        }

        self.n_records = len(dataset["atomic_subsystem_counts"])
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
            single_atom_start_idxs_by_rec[: self.n_records], dataset["n_confs"]
        )
        self.single_atom_end_idxs_by_conf = np.repeat(
            single_atom_start_idxs_by_rec[1 : self.n_records + 1], dataset["n_confs"]
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
        return self.n_records

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
        Fetch a tuple of the values for the properties of interest for a given conformer index.

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
                Index of the conformer in the dataset.
            - 'atomic_subsystem_counts': torch.Tensor, shape [1]
                Number of atoms in the conformer. Length one if __getitem__ is called with a single index, length batch_size if collate_conformers is used with DataLoader
        """
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
        ).to(torch.int64)
        positions = (
            self.properties_of_interest["positions"][
                series_atom_start_idx:series_atom_end_idx
            ]
            .clone()
            .detach()
        ).to(torch.float32)
        E_label = (
            (self.properties_of_interest["E_label"][idx])
            .clone()
            .detach()
            .to(torch.float64)
        )  # NOTE: upgrading to float64 to avoid precision issues

        return {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "E_label": E_label,
            "atomic_subsystem_counts": torch.tensor([atomic_numbers.shape[0]]),
            "idx": idx,
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
        Creates a TorchDataset from an HDF5Dataset, applying optional transformations.

        Parameters
        ----------
        data : HDF5Dataset
            The HDF5 dataset to convert.
        label_transform : Optional[Dict[str, Callable]], optional
            Transformations to apply to labels, keyed by property name.
        transform : Optional[Dict[str, Callable]], optional
            Transformations to apply to data, keyed by property name.

        Returns
        -------
        TorchDataset
            The resulting PyTorch-compatible dataset.
        """

        log.info(f"Creating {data.dataset_name} dataset")
        DatasetFactory._load_or_process_data(data, label_transform, transform)
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
        self.dataset_statistics = {}


    @classmethod
    def calculate_self_energies(
        cls, dataset: TorchDataset, collate: bool = True
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

        # Define the collate function based on the `collate` parameter
        collate_fn = collate_conformers if collate else None

        # Initialize variables to hold data for regression
        batch_size = 64
        # Determine the size of the counts tensor
        num_molecules = dataset.n_records
        # Determine up to which Z we detect elements
        max_atomic_number = 100
        # Initialize the counts tensor
        counts = torch.zeros(num_molecules, max_atomic_number + 1, dtype=torch.int64)
        # save energies in list
        energy_array = torch.zeros(dataset.n_records, dtype=torch.float64)
        # for filling in the element count matrix
        molecule_counter = 0
        # counter for saving energy values
        current_index = 0
        # save unique atomic numbers in list
        unique_atomic_numbers = set()

        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn):
            energies, atomic_numbers, molecules_id = (
                batch["E_label"].squeeze(),
                batch["atomic_numbers"].squeeze(-1),
                batch["atomic_subsystem_indices"].to(torch.int64),
            )

            # Update the energy array and unique atomic numbers set
            batch_size = energies.size(0)
            energy_array[current_index : current_index + batch_size] = (
                energies.squeeze()
            )
            current_index += batch_size
            unique_atomic_numbers |= set(atomic_numbers.tolist())
            atomic_numbers_ = atomic_numbers - 1

            # Count the occurrence of each atomic number in molecules
            for molecule_id in molecules_id.unique():
                mask = molecules_id == molecule_id
                counts[molecule_counter].scatter_add_(
                    0, atomic_numbers_[mask], torch.ones_like(atomic_numbers_[mask])
                )
                molecule_counter += 1

        # Prepare the data for lineare regression
        valid_elements_mask = counts.sum(dim=0) > 0
        filtered_counts = counts[:, valid_elements_mask]

        Xs = [
            filtered_counts[:, i].unsqueeze(1).detach().numpy()
            for i in range(filtered_counts.shape[1])
        ]

        A = np.hstack(Xs)
        y = energy_array.numpy()

        # Perform least squares regression
        least_squares_fit, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self_energies = {
            idx: energy for idx, energy in zip(unique_atomic_numbers, least_squares_fit)
        }

        log.info("Calculated self energies for elements.")

        log.info("Coefficients for each element:", self_energies)
        print(self_energies)
        return self_energies

    # print information about the unit system used in the dataset
    def print_unit_system(self) -> None:
        """
        Print information about the unit system used in the dataset.
        """
        from modelforge.utils import provide_details_about_used_unitsystem

        provide_details_about_used_unitsystem()

    def prepare_data(
        self, remove_self_energies: bool = True, normalize: bool = False
    ) -> None:
        """
        Prepares the dataset for use by calculating self energies, normalizing data, and splitting the dataset.

        Parameters
        ----------
        remove_self_energies : bool, optional
            Whether to remove self energies from the dataset. Defaults to True.
        normalize : bool, optional
            Whether to normalize the dataset. Defaults to True.
        """

        if self.split_file and os.path.exists(self.split_file):
            self.split_idxs = np.load(self.split_file, allow_pickle=True)
            log.debug(f"Loaded split indices from {self.split_file}")
        else:
            log.debug("Splitting strategy will be applied")

        # generate dataset
        factory = DatasetFactory()
        self.dataset = factory.create_dataset(self.data)
        dataset_statistics = {}
        # calculate self energies
        if remove_self_energies:
            self_energies = TorchDataModule.calculate_self_energies(self.dataset)
            self.dataset = TorchDataModule.remove_self_energies(
                self.dataset, self_energies
            )
            dataset_statistics["self_energies"] = self_energies

        # calculate average and variance
        if normalize:
            stats = TorchDataModule.calculate_mean_and_variance(self.dataset)
            self.dataset = TorchDataModule.normalize_energies(self.dataset, stats)
            dataset_statistics["stddev"] = stats["stddev"]
            dataset_statistics["mean"] = stats["mean"]

        self.dataset_statistics = dataset_statistics
        self.setup()

    @classmethod
    def normalize_energies(cls, dataset, stats: Dict[str, float]) -> None:
        """
        Normalizes the energies in the dataset.
        """
        from tqdm import tqdm

        for i in tqdm(range(len(dataset)), desc="Adjusting Energies"):
            energy = dataset[i]["E_label"]
            # Normalize using the computed mean and std
            modified_energy = (energy - stats["mean"]) / stats["stddev"]
            dataset[i] = {"E_label": modified_energy}

        return dataset

    @classmethod
    def calculate_mean_and_variance(cls, dataset) -> Dict[str, float]:
        """
        Calculates the mean and variance of the dataset.

        """
        import numpy as np

        log.info("Calculating mean and variance for normalization")
        energies = np.array([dataset[i]["E_label"] for i in range(len(dataset))])
        stats = {"mean": energies.mean(), "stddev": energies.std()}
        log.info(f"Mean and standard deviation of the dataset:{stats}")
        return stats

    @classmethod
    def remove_self_energies(cls, dataset, self_energies: Dict[str, float]) -> None:
        """
        Removes the self energies from the dataset.

        Parameters
        ----------
        self_energies : Dict[str, float]
            Dictionary containing the self energies for each element in the dataset.
        """
        from tqdm import tqdm

        log.info("Removing self energies from the dataset")
        for i in tqdm(range(len(dataset)), desc="Removing Self Energies"):
            atomic_numbers = dataset[i]["atomic_numbers"]
            E_label = dataset[i]["E_label"]
            for idx, Z in enumerate(atomic_numbers):
                E_label -= self_energies[Z.item()]
            dataset[i] = {"E_label": E_label}
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
            energy = dataset[i]["E_label"]
            if normalize:
                # Normalize using the computed mean and std
                modified_energy = (energy - dataset_mean) / dataset_std
            else:
                # Only adjust by subtracting the mean
                modified_energy = energy - dataset_mean
            dataset[i] = {"E_label": modified_energy}

        return dataset

    def setup(self) -> None:
        """
        Sets up datasets for the train, validation, and test stages.

        Parameters
        ----------
        stage : Optional[str]
            The stage for which to set up the dataset. Can be 'fit', 'validate', 'test', or 'predict'. Defaults to None.
        """
        from torch.utils.data import Subset

        # Create subsets for training, validation, and testing
        if self.split_idxs and os.path.exists(self.split_file):
            self.train_dataset = Subset(self.dataset, self.split_idxs["train_idx"])
            self.val_dataset = Subset(self.dataset, self.split_idxs["val_idx"])
            self.test_dataset = Subset(self.dataset, self.split_idxs["test_idx"])
        else:
            # Fallback to manual splitting if split_idxs is not available
            self.train_dataset, self.val_dataset, self.test_dataset = self.split.split(
                self.dataset
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


def collate_conformers(
    conf_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    # TODO: once TorchDataset is reimplemented for general properties, reimplement this function using formats too.
    """Concatenate the Z, R, and E tensors from a list of molecules into a single tensor each, and return a new dictionary with the concatenated tensors."""
    atomic_numbers_list = []
    positions_list = []
    E_list = []
    atomic_subsystem_counts = []
    atomic_subsystem_indices = []
    atomic_subsystem_indices_referencing_dataset = []
    for idx, conf in enumerate(conf_list):
        atomic_numbers_list.append(conf["atomic_numbers"])
        positions_list.append(conf["positions"])
        E_list.append(conf["E_label"])
        atomic_subsystem_counts.extend(conf["atomic_subsystem_counts"])
        atomic_subsystem_indices.extend([idx] * conf["atomic_subsystem_counts"][0])
        atomic_subsystem_indices_referencing_dataset.extend(
            [conf["idx"]] * conf["atomic_subsystem_counts"][0]
        )
    atomic_numbers_cat = torch.cat(atomic_numbers_list)
    positions_cat = torch.cat(positions_list).requires_grad_(True)
    E_stack = torch.stack(E_list)
    return {
        "atomic_numbers": atomic_numbers_cat,
        "positions": positions_cat,
        "E_label": E_stack,
        "atomic_subsystem_counts": torch.tensor(
            atomic_subsystem_counts, dtype=torch.int32
        ),
        "atomic_subsystem_indices": torch.tensor(
            atomic_subsystem_indices, dtype=torch.int32
        ),
        "atomic_subsystem_indices_referencing_dataset": torch.tensor(
            atomic_subsystem_indices_referencing_dataset, dtype=torch.int32
        ),
    }

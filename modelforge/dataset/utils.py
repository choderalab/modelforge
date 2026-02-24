"""
Utility functions for dataset handling.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Subset, random_split

_ATOMIC_NUMBER_TO_ELEMENT: Dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
}

_ATOMIC_ELEMENT_TO_NUMBER = {v: k for k, v in _ATOMIC_NUMBER_TO_ELEMENT.items()}

if TYPE_CHECKING:
    from modelforge.dataset.dataset import TorchDataset


def normalize_energies(dataset, stats: Dict[str, float]) -> None:
    """
    Normalizes the energies in the dataset.
    """
    from tqdm import tqdm

    for i in tqdm(range(len(dataset)), desc="Adjusting Energies"):
        energy = dataset[i]["E"]
        # Normalize using the computed mean and std
        modified_energy = (energy - stats["mean"]) / stats["stddev"]
        dataset[i] = {"E": modified_energy}

    return dataset


from torch.utils.data import Dataset, DataLoader


def calculate_mean_and_variance(
    torch_dataset: Dataset, batch_size: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Calculates the mean and variance of the dataset.

    """
    from loguru import logger as log
    from modelforge.utils.misc import Welford
    from modelforge.dataset.dataset import collate_conformers

    online_estimator = Welford()
    nr_of_atoms = 0
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        collate_fn=collate_conformers,
        num_workers=4,
        multiprocessing_context="fork"
    )
    import tqdm

    # NOTE: while what is shown below works I didn't think this through complelty
    # NOTE: it might not matter, but revisit this again
    log.info("Calculating mean and variance of atomic energies")
    for batch_data in tqdm.tqdm(dataloader):
        E_scaled = (
            batch_data.metadata.per_system_energy
            / batch_data.metadata.atomic_subsystem_counts.view(-1, 1)
        )
        online_estimator.update(E_scaled)

    stats = {
        "per_atom_energy_mean": online_estimator.mean,
        "per_atom_energy_stddev": online_estimator.stddev,
    }
    log.info(f"Mean and standard deviation of the dataset:{stats}")
    return stats


from openff.units import unit


def _calculate_self_energies(torch_dataset, collate_fn) -> Dict[str, unit.Quantity]:
    from torch.utils.data import DataLoader
    from modelforge.utils.units import GlobalUnitSystem
    import torch
    from loguru import logger as log

    # Initialize variables to hold data for regression
    batch_size = 64
    # Determine the size of the counts tensor
    # note this is actually the total number of configurations, not just number of molecules
    num_configs = torch_dataset.length
    # Determine up to which Z we detect elements
    maximum_atomic_number = 100
    # Initialize the counts tensor
    counts = torch.zeros(num_configs, maximum_atomic_number + 1, dtype=torch.int16)
    # save energies in list
    energy_array = torch.zeros(torch_dataset.length, dtype=torch.float64)

    # for filling in the element count matrix
    molecule_counter = 0
    # counter for saving energy values
    current_index = 0
    # save unique atomic numbers in list
    unique_atomic_numbers = set()

    for batch in DataLoader(
        torch_dataset, batch_size=batch_size, collate_fn=collate_fn
    ):
        energies, atomic_numbers, molecules_id = (
            batch.metadata.per_system_energy.squeeze(),
            batch.nnp_input.atomic_numbers.squeeze(-1).to(torch.int64),
            batch.nnp_input.atomic_subsystem_indices.to(torch.int16),
        )
        # Update the energy array and unique atomic numbers set
        batch_size = energies.size(0)

        energy_array[current_index : current_index + batch_size] = energies.squeeze()
        current_index += batch_size
        unique_atomic_numbers |= set(atomic_numbers.tolist())
        atomic_numbers_ = atomic_numbers - 1

        # Count the occurrence of each atomic number in molecules
        for molecule_id in molecules_id.unique():
            mask = molecules_id == molecule_id
            counts[molecule_counter].scatter_add_(
                0,
                atomic_numbers_[mask],
                torch.ones_like(atomic_numbers_[mask], dtype=torch.int16),
            )
            molecule_counter += 1

    # Prepare the data for linear regression
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
        _ATOMIC_NUMBER_TO_ELEMENT[int(idx)]: energy
        * GlobalUnitSystem.get_units("energy")
        for idx, energy in zip(unique_atomic_numbers, least_squares_fit)
    }  # NOTE: we have changed this to use global unit system, so everything should be consistent.

    log.debug("Calculated self energies for elements.")

    log.info("Atomic self energies for each element:", self_energies)
    return self_energies


class SplittingStrategy(ABC):
    """
    Base class for dataset splitting strategies.

    Attributes
    ----------
    seed : int, optional
        Random seed for reproducibility.
    generator : torch.Generator, optional
        Torch random number generator.
    test_seed: int, optional
        An optional seed that can be used to provide a fixed set of indices for testing purposes.
        If provided, the dataset will first be split into (train+val) and test using this seed, and then the
        (train+val) set will be split into train and val using the main seed.
        This allows randomization of the train/val split while keeping the test set fixed.
    """

    def __init__(
        self,
        split: List[float],
        seed: Optional[int] = None,
        test_seed: Optional[int] = None,
    ):
        self.seed = seed
        self.test_seed = test_seed

        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)

        # if we set test_seed, we'll initialize a separate generator
        # this will be used to initially split off the test set
        # then the main generator will be used to split the train/val set
        if self.test_seed is not None:
            self.test_generator = torch.Generator().manual_seed(self.test_seed)
        else:
            self.test_generator = None
        self.train_size, self.val_size, self.test_size = split[0], split[1], split[2]
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []
        assert np.isclose(sum(split), 1.0), "Splits must sum to 1.0"

    @abstractmethod
    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Split the dataset according to the subclassed strategy.
        """

        raise NotImplementedError


class RandomSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset randomly.
    """

    def __init__(
        self,
        seed: int = 42,
        split: List[float] = [0.8, 0.1, 0.1],
        test_seed: Optional[int] = None,
    ):
        """
        Initializes the RandomSplittingStrategy with a specified seed and split ratios.

        This strategy splits a dataset randomly based on provided ratios for training, validation,
        and testing subsets. The sum of split ratios should be 1.0.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.
        split : List[float], optional
            List containing three float values representing the ratio of data for
            training, validation, and testing respectively, by default [0.8, 0.1, 0.1].
        test_seed : Optional[int], optional
            An optional seed that can be used to provide a fixed set of indices for testing purposes,
            by default None.
            If provided, the dataset will first be split into (train+val) and test using this seed, and then the
            (train+val) set will be split into train and val using the main seed.
            This allows randomization of the train/val split while keeping the test set fixed.

        Raises
        ------
        AssertionError
            If the sum of split ratios is not close to 1.0.

        Examples
        --------
        >>> dataset = [1, 2, 3, 4, 5]
        >>> random_split = RandomSplittingStrategy(seed=123, split=[0.7, 0.2, 0.1])
        >>> train_idx, val_idx, test_idx = random_split.split(dataset)
        """

        super().__init__(seed=seed, split=split, test_seed=test_seed)

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Splits the provided dataset into training, validation, and testing subsets based on the predefined ratios.

        This method uses the ratios defined during initialization to randomly partition the dataset.
        The result is a tuple of indices for each subset.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset to be split.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            A tuple containing three Subsets for training, validation, and testing subsets, respectively.

        Examples
        --------
        >>> dataset = TorchDataset(numpy_data)
        >>> random_strategy = RandomSplittingStrategy(seed=42, split=[0.7, 0.2, 0.1])
        >>> train_dataset, val_dataset, test_dataset = random_strategy.split(dataset)
        """

        logger.debug(f"Using random splitting strategy with seed {self.seed} ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        # if we do not have a test seed, we can just do a single random split
        # use the torch.random_split function
        if self.test_seed is None:
            train_d, val_d, test_d = random_split(
                dataset,
                lengths=[self.train_size, self.val_size, self.test_size],
                generator=self.generator,
            )
            self.train_indices, self.val_indices, self.test_indices = (
                list(train_d.indices),
                list(val_d.indices),
                list(test_d.indices),
            )
        # if we have a test seed, we need to first split off the test set
        # using the test seed, and then split the remaining data into train/val
        else:
            subset_lengths = calculate_size_of_splits(
                len(dataset),
                split_frac=[self.train_size, self.val_size, self.test_size],
            )

            self.train_indices, self.val_indices, self.test_indices = (
                two_stage_random_split(
                    len(dataset),
                    split_size=subset_lengths,
                    generator1=self.test_generator,
                    generator2=self.generator,
                )
            )

            train_d = Subset(dataset, self.train_indices)
            val_d = Subset(dataset, self.val_indices)
            test_d = Subset(dataset, self.test_indices)

        return (train_d, val_d, test_d)


class RandomRecordSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset randomly, keeping all configurations in a record in the same split.

    """

    def __init__(
        self,
        seed: int = 42,
        split: List[float] = [0.8, 0.1, 0.1],
        test_seed: Optional[int] = None,
    ):
        """
        This strategy splits a dataset randomly based on provided ratios for training, validation,
        and testing subsets. The sum of split ratios should be 1.0.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.
        split : List[float], optional
            List containing three float values representing the ratio of data for
            training, validation, and testing respectively, by default [0.8, 0.1, 0.1].
        test_seed : Optional[int], optional
            An optional seed that can be used to provide a fixed set of indices for testing purposes, by default None.
            If provided, the dataset will first be split into (train+val) and test using this seed, and then the
            (train+val) set will be split into train and val using the main seed.
            This allows randomization of the train/val split while keeping the test set fixed.
        Raises
        ------
        AssertionError
            If the sum of split ratios is not close to 1.0.

        Examples
        --------
        >>> dataset = [1, 2, 3, 4, 5]
        >>> random_split = RandomRecordSplittingStrategy(seed=123, split=[0.7, 0.2, 0.1])
        >>> train_idx, val_idx, test_idx = random_split.split(dataset)
        """

        super().__init__(split=split, seed=seed, test_seed=test_seed)

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Splits the provided dataset into training, validation, and testing subsets based on the predefined ratios.

        This method uses the ratios defined during initialization to randomly partition the dataset.
        The result is a tuple of indices for each subset.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset to be split.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            A tuple containing three Subsets for training, validation, and testing subsets, respectively.

        Examples
        --------
        >>> dataset = TorchDataset(numpy_data)
        >>> random_strategy = RandomSplittingStrategy(seed=42, split=[0.7, 0.2, 0.1])
        >>> train_dataset, val_dataset, test_dataset = random_strategy.split(dataset)
        """

        logger.debug(f"Using random splitting strategy with seed {self.seed} ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        train_d, val_d, test_d = random_record_split(
            dataset,
            lengths=[self.train_size, self.val_size, self.test_size],
            generator=self.generator,
            test_generator=self.test_generator,
        )

        return (train_d, val_d, test_d)


def calculate_size_of_splits(total_size: int, split_frac: List[float]) -> List[int]:
    """
    Calculate the size of each split based on the total size and the split ratios.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * total_size) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Parameters
    ----------
    total_size : int
        Total size of the dataset.
    split_frac : List[float]
        List containing three float values representing the ratio of data for
        training, validation, and testing respectively.

    Returns
    -------
    List[int]
        List containing the sizes of each split.
    """
    if np.isclose(sum(split_frac), 1) and sum(split_frac) <= 1:
        subset_lengths: List[int] = []

        for i, frac in enumerate(split_frac):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(np.floor(total_size * frac))
            subset_lengths.append(n_items_in_split)

        remainder = total_size - sum(subset_lengths)

        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

        return subset_lengths
    else:
        raise ValueError("Split ratios must sum to 1.0")


def two_stage_random_split(
    dataset_size: int,
    split_size: List[int],
    generator1: torch.Generator,
    generator2: torch.Generator,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Perform a two-stage random split of a dataset.

    In the first stage, the dataset is split into a combined training and validation set,
    and a test set using the first generator. In the second stage, the combined training
    and validation set is further split into separate training and validation sets using
    the second generator.

    Parameters
    ----------
    dataset_size : int
        Total size of the dataset.
    split_size : List[int]
        List containing three int values representing the sizes of the
        training, validation, and testing splits respectively.
    generator1 : torch.Generator
        Torch random number generator for the first stage of splitting.
    generator2 : torch.Generator
        Torch random number generator for the second stage of splitting.

    Returns
    -------
    Tuple[List[int], List[int], List[int]]
        A tuple containing three lists of indices for training, validation, and testing subsets, respectively.
    """
    if len(split_size) != 3 or not np.isclose(sum(split_size), dataset_size):
        raise ValueError(
            "Split must be a list of three integers that sum to the total dataset size."
        )

    train_size, val_size, test_size = split_size

    # First stage: Randomize all indices and Split into (train + val) and test
    indices_first_shuffle = torch.randperm(dataset_size, generator=generator1)

    test_indices = indices_first_shuffle[:test_size]
    train_val_indices = indices_first_shuffle[test_size:]

    # Second stage: Shuffle the (train + val) indices and  Split (train + val) into train and val

    indices_second_shuffle = train_val_indices[
        torch.randperm(len(train_val_indices), generator=generator2)
    ]

    train_indices = indices_second_shuffle[:train_size]
    val_indices = indices_second_shuffle[train_size : train_size + val_size]

    return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()


def random_record_split(
    dataset: "TorchDataset",
    lengths: List[Union[int, float]],
    generator: Optional[torch.Generator] = torch.default_generator,
    test_generator: Optional[torch.Generator] = None,
) -> List[Subset]:
    """
    Randomly split a TorchDataset into non-overlapping new datasets of given lengths, keeping all conformers in a record in the same split

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.


    Parameters
    ----------
    dataset : TorchDataset
        Dataset to be split.
    lengths : List[int]
        Lengths of splits to be produced.
    generator : Optional[torch.Generator], optional
        Generator used for the random permutation, by default None
    test_generator: Optional[torch.Generator], optional
        An optional generator that can be used to provide a fixed set of indices for testing purposes,
    Returns
    -------
    List[Subset]
        List of subsets of the dataset.

    """
    if np.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []

        subset_lengths = calculate_size_of_splits(
            total_size=dataset.record_len(), split_frac=lengths
        )  # type: ignore[arg-type]

        lengths = subset_lengths

        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != dataset.record_len():  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the number of records of the input dataset!"
        )
    if test_generator is None:
        record_indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]

    else:
        # if we do define the test_generator, we will first shuffle the entire dataset
        # and extract the test set
        # then reshuffle the remaining indices to get train and val

        training_indices, val_indices, test_indices = two_stage_random_split(
            dataset_size=dataset.record_len(),  # type: ignore[arg-type]
            split_size=lengths,
            generator1=test_generator,
            generator2=generator,
        )

        record_indices = training_indices + val_indices + test_indices

    indices_by_split: List[List[int]] = []
    for offset, length in zip(np.cumsum(lengths), lengths):
        indices = []
        for record_idx in record_indices[offset - length : offset]:
            indices.extend(dataset.get_series_mol_idxs(record_idx))
        indices_by_split.append(indices)

    if sum([len(indices) for indices in indices_by_split]) != len(dataset):
        raise ValueError(
            "Sum of all split lengths does not equal the length of the input dataset!"
        )

    return [
        Subset(dataset, indices_by_split[split_idx])
        for split_idx in range(len(lengths))
    ]


class FirstComeFirstServeSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset based on idx.

    Examples
    --------
    >>> dataset = [1, 2, 3, 4, 5]
    >>> strategy = FirstComeFirstServeSplittingStrategy()
    >>> train_idx, val_idx, test_idx = strategy.split(dataset)
    """

    def __init__(self, split: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(seed=42, split=split)

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        logger.debug(f"Using first come/first serve splitting strategy ...")
        logger.debug(
            f"Splitting dataset into {self.train_size}, {self.val_size}, {self.test_size} ..."
        )

        len_dataset = len(dataset)
        first_split_on = int(len_dataset * self.train_size)
        second_split_on = first_split_on + int(len_dataset * self.val_size)
        indices = np.arange(len_dataset, dtype=int)
        train_d, val_d, test_d = (
            Subset(dataset, list(indices[0:first_split_on])),
            Subset(dataset, list(indices[first_split_on:second_split_on])),
            Subset(dataset, list(indices[second_split_on:])),
        )

        return (train_d, val_d, test_d)


REGISTERED_SPLITTING_STRATEGIES = {
    "first_come_first_serve": FirstComeFirstServeSplittingStrategy,
    "random_record_splitting_strategy": RandomRecordSplittingStrategy,
    "random_conformer_splitting_strategy": RandomSplittingStrategy,
}

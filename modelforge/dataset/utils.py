from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
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
}


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
    )
    import tqdm

    # NOTE: while what is shown below works I didn't think this through complelty
    # NOTE: it might not matter, but revisit this again
    log.info("Calculating mean and variance of atomic energies")
    for batch_data in tqdm.tqdm(dataloader):
        E_scaled = (
            batch_data.metadata.E
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
    import torch
    from loguru import logger as log

    # Initialize variables to hold data for regression
    batch_size = 64
    # Determine the size of the counts tensor
    num_molecules = torch_dataset.number_of_records
    # Determine up to which Z we detect elements
    max_atomic_number = 100
    # Initialize the counts tensor
    counts = torch.zeros(num_molecules, max_atomic_number + 1, dtype=torch.int16)
    # save energies in list
    energy_array = torch.zeros(torch_dataset.number_of_records, dtype=torch.float64)
    # for filling in the element count matrix
    molecule_counter = 0
    # counter for saving energy values
    current_index = 0
    # save unique atomic numbers in list
    unique_atomic_numbers = set()

    for batch in DataLoader(
        torch_dataset, batch_size=batch_size, collate_fn=collate_fn
    ):
        a = 7
        energies, atomic_numbers, molecules_id = (
            batch.metadata.E.squeeze(),
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
        _ATOMIC_NUMBER_TO_ELEMENT[int(idx)]: energy * unit.kilojoule_per_mole
        for idx, energy in zip(unique_atomic_numbers, least_squares_fit)
    }  # NOTE: we are assinging units here without check that the dataset is in unit system

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
    """

    def __init__(
        self,
        split: List[float],
        seed: Optional[int] = None,
    ):
        self.seed = seed
        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)

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

    def __init__(self, seed: int = 42, split: List[float] = [0.8, 0.1, 0.1]):
        """
        Initializes the RandomSplittingStrategy with a specified seed and split ratios.

        This strategy splits a dataset randomly based on provided ratios for runtime_defaults, validation,
        and testing subsets. The sum of split ratios should be 1.0.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.
        split : List[float], optional
            List containing three float values representing the ratio of data for
            runtime_defaults, validation, and testing respectively, by default [0.8, 0.1, 0.1].

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

        super().__init__(seed=seed, split=split)

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Splits the provided dataset into runtime_defaults, validation, and testing subsets based on the predefined ratios.

        This method uses the ratios defined during initialization to randomly partition the dataset.
        The result is a tuple of indices for each subset.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset to be split.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            A tuple containing three Subsets for runtime_defaults, validation, and testing subsets, respectively.

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
        return (train_d, val_d, test_d)


class RandomRecordSplittingStrategy(SplittingStrategy):
    """
    Strategy to split a dataset randomly, keeping all conformers in a record in the same split.

    """

    def __init__(self, seed: int = 42, split: List[float] = [0.8, 0.1, 0.1]):
        """
        This strategy splits a dataset randomly based on provided ratios for runtime_defaults, validation,
        and testing subsets. The sum of split ratios should be 1.0.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.
        split : List[float], optional
            List containing three float values representing the ratio of data for
            runtime_defaults, validation, and testing respectively, by default [0.8, 0.1, 0.1].

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

        super().__init__(split=split, seed=seed)

    def split(self, dataset: "TorchDataset") -> Tuple[Subset, Subset, Subset]:
        """
        Splits the provided dataset into runtime_defaults, validation, and testing subsets based on the predefined ratios.

        This method uses the ratios defined during initialization to randomly partition the dataset.
        The result is a tuple of indices for each subset.

        Parameters
        ----------
        dataset : TorchDataset
            The dataset to be split.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            A tuple containing three Subsets for runtime_defaults, validation, and testing subsets, respectively.

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
        )

        return (train_d, val_d, test_d)


def random_record_split(
    dataset: "TorchDataset",
    lengths: List[Union[int, float]],
    generator: Optional[torch.Generator] = torch.default_generator,
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

    Returns
    -------
    List[Subset]
        List of subsets of the dataset.

    """
    if np.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []

        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                np.floor(dataset.record_len() * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)

        remainder = dataset.record_len() - sum(subset_lengths)  # type: ignore[arg-type]

        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

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

    record_indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]

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

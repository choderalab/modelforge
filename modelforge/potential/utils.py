"""
Utility functions for neural network potentials.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from openff.units import unit

from modelforge.dataset.dataset import NNPInput


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


def shared_config_prior():

    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
    }


import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class Dense(nn.Linear):
    """
    Fully connected linear layer with activation function.

    forward(input)
        Forward pass of the layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: nn.Module = nn.Identity(),
    ):
        """
        A linear or non-linear transformation

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. Default is True.
        activation_function : Type[torch.nn.Module] , optional
            Activation function to be applied. Default is nn.Identity(), which applies the identity function
            and makes this a linear transformation.
        """

        super().__init__(in_features, out_features, bias)

        self.activation_function = activation_function

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and activation function.

        """
        y = F.linear(input, self.weight, self.bias)
        return self.activation_function(y)


class DenseWithCustomDist(nn.Linear):
    """
    Fully connected linear layer with activation function.

    Attributes
    ----------
    weight_init_distribution : Callable
        distribution used to initialize the weights.
    bias_init_distribution : Callable
        Distribution used to initialize the bias.

    Methods
    -------
    reset_parameters()
        Reset the weights and bias using the specified initialization distributions.
    forward(input)
        Forward pass of the layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: nn.Module = nn.Identity(),
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        A linear or non-linear transformation

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. Default is True.
        activation_function : nn.Module , optional
            Activation function to be applied. Default is nn.Identity(), which applies the identity function
            and makes this a linear ransformation.
        weight_init : Callable, optional
            Callable to initialize the weights. Default is xavier_uniform_.
        bias_init : Callable, optional
            Function to initialize the bias. Default is zeros_.
        """
        # NOTE: these two variables need to come before the init
        self.weight_init_distribution = weight_init
        self.bias_init_distribution = bias_init

        super().__init__(
            in_features, out_features, bias
        )  # NOTE: the `reset_paramters` method is called in the super class

        self.activation_function = activation_function

    def reset_parameters(self):
        """
        Reset the weights and bias using the specified initialization distributions.
        """
        self.weight_init_distribution(self.weight)
        if self.bias is not None:
            self.bias_init_distribution(self.bias)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and activation function.

        """
        y = F.linear(input, self.weight, self.bias)
        return self.activation_function(y)


from typing import Dict

from openff.units import unit


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_2 = math.log(2.0)

    def forward(self, x: torch.Tensor):
        """
        Compute shifted soft-plus activation function.

        The shifted soft-plus activation function is defined as:
        y = ln(1 + exp(-x)) - ln(2)

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -----------
        torch.Tensor
            Shifted soft-plus of the input.
        """

        return F.softplus(x) - self.log_2


def pair_list(
    atomic_subsystem_indices: torch.Tensor,
    only_unique_pairs: bool = False,
) -> torch.Tensor:
    """Compute all pairs of atoms and their distances.

    Parameters
    ----------
    atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
        Atom indices to indicate which atoms belong to which molecule
    only_unique_pairs : bool, optional
        If True, only unique pairs are returned (default is False).
        Otherwise, all pairs are returned.
    """
    # generate index grid
    n = len(atomic_subsystem_indices)

    # get device that passed tensors lives on, initialize on the same device
    device = atomic_subsystem_indices.device

    if only_unique_pairs:
        i_indices, j_indices = torch.triu_indices(n, n, 1, device=device)
    else:
        # Repeat each number n-1 times for i_indices
        i_indices = torch.repeat_interleave(
            torch.arange(n, device=device), repeats=n - 1
        )

        # Correctly construct j_indices
        j_indices = torch.cat(
            [
                torch.cat(
                    (
                        torch.arange(i, device=device),
                        torch.arange(i + 1, n, device=device),
                    )
                )
                for i in range(n)
            ]
        )

    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = (
        atomic_subsystem_indices[i_indices] == atomic_subsystem_indices[j_indices]
    )

    # Apply mask to get final pair indices
    i_final_pairs = i_indices[same_molecule_mask]
    j_final_pairs = j_indices[same_molecule_mask]

    # concatenate to form final (2, n_pairs) tensor
    pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    return pair_indices.to(device)


from openff.units import unit


def convert_str_to_unit_in_dataset_statistics(
    dataset_statistic: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, unit.Quantity]]:
    for key, value in dataset_statistic.items():
        for sub_key, sub_value in value.items():
            dataset_statistic[key][sub_key] = unit.Quantity(sub_value)
    return dataset_statistic


def remove_units_from_dataset_statistics(
    dataset_statistic: Dict[str, Dict[str, unit.Quantity]]
) -> Dict[str, Dict[str, float]]:
    from openff.units import unit

    from modelforge.utils.units import chem_context

    dataset_statistic_without_units = {}
    for key, value in dataset_statistic.items():
        dataset_statistic_without_units[key] = {}
        for sub_key, sub_value in value.items():
            dataset_statistic_without_units[key][sub_key] = (
                unit.Quantity(sub_value).to(unit.kilojoule_per_mole, "chem").m
            )
    return dataset_statistic_without_units


def read_dataset_statistics(
    dataset_statistic_filename: str, remove_units: bool = False
):
    import toml

    # read file
    dataset_statistic = toml.load(dataset_statistic_filename)
    # convert to float (to kJ/mol and then strip the units)
    # dataset statistic is a Dict[str, Dict[str, unit.Quantity]], we need to strip the units
    if remove_units:
        return remove_units_from_dataset_statistics(dataset_statistic=dataset_statistic)
    else:
        return dataset_statistic


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: int,
) -> torch.Tensor:
    r"""
    Computes the softmax operation over values in the `src` tensor that share indices specified in the `index` tensor
    along a given axis `dim`.

    For one-dimensional tensors, the operation computes:

    .. math::
        \text{out}_i = \text{softmax}(\text{src})_i =
        \frac{\exp(\text{src}_i)}{\sum_j \exp(\text{src}_j)}

    where the summation :math:`\sum_j` is over all :math:`j` such that :math:`\text{index}_j = i`.

    Parameters
    ----------
    src : Tensor
        The source tensor containing the values to which the softmax operation will be applied.
    index : LongTensor
        The indices of elements to scatter, determining which elements in `src` are grouped together for the
        softmax calculation.
    dim : int
        The axis along which to index. Default is `-1`.
    dim_size : int
        The number of classes, i.e., the number of unique indices in `index`.

    Returns
    -------
    Tensor
        A tensor where the softmax operation has been applied along the specified dimension.

    Notes
    -----
    This implementation is adapted from the following source:
    `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter/blob/c31915e1c4ceb27b2e7248d21576f685dc45dd01/torch_scatter/composite/softmax.py>`_.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            "`scatter_softmax` can only be computed over tensors "
            "with floating point data types."
        )

    assert dim >= 0, f"dim must be non-negative, got {dim}"
    assert (
        dim < src.dim()
    ), f"dim must be less than the number of dimensions of src {src.dim()}, got {dim}"

    out_shape = [
        other_dim_size if (other_dim != dim) else dim_size
        for (other_dim, other_dim_size) in enumerate(src.shape)
    ]
    index = index.to(torch.int64)
    zeros = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    max_value_per_index = zeros.scatter_reduce(
        dim, index, src, "amax", include_self=False
    )
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = torch.zeros(
        out_shape, dtype=src.dtype, device=src.device
    ).scatter_add(dim, index, recentered_scores_exp)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU,
    "CeLU": nn.CELU,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "ShiftedSoftplus": ShiftedSoftplus,
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    # Add more activation functions as needed
}

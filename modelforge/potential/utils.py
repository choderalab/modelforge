from typing import Callable, Tuple, Union, List, Optional

import numpy as np
import torch
import torch.nn as nn


def sequential_block(
    in_features: int,
    out_features: int,
    activation_fct: Callable = nn.Identity,
    bias: bool = True,
) -> nn.Sequential:
    """
    Create a sequential block for the neural network.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    activation_fct : Callable, optional
        Activation function, default is nn.Identity.
    bias : bool, optional
        Whether to use bias in Linear layers, default is True.

    Returns
    -------
    nn.Sequential
        Sequential layer block.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        activation_fct(),
        nn.Linear(out_features, out_features),
    )


def _scatter_add(
    src: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int
) -> torch.Tensor:
    """
    Performs a scatter addition operation.

    Parameters
    ----------
    src : torch.Tensor
        Source tensor.
    index : torch.Tensor
        Index tensor.
    dim_size : int
        Dimension size.
    dim : int

    Returns
    -------
    torch.Tensor
        The result of the scatter addition.
    """
    shape = list(src.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=src.dtype, device=src.device)
    y = tmp.index_add(dim, index, src)
    return y


# NOTE: change the scatter_add to the native pytorch function
def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
def scatter_softmax(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Softmax operation over all values in :attr:`src` tensor that share indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = {\textrm{softmax}(\mathrm{src})}_i =
        \frac{\exp(\mathrm{src}_i)}{\sum_j \exp(\mathrm{src}_j)}

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        dim_size: The number of classes, i.e. the number of unique indices in `index`.

    :rtype: :class:`Tensor`

    Adapted from: https://github.com/rusty1s/pytorch_scatter/blob/c31915e1c4ceb27b2e7248d21576f685dc45dd01/torch_scatter/composite/softmax.py 
    """
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    out_shape = [
        other_dim_size
        if (other_dim != dim)
        else dim_size
        for (other_dim, other_dim_size)
        in enumerate(src.shape)
    ]

    index = broadcast(index, src, dim)
    max_value_per_index = torch.zeros(out_shape).scatter_reduce(dim, index, src, "amax", include_self=False)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = torch.zeros(out_shape).scatter_add(dim, index, recentered_scores_exp)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


def gaussian_rbf(
    d_ij: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian radial basis function (RBF) transformation.

    Parameters
    ----------
    d_ij : torch.Tensor
        coordinates.
    offsets : torch.Tensor
        Offsets for Gaussian functions.
    widths : torch.Tensor
        Widths for Gaussian functions.

    Returns
    -------
    torch.Tensor
        Transformed tensor with Gaussian RBF applied
    """

    coeff = -0.5 / torch.pow(widths, 2)
    diff = d_ij[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y.to(dtype=torch.float32)


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float):
        r"""
        Behler-style cosine cutoff module.

        Args:
            cutoff (float): The cutoff distance.

        Attributes:
            cutoff (torch.Tensor): The cutoff distance as a tensor.

        """
        super().__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)


def cosine_cutoff(d_ij: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Compute the cosine cutoff for a distance tensor.

    Parameters
    ----------
    d_ij : Tensor
        Pairwise distance tensor. Shape: [..., N]
    cutoff : float
        The cutoff distance.

    Returns
    -------
    Tensor
        The cosine cutoff tensor. Shape: [..., N]
    """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(d_ij * np.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut = input_cut * (d_ij < cutoff)
    return input_cut


class EnergyReadout(nn.Module):
    """
    Defines the energy readout module.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass for the energy readout.
    """

    def __init__(self, n_filters: int):
        """
        Initialize the EnergyReadout class.

        Parameters
        ----------
        n_filters : int
            Number of filters after the last message passing layer.
        """
        super().__init__()
        self.energy_layer = nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor, atomic_subsystem_counts: List[int]) -> torch.Tensor:
        """
        Forward pass for the energy readout.

        Parameters
        ----------
        x : Tensor, shape [n_atoms, n_filters]
            Input tensor for the forward pass.
        atomic_subsystem_counts : List[int], length [n_confs]
            Number of atoms in each subsystem.

        Returns
        -------
        Tensor, shape [n_confs, 1]
            The total energy tensor.
        """
        x = self.energy_layer(
            x
        )  # in [n_atoms, n_atom_basis], out [n_atoms, 1]
        atomic_subsystem_index = torch.repeat_interleave(torch.arange(len(atomic_subsystem_counts)), atomic_subsystem_counts).unsqueeze(-1)
        print("atomic_subsystem_counts", atomic_subsystem_counts)
        total_energy = torch.zeros(len(atomic_subsystem_counts), 1).scatter_add(0, atomic_subsystem_index, x) # in [batch, n_atoms, 1], out [batch, 1]

        return total_energy


class ShiftedSoftplus(nn.Module):
    """
    Compute shifted soft-plus activation function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.softplus(x) - np.log(2.0)


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function module.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass for the GaussianRBF.
    """

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Initialize the GaussianRBF class.

        Parameters
        ----------
        n_rbf : int
            Number of radial basis functions.
        cutoff : float
            The cutoff distance.
        start: float
            center of first Gaussian function.
        trainable: boolean
        If True, widths and offset of Gaussian functions are adjusted during training process.

        """
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.tensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset),
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GaussianRBF.

        Parameters
        ----------
        d_ij : torch.Tensor
            Pairwise distances for the forward pass.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return gaussian_rbf(d_ij, self.offsets, self.widths)


def _distance_to_radial_basis(
    d_ij: torch.Tensor, radial_basis: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert distances to radial basis functions.

    Parameters
    ----------
    d_ij : torch.Tensor, shape [n_pairs]
        Pairwise distances between atoms.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Radial basis functions, shape [n_pairs, n_rbf]
        - cutoff values, shape [n_pairs]
    """
    f_ij = radial_basis(d_ij)
    rcut_ij = cosine_cutoff(d_ij, radial_basis.cutoff)
    return f_ij, rcut_ij


# taken from torchani repository: https://github.com/aiqm/torchani
def neighbor_pairs_nopbc(
    mask: torch.Tensor, R: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """
    Calculate neighbor pairs without periodic boundary conditions.
    Parameters
    ----------
    mask : torch.Tensor
        Mask tensor to indicate invalid atoms, shape (batch_size, n_atoms).
        1 == is padding.
    R : torch.Tensor
        Coordinates tensor, shape (batch_size, n_atoms, 3).
    cutoff : float
        Cutoff distance for neighbors.

    Returns
    -------
    torch.Tensor
        Tensor containing indices of neighbor pairs, shape (n_pairs, 2).

    Notes
    -----
    This function assumes no periodic boundary conditions and calculates neighbor pairs based solely on the cutoff distance.

    Examples
    --------
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 0]])
    >>> R = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]])
    >>> cutoff = 1.5
    >>> neighbor_pairs_nopbc(mask, R, cutoff)
    """
    import math

    R = R.detach().masked_fill(mask.unsqueeze(-1), math.nan)
    current_device = R.device
    num_atoms = mask.shape[1]
    num_mols = mask.shape[0]
    p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
    p12_all_flattened = p12_all.view(-1)
    pair_coordinates = R.index_select(1, p12_all_flattened).view(num_mols, 2, -1, 3)
    distances = (pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...]).norm(2, -1)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index] + molecule_index
    return atom_index12


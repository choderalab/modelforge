from typing import Callable, Tuple, Union

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

    def __init__(self, n_atom_basis: int):
        """
        Initialize the EnergyReadout class.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis.
        """
        super().__init__()
        self.energy_layer = nn.Linear(n_atom_basis, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the energy readout.

        Parameters
        ----------
        x : Tensor, shape [batch, n_atoms, n_atom_basis]
            Input tensor for the forward pass.

        Returns
        -------
        Tensor, shape [batch, 1]
            The total energy tensor.
        """
        x = self.energy_layer(
            x
        )  # in [batch, n_atoms, n_atom_basis], out [batch, n_atoms, 1]
        total_energy = x.sum(dim=1)  # in [batch, n_atoms, 1], out [batch, 1]
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
    padding_mask: torch.Tensor, coordinates: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Compute pairs of atoms that are neighbors (doesn't use PBC)

    This function bypasses the calculation of shifts and duplication
    of atoms in order to make calculations faster

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cutoff (float): the cutoff inside which atoms are considered pairs
    """
    import math

    coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
    current_device = coordinates.device
    num_atoms = padding_mask.shape[1]
    num_mols = padding_mask.shape[0]
    p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
    p12_all_flattened = p12_all.view(-1)

    pair_coordinates = coordinates.index_select(1, p12_all_flattened).view(
        num_mols, 2, -1, 3
    )
    distances = (pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...]).norm(2, -1)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index] + molecule_index
    return atom_index12

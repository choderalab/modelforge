import torch
from typing import Callable, Union
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from loguru import logger


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
    inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian radial basis function (RBF) transformation.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.
    offsets : torch.Tensor
        Offsets for Gaussian functions.
    widths : torch.Tensor
        Widths for Gaussian functions.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """

    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y.to(dtype=torch.float32)


def cosine_cutoff(d_ij: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Compute the cosine cutoff for a distance tensor.

    Parameters
    ----------
    d_ij : torch.Tensor
        Pairwise distance tensor.
    cutoff : float
        Cutoff distance.
    Returns
    -------
    torch.Tensor
        The cosine cutoff tensor.
    """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(d_ij * np.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= d_ij < cutoff
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
        x : torch.Tensor, shape [batch, n_atoms, n_atom_basis]
            Input tensor for the forward pass.

        Returns
        -------
        torch.Tensor
            The output tensor.
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
        self,
        n_rbf: int,
        cutoff: float,
    ):
        """
        Initialize the GaussianRBF class.

        Parameters
        ----------
        n_rbf : int
            Number of radial basis functions.
        cutoff : float
            The cutoff distance.
        """
        super().__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(0, cutoff, n_rbf)
        widths = torch.tensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset),
        )
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GaussianRBF.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the forward pass.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return gaussian_rbf(inputs, self.offsets, self.widths)


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

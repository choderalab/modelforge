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

    def forward(
        self, x: torch.Tensor, atomic_subsystem_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the energy readout.

        Parameters
        ----------
        x : Tensor, shape [nr_of_atoms_in_batch, n_atom_basis]
            Input tensor for the forward pass.

        Returns
        -------
        Tensor, shape [nr_of_moleculs_in_batch, 1]
            The total energy tensor.
        """
        import torch_scatter

        x = self.energy_layer(x)

        # Perform scatter add operation
        result = torch_scatter.scatter_add(
            x.t(), atomic_subsystem_indices.to(torch.int64), dim=1
        ).t()

        # Sum across feature dimension to get final tensor of shape (num_molecules, 1)
        total_energy_per_molecule = result.sum(dim=1, keepdim=True)

        return total_energy_per_molecule


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


def neighbor_pairs_nopbc(
    coordinates: torch.Tensor, atomic_subsystem_indices: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Compute pairs of atoms that are neighbors (doesn't use PBC)

    Parameters
    ----------
    coordinates : torch.Tensor, shape (nr_atoms_per_systems, 3)
    atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
        Atom indices to indicate which atoms belong to which molecule
    cutoff : float
        the cutoff inside which atoms are considered pairs
    """
    positions = coordinates.detach()
    # generate index grid
    n = len(atomic_subsystem_indices)
    i_indices, j_indices = torch.triu_indices(n, n, 1)
    print(atomic_subsystem_indices[i_indices])
    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = (
        atomic_subsystem_indices[i_indices] == atomic_subsystem_indices[j_indices]
    )

    # Apply mask to get final pair indices
    i_final_pairs = i_indices[same_molecule_mask]
    j_final_pairs = j_indices[same_molecule_mask]

    # concatenate to form final (2, n_pairs) tensor
    pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    # create pair_coordinates tensor
    pair_coordinates = positions[pair_indices.T]
    pair_coordinates = pair_coordinates.view(-1, 2, 3)

    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )
    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )

    # Find pairs within the cutoff
    in_cutoff = (distances <= cutoff).nonzero(as_tuple=False).squeeze()

    # Get the atom indices within the cutoff
    pair_indices_within_cutoff = pair_indices[:, in_cutoff]

    return pair_indices_within_cutoff

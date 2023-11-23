from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class SlicedEmbedding(nn.Module):
    """
    A module that performs embedding on a selected slice of input tensor.

    Parameters
    ----------
    max_Z : int
        Highest atomic number to embed, this will define the upper bound of the vocabulary that is used for embedding.
    embedding_dim : int
        Size of the embedding.
    sliced_dim : int, optional
        The dimension along which to slice the input tensor (default is 0).
        This is relevant since the input dimensions are (n_atoms_in_batch, nr_of_properties, property_size), but we want to embed a specific
        property.

    Attributes
    ----------
    embedding : nn.Embedding
        The embedding layer.
    sliced_dim : int
        The dimension along which the input tensor is sliced.

    Methods
    -------
    forward(x)
        Forward pass for the Embedding.

    Properties
    ----------
    embedding_dim : int
        The dimensionality of the embedding.

    """

    def __init__(self, max_Z: int, embedding_dim: int, sliced_dim: int = 0):
        """
        Initialize the Embedding class.

        Parameters
        ----------
        max_Z : int
            Highest atomic numbers.
        embedding_dim : int
            Dimensionality of the embedding.
        sliced_dim : int, optional
            The dimension along which to slice the input tensor (default is 0).
        """
        super().__init__()
        self.embedding = nn.Embedding(max_Z, embedding_dim)
        self.sliced_dim = sliced_dim

    @property
    def embedding_dim(self):
        """
        Get the dimensionality of the embedding.

        Returns
        -------
        int
            The dimensionality of the embedding.
        """
        return self.embedding.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for the forward pass.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        selected_tensor = x[:, self.sliced_dim, ...]

        return self.embedding(selected_tensor)


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

    def __init__(self, n_atom_basis: int, nr_of_layers: int = 1):
        """
        Initialize the EnergyReadout class.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis.
        """
        super().__init__()
        if nr_of_layers == 1:
            self.energy_layer = nn.Linear(n_atom_basis, 1)
        else:
            activation_fct = nn.ReLU()
            energy_layer_start = nn.Linear(n_atom_basis, n_atom_basis)
            energy_layer_end = nn.Linear(n_atom_basis, 1)
            energy_layer_intermediate = [
                (nn.Linear(n_atom_basis, n_atom_basis), activation_fct)
                for _ in range(nr_of_layers - 2)
            ]
            self.energy_layer = nn.Sequential(
                energy_layer_end, *energy_layer_intermediate, energy_layer_start
            )

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

        x = self.energy_layer(x)

        # Perform scatter add operation
        indices = atomic_subsystem_indices.to(torch.int64).unsqueeze(1)
        result = torch.zeros(len(atomic_subsystem_indices.unique()), 1).scatter_add(
            0, indices, x
        )

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
            The cutoff distance. NOTE: IN ANGSTROM #FIXME
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

    # Find pairs within the cutoff
    in_cutoff = (distances <= cutoff).nonzero(as_tuple=False).squeeze()

    # Get the atom indices within the cutoff
    pair_indices_within_cutoff = pair_indices[:, in_cutoff]

    return pair_indices_within_cutoff

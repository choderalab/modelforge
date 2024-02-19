from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def triple_by_molecule(
    atom_pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input: indices for pairs of atoms that are close to each other.
    each pair only appear once, i.e. only one of the pairs (1, 2) and
    (2, 1) exists.

    NOTE: this function is taken from https://github.com/aiqm/torchani/blob/17204c6dccf6210753bc8c0ca4c92278b60719c9/torchani/aev.py
            with little modifications.
    """

    def cumsum_from_zero(input_: torch.Tensor) -> torch.Tensor:
        cumsum = torch.zeros_like(input_)
        torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
        return cumsum

    # convert representation from pair to central-others
    ai1 = atom_pairs.view(-1)
    sorted_ai1, rev_indices = ai1.sort()

    # sort and compute unique key
    uniqued_central_atom_index, counts = torch.unique_consecutive(
        sorted_ai1, return_inverse=False, return_counts=True
    )

    # compute central_atom_index
    pair_sizes = counts * (counts - 1) // 2
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)

    # do local combinations within unique key, assuming sorted
    m = counts.max().item() if counts.numel() > 0 else 0
    n = pair_sizes.shape[0]
    intra_pair_indices = (
        torch.tril_indices(m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
    )
    mask = (
        torch.arange(intra_pair_indices.shape[2], device=ai1.device)
        < pair_sizes.unsqueeze(1)
    ).flatten()
    sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
    sorted_local_index12 += cumsum_from_zero(counts).index_select(0, pair_indices)

    # unsort result from last part
    local_index12 = rev_indices[sorted_local_index12]

    # compute mapping between representation of central-other to pair
    n = atom_pairs.shape[1]
    sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
    return central_atom_index, local_index12 % n, sign12


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
    def data(self):
        return self.embedding.weight.data

    @data.setter
    def data(self, data):
        self.embedding.weight.data = data

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


from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_
import torch.nn.functional as F


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


from openff.units import unit


class CosineCutoff(nn.Module):
    def __init__(self, cutoff: unit.Quantity):
        """
        Behler-style cosine cutoff module.

        Parameters:
        ----------
        cutoff: unit.Quantity
            The cutoff distance.

        """
        super().__init__()
        cutoff = cutoff.to(unit.nanometer).m
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):
        """
        Compute the cosine cutoff for a distance tensor.
        NOTE: the cutoff function doesn't care about units as long as they are consisten,

        Parameters
        ----------
        d_ij : Tensor
            Pairwise distance tensor. Shape: [n_pairs, distance]

        Returns
        -------
        Tensor
            The cosine cutoff tensor. Shape: [..., N]
        """
        # Compute values of cutoff function
        input_cut = 0.5 * (torch.cos(d_ij * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        input_cut *= (d_ij < self.cutoff).float()
        return input_cut


def embed_atom_features(
    atomic_numbers: torch.Tensor, embedding: nn.Embedding
) -> torch.Tensor:
    """
    Embed atomic numbers to atom features.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers of the atoms.
    embedding : nn.Embedding
        The embedding layer.

    Returns
    -------
    torch.Tensor
        The atom features.
    """
    # Perform atomic embedding
    from .utils import SlicedEmbedding

    assert isinstance(embedding, SlicedEmbedding), "embedding must be SlicedEmbedding"
    assert embedding.embedding_dim > 0, "embedding_dim must be > 0"

    atomic_embedding = embedding(
        atomic_numbers
    )  # shape (nr_of_atoms_in_batch, nr_atom_basis)
    return atomic_embedding


from typing import Dict


class EnergyReadout(nn.Module):
    """
    Defines the energy readout module.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass for the energy readout.
    """

    def __init__(self, nr_atom_basis: int, nr_of_layers: int = 1):
        """
        Initialize the EnergyReadout class.

        Parameters
        ----------
        nr_atom_basis : int
            Number of atom basis.
        """
        super().__init__()
        if nr_of_layers == 1:
            self.energy_layer = nn.Linear(nr_atom_basis, 1)
        else:
            activation_fct = nn.ReLU()
            energy_layer_start = nn.Linear(nr_atom_basis, nr_atom_basis)
            energy_layer_end = nn.Linear(nr_atom_basis, 1)
            energy_layer_intermediate = [
                (nn.Linear(nr_atom_basis, nr_atom_basis), activation_fct)
                for _ in range(nr_of_layers - 2)
            ]
            self.energy_layer = nn.Sequential(
                energy_layer_end, *energy_layer_intermediate, energy_layer_start
            )

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the energy readout.

        Parameters
        ----------
        input : Dict[str, torch.Tensor],
            "scalar_representation", shape [nr_of_atoms_in_batch, nr_atom_basis]
            "atomic_subsystem_indices", shape [nr_of_atoms_in_batch]
        Returns
        -------
        Tensor, shape [nr_of_moleculs_in_batch, 1]
            The total energy tensor.
        """
        x = self.energy_layer(input["scalar_representation"])
        atomic_subsystem_indices = input["atomic_subsystem_indices"]

        # Perform scatter add operation
        indices = atomic_subsystem_indices.unsqueeze(1).to(torch.int64)
        result = torch.zeros(
            len(atomic_subsystem_indices.unique()), 1, dtype=x.dtype, device=x.device
        ).scatter_add(0, indices, x)

        # Sum across feature dimension to get final tensor of shape (num_molecules, 1)
        total_energy_per_molecule = result.sum(dim=1, keepdim=True)
        return total_energy_per_molecule


def _shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    from torch.nn import functional
    import math

    return functional.softplus(x) - math.log(0.2)


from typing import Optional


class AngularSymmetryFunction(nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()


class RadialSymmetryFunction(nn.Module):
    """
    Gaussian Radial Basis Function module.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass for the RadialSymmetryFunction.
    """

    def __init__(
        self,
        number_of_gaussians: int,
        radial_cutoff: unit.Quantity,
        radial_start: unit.Quantity = 0.0 * unit.nanometer,
        dtype: Optional[torch.dtype] = None,
        ani_style: bool = False,
    ):
        """
        Initialize the RadialSymmetryFunction class.

        Parameters
        ----------
        number_of_gaussians : int
            Number of radial basis functions.
        cutoff : unit.Quantity
            The cutoff distance.
        start: float
            center of first Gaussian function.

        """
        from loguru import logger as log

        super().__init__()
        self.number_of_gaussians = number_of_gaussians
        self.radial_cutoff = radial_cutoff
        _unitless_radial_cutoff = radial_cutoff.to(unit.nanometer).m
        self.radial_start = radial_start
        _unitless_radial_start = radial_start.to(unit.nanometer).m
        # calculate offsets
        # ===============
        if ani_style:
            offsets = torch.linspace(
                _unitless_radial_start,
                _unitless_radial_cutoff,
                number_of_gaussians + 1,
                dtype=dtype,
            )[:-1]
        else:
            offsets = torch.linspace(
                _unitless_radial_start,
                _unitless_radial_cutoff,
                number_of_gaussians,
                dtype=dtype,
            )  # R_s
        # calculate eta
        # ===============
        if ani_style:
            eta = (torch.tensor([19.7]) * torch.ones_like(offsets)).to(
                dtype
            )  # since we are in nanometer
            eta = eta * 100  # NOTE: this is a hack to get eta to be in the right range
            prefactor = 0.25
        else:
            widths = (torch.abs(offsets[1] - offsets[0]) * torch.ones_like(offsets)).to(
                dtype
            )
            eta = 0.5 / torch.pow(widths, 2)  # eta
            prefactor = 1.0
        # ===============

        self.register_buffer("R_s", offsets)
        self.prefactor = prefactor
        self.register_buffer("eta", eta)
        log.info(
            f"RadialSymmetryFunction: cutoff={self.radial_cutoff}, number_of_gaussians={self.number_of_gaussians}, eta={eta}"
        )

    def forward(self, R_ij: torch.Tensor) -> torch.Tensor:
        """
        Computes the radial basis functions for the distance tensor.
        This computes the terms of the following equation
        G_{m}^{R} = \sum_{j!=i}^{N} exp(-eta(|R_{ij} - R_s|)^2))
        (NOTE: sum is not performed)
        Parameters
        ----------
        d_ij : torch.Tensor
            Pairwise distances.

        Returns
        -------
        torch.Tensor
            The radial basis functions. Shape: [..., N, number_of_gaussians]
        """
        diff = R_ij[..., None] - self.R_s  # R_ij - R_s
        y = self.prefactor * torch.exp((-1 * self.eta) * torch.pow(diff, 2))
        return y


def _distance_to_radial_basis(
    d_ij: torch.Tensor, radial_symmetry_function_module: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert distances to radial basis functions.

    Parameters
    ----------
    d_ij : torch.Tensor, shape [n_pairs,1 ]
        Pairwise distances between atoms.
    radial_symmetry_function_module : Callable
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Radial basis functions, shape [n_pairs, number_of_gaussians]
        - cutoff values, shape [n_pairs]
    """
    assert d_ij.dim() == 2
    f_ij = radial_symmetry_function_module(d_ij)
    cosine_cutoff_module = CosineCutoff(radial_symmetry_function_module.radial_cutoff)
    rcut_ij = cosine_cutoff_module(d_ij)
    return f_ij, rcut_ij


def _pair_list(
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


def _neighbor_list_with_cutoff(
    coordinates: torch.Tensor,  # in nanometer
    atomic_subsystem_indices: torch.Tensor,
    cutoff: float,
    only_unique_pairs: bool = False,
) -> torch.Tensor:
    """Compute all pairs of atoms and their distances.

    Parameters
    ----------
    coordinates : torch.Tensor, shape (nr_atoms_per_systems, 3), in nanometer
    atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
        Atom indices to indicate which atoms belong to which molecule
    cutoff : float
        The cutoff distance.
    """
    positions = coordinates.detach()
    pair_indices = _pair_list(
        atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )

    # create pair_coordinates tensor
    pair_coordinates = positions[pair_indices.T]
    pair_coordinates = pair_coordinates.view(-1, 2, 3)

    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )

    # Find pairs within the cutoff
    # cutoff = cutoff.to(unit.nanometer).m
    in_cutoff = (distances <= cutoff).nonzero(as_tuple=False).squeeze()

    # Get the atom indices within the cutoff
    pair_indices_within_cutoff = pair_indices[:, in_cutoff]

    return pair_indices_within_cutoff

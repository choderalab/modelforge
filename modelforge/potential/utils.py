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

    def forward(self, input: torch.Tensor):
        return _cosine_cutoff(input, self.cutoff)


def _cosine_cutoff(d_ij: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Compute the cosine cutoff for a distance tensor.
    NOTE: the cutoff function doesn't care about units as long as they are consisten,

    Parameters
    ----------
    d_ij : Tensor
        Pairwise distance tensor. Shape: [n_pairs, distance]
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
    input_cut *= (d_ij < cutoff).float()
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


from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass
class AtomicSelfEnergies:
    """
    AtomicSelfEnergies stores a mapping of atomic elements to their self energies.

    Provides lookup by atomic number or symbol, iteration over the mapping,
    and utilities to convert between atomic number and symbol.

    Intended as a base class to be extended with specific element-energy values.
    """    
    # We provide a dictionary with {str:float} of element name to atomic self-energy,
    # which can then be accessed by atomic index or element name
    energies: Dict[str, float] = field(default_factory=dict)
    # Example mapping, replace or extend as necessary
    atomic_number_to_element: Dict[int, str] = field(
        default_factory=lambda: {
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
            # Add more elements as needed
        }
    )

    def __getitem__(self, key):
        if isinstance(key, int):
            # Convert atomic number to element symbol
            element = self.atomic_number_to_element.get(key)
            if element is None:
                raise KeyError(f"Atomic number {key} not found.")
            return self.energies.get(element)
        elif isinstance(key, str):
            # Directly access by element symbol
            if key not in self.energies:
                raise KeyError(f"Element {key} not found.")
            return self.energies[key]
        else:
            raise TypeError(
                "Key must be an integer (atomic number) or string (element name)."
            )

    def __iter__(self) -> Iterator[Dict[str, float]]:
        """Iterate over the energies dictionary."""
        for element, energy in self.energies.items():
            atomic_number = self.element_to_atomic_number(element)
            yield (atomic_number, energy)

    def __len__(self) -> int:
        """Return the number of element-energy pairs."""
        return len(self.energies)

    def element_to_atomic_number(self, element: str) -> int:
        """Return the atomic number for a given element symbol."""
        for atomic_number, elem_symbol in self.atomic_number_to_element.items():
            if elem_symbol == element:
                return atomic_number
        raise ValueError(f"Element symbol '{element}' not found in the mapping.")


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        import math

        self.log_2 = math.log(2.0)

    def forward(self, x: torch.Tensor):
        """Compute shifted soft-plus activation function.

        y = \ln\left(1 + e^{-x}\right) - \ln(2)

        Parameters:
        -----------
        x:torch.Tensor
            input tensor

        Returns:
        -----------
        torch.Tensor: shifted soft-plus of input.

        """
        from torch.nn import functional

        return functional.softplus(x) - self.log_2


from typing import Optional


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
        cutoff: unit.Quantity,
        start: unit.Quantity = 0.0 * unit.nanometer,
        trainable: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the GaussianRBF class.

        Parameters
        ----------
        n_rbf : int
            Number of radial basis functions.
        cutoff : unit.Quantity
            The cutoff distance.
        start: unit.Quantity
            center of first Gaussian function.
        trainable: boolean
        If True, widths and offset of Gaussian functions are adjusted during training process.

        """
        super().__init__()
        self.n_rbf = n_rbf
        cutoff = cutoff.to(unit.nanometer).m
        start = start.to(unit.nanometer).m
        self.cutoff = cutoff
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf, dtype=dtype)
        widths = (torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)).to(dtype)

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
    d_ij : torch.Tensor, shape [n_pairs,1 ]
        Pairwise distances between atoms.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Radial basis functions, shape [n_pairs, n_rbf]
        - cutoff values, shape [n_pairs]
    """
    assert d_ij.dim() == 2
    f_ij = radial_basis(d_ij)
    rcut_ij = _cosine_cutoff(d_ij, radial_basis.cutoff)
    return f_ij, rcut_ij


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


def neighbor_list_with_cutoff(
    coordinates: torch.Tensor,  # in nanometer
    atomic_subsystem_indices: torch.Tensor,
    cutoff: unit.Quantity,
    only_unique_pairs: bool = False,
) -> torch.Tensor:
    """Compute all pairs of atoms and their distances.

    Parameters
    ----------
    coordinates : torch.Tensor, shape (nr_atoms_per_systems, 3), in nanometer
    atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
        Atom indices to indicate which atoms belong to which molecule
    cutoff : unit.Quantity
        The cutoff distance.
    """
    positions = coordinates.detach()
    pair_indices = pair_list(
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
    cutoff = cutoff.to(unit.nanometer).m
    in_cutoff = (distances <= cutoff).nonzero(as_tuple=False).squeeze()

    # Get the atom indices within the cutoff
    pair_indices_within_cutoff = pair_indices[:, in_cutoff]

    return pair_indices_within_cutoff


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

    max_value_per_index = torch.zeros(out_shape, dtype=src.dtype).scatter_reduce(dim, index, src, "amax", include_self=False)
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = torch.zeros(out_shape, dtype=src.dtype).scatter_add(dim, index, recentered_scores_exp)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)

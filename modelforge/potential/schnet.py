from typing import Tuple
from loguru import logger

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch import dtype

from modelforge.utils import Inputs

from .models import BaseNNP
from .utils import Dense, GaussianRBF, cosine_cutoff, shifted_softplus, scatter_add


class Schnet(BaseNNP):
    """
    Implementation of the SchNet architecture for quantum mechanical property prediction.
    """

    def __init__(
        self,
        n_atom_basis: int,  # number of features per atom
        n_interactions: int,  # number of interaction blocks
        n_filters: int = 0,  # number of filters
        dtype: dtype = torch.float32,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Initialize the SchNet model.

        Parameters
        ----------
        n_atom_basis : int
            Number of features per atom.
        n_interactions : int
            Number of interaction blocks.
        n_filters : int, optional
            Number of filters, defaults to None.
        dtype : torch.dtype, optional
            Data type for PyTorch tensors, defaults to torch.float32.
        device : torch.device, optional
            Device ("cpu" or "cuda") on which computations will be performed.

        """

        super().__init__(dtype, device)

        # generate atom embeddings
        # atoms are described by a tuple of features
        # X^{l} = x^{l}_{1},..., x^{l}_{n} with
        # x^{l}_{i} ∈ R^{F} with the number of feature maps F,
        # the number of atoms n and the current layer l

        # initialize atom embeddings
        max_z: int = 100  # max nuclear charge (i.e. atomic number)
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # initialize radial basis functions and other constants
        n_rbf = 20
        self.radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=5.0)
        self.cutoff = 5.0
        self.activation = shifted_softplus
        self.n_interactions = n_interactions
        self.n_atom_basis = n_atom_basis

        # initialize dense yalers for atom feature transformation
        # Dense layers are applied consecutively to the initialized atom embeddings x^{l}_{0}
        # to generate x_i^l+1 = W^lx^l_i + b^l
        logger.debug("in2f")
        print("in2f")
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        print("f2out")
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=self.activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )

        # Initialize filter network
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=self.activation),
            Dense(n_filters, n_filters),
        )

    def _setup_ase_system(self, inputs: Inputs) -> Atoms:
        """
        Transform inputs to an ASE Atoms object.

        Parameters
        ----------
        inputs : Inputs
            Input features including atomic numbers and positions.

        Returns
        -------
        ase.Atoms
            Transformed ASE Atoms object.

        """
        _atomic_numbers = torch.clone(inputs.Z)
        atomic_numbers = list(_atomic_numbers.detach().cpu().numpy())
        positions = list(inputs.R.detach().cpu().numpy())
        ase_atoms = Atoms(numbers=atomic_numbers, positions=positions)
        return ase_atoms

    def _compute_distances(
        self, inputs: Inputs
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Compute atomic distances using ASE's neighbor list.

        Parameters
        ----------
        inputs : Inputs
            Input features including atomic numbers and positions.

        Returns
        -------
        torch.Tensor, np.ndarray, np.ndarray
            Pairwise distances, index of atom i, and index of atom j.

        """

        ase_atoms = self._setup_ase_system(inputs)
        idx_i, idx_j, _, r_ij = neighbor_list(
            "ijSD", ase_atoms, 5.0, self_interaction=False
        )
        r_ij = torch.from_numpy(r_ij)
        return r_ij, idx_i, idx_j

    def _distance_to_radial_basis(self, r_ij):
        """
        Transform distances to radial basis functions.

        Parameters
        ----------
        r_ij : torch.Tensor
            Pairwise distances between atoms.

        Returns
        -------
        torch.Tensor, torch.Tensor
            Radial basis functions and cutoff values.

        """
        d_ij = torch.norm(r_ij, dim=1)  # calculate pairwise distances
        f_ij = self.radial_basis(d_ij)
        rcut_ij = cosine_cutoff(d_ij, self.cutoff)
        return f_ij, rcut_ij

    def _interaction_block(self, inputs: Inputs, f_ij, idx_i, idx_j, rcut_ij):
        """
        Compute the interaction block which updates atom features.

        Parameters
        ----------
        inputs : Inputs
            Input features including atomic numbers and positions.
        f_ij : torch.Tensor
            Radial basis functions.
        idx_i : np.ndarray
            Indices of center atoms.
        idx_j : np.ndarray
            Indices of neighboring atoms.
        rcut_ij : torch.Tensor
            Cutoff values for each pair of atoms.

        Returns
        -------
        torch.Tensor
            Updated atom features.

        """

        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        x_emb = self.embedding(inputs.Z)
        print("After embedding: x.shape", x_emb.shape)
        idx_i = torch.from_numpy(idx_i).to(self.device, torch.int64)

        # interaction blocks
        for _ in range(self.n_interactions):
            # atom wise update of features
            x = self.in2f(x_emb)
            print("After in2f: x.shape", x.shape)

            # Filter generation networks
            Wij = self.filter_network(f_ij)
            Wij = Wij * rcut_ij[:, None]
            Wij = Wij.to(dtype=self.dtype)

            # continuous-ﬁlter convolutional layers
            x_j = x[idx_j]
            x_ij = x_j * Wij
            x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
            # Update features
            x = self.f2out(x)

            x_emb = x_emb + x

        return x_emb

    def calculate_energies_and_forces(self, inputs: Inputs) -> torch.Tensor:
        """
        Compute energies and forces for given atomic configurations.

        Parameters
        ----------
        inputs : Inputs
            Input features including atomic numbers and positions.

        Returns
        -------
        torch.Tensor
            Energies and forces for the given configurations.

        """
        r_ij, idx_i, idx_j = self._compute_distances(inputs)
        f_ij, rcut_ij = self._distance_to_radial_basis(r_ij)
        x = self._interaction_block(inputs, f_ij, idx_i, idx_j, rcut_ij)
        return x

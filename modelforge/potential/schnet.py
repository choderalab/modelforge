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
from .utils import (
    GaussianRBF,
    cosine_cutoff,
    ShiftedSoftplus,
    scatter_add,
    EnergyReadout,
)


class Schnet(BaseNNP):
    def __init__(self, n_atom_basis, n_interactions, n_filters=0):
        super().__init__()
        self.representation = SchNetRepresentation(
            n_atom_basis, n_filters, n_interactions
        )
        self.readout = EnergyReadout(n_atom_basis)

    def calculate_energy(self, inputs):
        x = self.representation(inputs)
        # pool average over atoms
        return self.readout(x)


def sequential_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        ShiftedSoftplus(),
        nn.Linear(out_features, out_features),
    )


class SchNetInteractionBlock(nn.Module):
    def __init__(self, n_atom_basis, n_filters):
        super().__init__()
        n_rbf = 20
        self.intput_to_feature = nn.Linear(n_atom_basis, n_filters)
        self.feature_to_output = sequential_block(n_filters, n_atom_basis)
        self.filter_network = sequential_block(n_rbf, n_filters)

    def forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
        # atom wise update of features
        logger.debug(f"Input to feature: x.shape {x.shape}")
        x = self.intput_to_feature(x)
        logger.debug("After input_to_feature call: x.shape {x.shape}")

        # Filter generation networks
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]
        Wij = Wij.to(dtype=x.dtype)

        # continuous-ﬁlter convolutional layers
        x_j = x[idx_j]
        x_ij = x_j * Wij
        logger.debug("After x_j * Wij: x_ij.shape {x_ij.shape}")
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        logger.debug("After scatter_add: x.shape {x.shape}")
        # Update features
        x = self.feature_to_output(x)
        return x


class SchNetRepresentation(nn.Module):
    def __init__(self, n_atom_basis, n_filters, n_interactions):
        super().__init__()

        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.interactions = nn.ModuleList(
            [
                SchNetInteractionBlock(n_atom_basis, n_filters)
                for _ in range(n_interactions)
            ]
        )
        self.cutoff = 5.0
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=self.cutoff)

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

    def forward(self, inputs):
        logger.debug("Compute distances ...")
        r_ij, idx_i, idx_j = self._compute_distances(inputs)
        logger.debug("Convert distances to radial basis ...")
        f_ij, rcut_ij = self._distance_to_radial_basis(r_ij)
        logger.debug("Compute interaction block ...")

        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        logger.debug("Embedding inputs.Z")
        x = self.embedding(inputs.Z)
        logger.debug(f"After embedding: {x.shape=}")
        idx_i = torch.from_numpy(idx_i).to(torch.int64)
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        return x


# class SchNetPotential(BaseNNP):
#     def __init__(
#         self,
#         n_atom_basis: int,  # number of features per atom
#         n_interactions: int,  # number of interaction blocks
#         n_filters: int = 0,  # number of filters
#     )
#         super().__init__()
#         representation = SchNetRepresentation(
#             n_atom_basis, n_filters, n_gaussians, n_interactions
#         )
#         input_modules = [SchNetInputModule()]  # Optional
#         output_modules = [SchNetOutputModule()]  # Optional
#         super().__init__(representation, input_modules, output_modules)


# class SchNetInteractionBlock(nn.Module):
#     def __init__(
#         self, n_atom_basis: int, n_filters: int, n_gaussians: int, n_interactions: int
#     ):
#         super().__init__()
#         self.dense1 = nn.Linear(n_atom_basis, n_filters)
#         self.dense2 = nn.Linear(n_filters, n_atom_basis)
#         self.activation = nn.ReLU()
#         self.distance_expansion = nn.Linear(n_gaussians, n_filters, bias=False)

#     def forward(self, x, r, neighbors):
#         # Distance expansion
#         r_expanded = self.distance_expansion(r)

#         # Interaction with neighbors
#         for i, neighbors_i in enumerate(neighbors):
#             x_neighbors = x[neighbors_i]
#             r_neighbors = r_expanded[neighbors_i]
#             messages = self.activation(self.dense1(x_neighbors))
#             messages = self.dense2(messages * r_neighbors)
#             x[i] += messages.sum(dim=0)

#         return x


# class SchNetRepresentation(nn.Module):
#     def __init__(self, n_atom_basis, n_filters, n_gaussians, n_interactions):
#         super().__init__()
#         self.embedding = nn.Embedding(n_atom_basis, n_atom_basis)
#         self.interactions = nn.ModuleList(
#             [
#                 SchNetInteractionBlock(n_atom_basis, n_filters, n_gaussians)
#                 for _ in range(n_interactions)
#             ]
#         )

#     def forward(self, species, coordinates, neighbor_list):
#         # Embedding layer
#         x = self.embedding(species)

#         # Compute pairwise distances and neighbor list
#         r = torch.pdist(coordinates)
#         neighbors = neighbor_list  # Assuming neighbor_list is precomputed

#         # Interaction blocks
#         for interaction in self.interactions:
#             x = interaction(x, r, neighbors)

#         return x

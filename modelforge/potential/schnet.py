import numpy as np
import torch.nn as nn
from loguru import logger

from .models import BaseNNP
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    cosine_cutoff,
    scatter_add,
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
        logger.debug(f"Wij {Wij.shape=}")

        # continuous-ﬁlter convolutional layers
        logger.debug(f"Before x[idx_j]: x.shape {x.shape=}")
        logger.debug(f"idx_j.shape {idx_j.shape=}")
        x_j = x[idx_j]
        x_ij = x_j * Wij
        logger.debug(f"After x_j * Wij: x_ij.shape {x_ij.shape=}")
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        logger.debug(f"After scatter_add: x.shape {x.shape=}")
        # Update features
        x = self.feature_to_output(x)
        return x


class SchNetRepresentation(nn.Module):
    def __init__(self, n_atom_basis, n_filters, n_interactions):
        super().__init__()
        from .utils import neighbor_pairs_nopbc

        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=-1)
        self.interactions = nn.ModuleList(
            [
                SchNetInteractionBlock(n_atom_basis, n_filters)
                for _ in range(n_interactions)
            ]
        )
        self.cutoff = 5.0
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=self.cutoff)
        self.calculate_neighbors = neighbor_pairs_nopbc

    def _distance_to_radial_basis(self, d_ij):
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
        f_ij = self.radial_basis(d_ij)
        rcut_ij = cosine_cutoff(d_ij, self.cutoff)
        return f_ij, rcut_ij

    def compute_distance(self, atom_index12, R):
        coordinates = R
        coordinates_ = coordinates
        coordinates = coordinates_.flatten(0, 1)

        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(
            2, -1, 3
        )
        vec = selected_coordinates[0] - selected_coordinates[1]
        return vec.norm(2, -1)

    def forward(self, inputs):
        logger.debug("Compute distances ...")
        mask = inputs["Z"] == -1
        atom_index12 = self.calculate_neighbors(mask, inputs["R"], self.cutoff)
        d_ij = self.compute_distance(atom_index12, inputs["R"])
        logger.debug("Convert distances to radial basis ...")
        f_ij, rcut_ij = self._distance_to_radial_basis(d_ij)
        logger.debug("Compute interaction block ...")

        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        logger.debug("Embedding inputs.Z")
        Z = inputs["Z"]
        logger.debug(f"{Z.shape=}")
        x = self.embedding(Z)
        logger.debug(f"After embedding: {x.shape=}")
        idx_i = atom_index12[0]
        idx_j = atom_index12[1]
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

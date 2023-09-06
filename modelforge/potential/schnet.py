import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple

from .models import BaseNNP
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    cosine_cutoff,
    scatter_add,
)
import torch


class Schnet(BaseNNP):
    def __init__(
        self, n_atom_basis: int, n_interactions: int, n_filters: int = 0
    ) -> None:
        """
        Initialize the Schnet class.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        n_interactions : int
            Number of interaction blocks in the architecture.
        n_filters : int, optional
            Number of filters, defines the dimensionality of the intermediate features.
            Default is 0.

        """

        super().__init__()
        self.representation = SchNetRepresentation(
            n_atom_basis, n_filters, n_interactions
        )
        self.readout = EnergyReadout(n_atom_basis)

    def calculate_energy(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : dict, contains
            - 'Z': torch.Tensor, shape [batch_size, n_atoms]
                Atomic numbers for each atom in each molecule in the batch.
            - 'R': torch.Tensor, shape [batch_size, n_atoms, 3]
                Coordinates for each atom in each molecule in the batch.

        Returns
        -------
        torch.Tensor, shape [batch_size]
            Calculated energies for each molecule in the batch.

        """
        x = self.representation(inputs)
        logger.debug(f"{x.shape=}")
        # pool average over atoms
        return self.readout(x)


def sequential_block(in_features: int, out_features: int):
    """
    Create a sequential block for the neural network.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Returns
    -------
    nn.Sequential
        Sequential layer block.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        ShiftedSoftplus(),
        nn.Linear(out_features, out_features),
    )


class SchNetInteractionBlock(nn.Module):
    def __init__(self, n_atom_basis: int, n_filters: int):
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        n_filters : int
            Number of filters, defines the dimensionality of the intermediate features.

        """
        super().__init__()
        n_rbf = 20
        self.intput_to_feature = nn.Linear(n_atom_basis, n_filters)
        self.feature_to_output = sequential_block(n_filters, n_atom_basis)
        self.filter_network = sequential_block(n_rbf, n_filters)

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Input feature tensor for atoms.
        f_ij : torch.Tensor, shape [n_pairs, n_rbf]
            Radial basis functions for pairs of atoms.
        idx_i : torch.Tensor, shape [n_pairs]
            Indices for the first atom in each pair.
        idx_j : torch.Tensor, shape [n_pairs]
            Indices for the second atom in each pair.
        rcut_ij : torch.Tensor, shape [n_pairs]
            Cutoff values for each pair.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Updated feature tensor after interaction block.
        """
        batch_size, nr_of_atoms = x.shape[0], x.shape[1]

        logger.debug(f"Input to feature: {x.shape=}")
        logger.debug(f"Input to feature: {f_ij.shape=}")
        logger.debug(f"Input to feature: {idx_i.shape=}")
        logger.debug(f"Input to feature: {rcut_ij.shape=}")
        x = self.intput_to_feature(x)
        logger.debug(f"After input_to_feature call: {x.shape=}")
        x = x.flatten(0, 1)
        logger.debug(f"Flatten x: {x.shape=}")

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
        logger.debug(f"After feature_to_output: x.shape {x.shape=}")
        x = x.reshape(batch_size, nr_of_atoms, 128)
        logger.debug(f"After reshape: x.shape {x.shape=}")
        return x


class SchNetRepresentation(nn.Module):
    def __init__(self, n_atom_basis: int, n_filters: int, n_interactions: int):
        """
        Initialize the SchNet representation layer.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis.
        n_filters : int
            Number of filters.
        n_interactions : int
            Number of interaction layers.
        """
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

    def _distance_to_radial_basis(
        self, d_ij: torch.Tensor
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
        f_ij = self.radial_basis(d_ij)
        rcut_ij = cosine_cutoff(d_ij, self.cutoff)
        return f_ij, rcut_ij

    def compute_distance(
        self, atom_index12: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances based on atom indices and coordinates.

        Parameters
        ----------
        atom_index12 : torch.Tensor, shape [n_pairs, 2]
            Atom indices for pairs of atoms
        R : torch.Tensor, shape [batch_size, n_atoms, n_dims]
            Atom coordinates.

        Returns
        -------
        torch.Tensor, shape [n_pairs]
            Computed distances.
        """

        logger.debug(f"{atom_index12.shape=}")
        logger.debug(f"{R.shape=}")
        coordinates = R.flatten(0, 1)
        logger.debug(f"{coordinates.shape=}")
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(
            2, -1, 3
        )
        logger.debug(f"{selected_coordinates.shape=}")
        vec = selected_coordinates[0] - selected_coordinates[1]
        return vec.norm(2, -1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Dictionary containing input tensors, specifically atomic numbers and coordinates.
            - 'Z': Atomic numbers, shape [batch_size, n_atoms]
            - 'R': Atom coordinates, shape [batch_size, n_atoms, 3]


        Returns
        -------
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Output tensor after forward pass.
        """
        logger.debug("Compute distances ...")
        Z = inputs["Z"]
        R = inputs["R"]
        mask = Z == -1

        atom_index12 = self.calculate_neighbors(mask, R, self.cutoff)
        d_ij = self.compute_distance(atom_index12, R)
        logger.debug(f"{d_ij.shape=}")
        logger.debug("Convert distances to radial basis ...")
        f_ij, rcut_ij = self._distance_to_radial_basis(d_ij)
        logger.debug("Compute interaction block ...")

        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        logger.debug("Embedding inputs.Z")
        logger.debug(f"{Z.shape=}")
        x = self.embedding(Z)

        logger.debug(f"After embedding: {x.shape=}")
        idx_i = atom_index12[0]
        idx_j = atom_index12[1]
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        return x

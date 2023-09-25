import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple, List

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
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_filters: int = 0,
        cutoff: float = 5.0,
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
        cutoff : float, optional
            Cutoff value for the pairlist. Default is 5.0.
        """
        from .models import PairList

        super().__init__()

        self.calculate_distances_and_pairlist = PairList(cutoff)

        self.representation = SchNetRepresentation(
            n_atom_basis, n_filters, n_interactions
        )
        self.readout = EnergyReadout(n_atom_basis)
        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=-1)

        self.interactions = nn.ModuleList(
            [
                SchNetInteractionBlock(n_atom_basis, n_filters)
                for _ in range(n_interactions)
            ]
        )

    def calculate_energy(
        self, inputs: Dict[str, torch.Tensor], cached_pairlist: bool = False
    ) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : dict, contains
            - 'Z': torch.Tensor, shape [batch_size, n_atoms]
                Atomic numbers for each atom in each molecule in the batch.
            - 'R': torch.Tensor, shape [batch_size, n_atoms, 3]
                Coordinates for each atom in each molecule in the batch.
        cached_pairlist : bool, optional
            Whether to use a cached pairlist. Default is False. NOTE: is this really needed?
        Returns
        -------
        torch.Tensor, shape [batch_size]
            Calculated energies for each molecule in the batch.

        """
        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        Z = inputs["Z"]
        x = self.embedding(Z)
        mask = Z == -1
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])

        representation = self.representation(x, pairlist)
        for interaction in self.interactions:
            v = interaction(
                representation["f_ij"],
                representation["idx_i"],
                representation["idx_j"],
                representation["rcut_ij"],
            )
            x = x + v

        # x with shape torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]

        # pool average over atoms # TODO: check if is this correct?
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

        x = self.intput_to_feature(x)
        x = x.flatten(0, 1)

        # Filter generation networks
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]
        Wij = Wij.to(dtype=x.dtype)

        # continuous-ﬁlter convolutional layers
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        # Update features
        x = self.feature_to_output(x)
        x = x.reshape(batch_size, nr_of_atoms, 128)
        return x


class SchNetRepresentation(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_filters: int,
        n_interactions: int,
    ):
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

        self.cutoff = 5.0
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=self.cutoff)

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

    def forward(self, pairlist: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Input feature tensor for atoms.
        pairlist: Dict[str, torch.Tensor]
            Pairlist dictionary containing the following keys:
            - 'atom_index12': torch.Tensor, shape [n_pairs, 2]
                Atom indices for pairs of atoms
            - 'd_ij': torch.Tensor, shape [n_pairs]
                Pairwise distances between atoms.
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the following keys:
            - 'f_ij': torch.Tensor, shape [n_pairs, n_rbf]
                Radial basis functions for pairs of atoms.
            - 'idx_i': torch.Tensor, shape [n_pairs]
                Indices for the first atom in each pair.
            - 'idx_j': torch.Tensor, shape [n_pairs]
                Indices for the second atom in each pair.
            - 'rcut_ij': torch.Tensor, shape [n_pairs]
                Cutoff values for each pair.
        """

        atom_index12 = pairlist["atom_index12"]
        d_ij = pairlist["d_ij"]

        f_ij, rcut_ij = self._distance_to_radial_basis(d_ij)

        idx_i, idx_j = atom_index12[0], atom_index12[1]

        return {"f_ij": f_ij, "idx_i": idx_i, "idx_j": idx_j, "rcut_ij": rcut_ij}

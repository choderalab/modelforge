import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple, List, Type

from .models import BaseNNP, LighningModuleMixin
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    CosineCutoff,
    scatter_add,
    sequential_block,
    _distance_to_radial_basis,
)
import torch


class Schnet(BaseNNP):
    def __init__(
        self,
        nr_atom_basis: int,
        nr_interactions: int,
        nr_filters: int = 0,
        cutoff: float = 5.0,
        nr_of_embeddings: int = 100,
    ) -> None:
        """
        Initialize the Schnet class.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis; defines the dimensionality of the output features.
        n_interactions : int
            Number of interaction blocks in the architecture.
        n_filters : int, optional
            Number of filters; defines the dimensionality of the intermediate features (default is 0).
        cutoff : float, optional
            Cutoff value for the pairlist (default is 5.0).
        nr_of_embeddings: int, optional
            Number of embeddings (default is 100).
        """
        from .models import PairList  # Local import to avoid circular dependencies

        super().__init__()

        self.calculate_distances_and_pairlist = PairList(cutoff)
        self.representation = SchNetRepresentation(
            nr_atom_basis, nr_filters, nr_interactions
        )
        self.readout = EnergyReadout(nr_atom_basis)
        self.embedding = nn.Embedding(nr_of_embeddings, nr_atom_basis, padding_idx=0)

    def forward(
        self, inputs: Dict[str, torch.Tensor], cached_pairlist: bool = False
    ) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Inputs containing atomic numbers ('Z') and coordinates ('R').
            - 'Z': shape (batch_size, n_atoms)
            - 'R': shape (batch_size, n_atoms, 3)
        cached_pairlist : bool, optional
            Whether to use a cached pairlist (default is False).

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (batch_size,).
        """
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        Z = inputs["Z"]
        x = self.embedding(Z)  # shape (batch_size, n_atoms, n_atom_basis)
        mask = Z == 0
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])

        x = self.representation(
            x, pairlist
        )  # shape (batch_size, n_atoms, n_atom_basis)
        # pool average over atoms
        return self.readout(x)  # shape (batch_size,)


class SchNetInteractionBlock(nn.Module):
    def __init__(self, nr_atom_basis: int, nr_filters: int, nr_rbf: int = 20):
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        n_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        n_rbf : int, optional
            Number of radial basis functions. Default is 20.
        """
        super().__init__()
        nr_rbf = 20
        self.intput_to_feature = nn.Linear(nr_atom_basis, nr_filters)
        self.feature_to_output = sequential_block(
            nr_filters, nr_atom_basis, ShiftedSoftplus
        )
        self.filter_network = sequential_block(nr_rbf, nr_filters, ShiftedSoftplus)

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

        # Using custom scatter_add
        x_custom = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        # Using native scatter_add
        shape = list(x.shape)  # note that we're using x.shape, not x_ij.shape
        x_native = torch.zeros(shape, dtype=x.dtype)

        # Extend the dimensionality of idx_i to match that of x_native
        idx_i_expanded = idx_i.unsqueeze(1).expand_as(x_ij)

        # Perform the scatter_add operation
        x_native.scatter_add_(0, idx_i_expanded, x_ij)

        assert torch.equal(x_native, x_custom)

        # Update features
        x = self.feature_to_output(x_native)
        x = x.reshape(batch_size, nr_of_atoms, 128)
        return x


class SchNetRepresentation(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_filters: int,
        n_interactions: int,
        cutoff: float = 5.0,
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
        cutoff: float, optional
            Cutoff value for the pairlist. Default is 5.0.
        """
        super().__init__()

        self.interactions = nn.ModuleList(
            [
                SchNetInteractionBlock(n_atom_basis, n_filters)
                for _ in range(n_interactions)
            ]
        )
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=cutoff)

    def forward(
        self, x: torch.Tensor, pairlist: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
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
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Output tensor after forward pass.
        """
        atom_index12 = pairlist["atom_index12"]
        d_ij = pairlist["d_ij"]

        f_ij, rcut_ij = _distance_to_radial_basis(d_ij, self.radial_basis)

        idx_i, idx_j = atom_index12[0], atom_index12[1]
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        return x


class LighningSchnet(Schnet, LighningModuleMixin):
    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_filters: int = 0,
        cutoff: float = 5.0,
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the SchNet model.

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
        loss : nn.Module, optional
            Loss function to use. Default is nn.MSELoss.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to use. Default is torch.optim.Adam.
        lr : float, optional
            Learning rate. Default is 1e-3.
        """

        super().__init__(n_atom_basis, n_interactions, n_filters, cutoff)
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

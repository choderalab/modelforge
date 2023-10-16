from typing import Dict, Type

import torch
import torch.nn as nn
from loguru import logger

from .models import BaseNNP, LightningModuleMixin
from .utils import _distance_to_radial_basis


class SchNET(BaseNNP):
    def __init__(
        self,
        nr_atom_basis: int,
        nr_interactions: int,
        nr_filters: int = 0,
        cutoff: float = 5.0,
        nr_of_embeddings: int = 100,
    ) -> None:
        """
        Initialize the SchNet class.

        Parameters
        ----------
        nr_atom_basis : int
            Number of atom basis; defines the dimensionality of the output features.
        nr_interactions : int
            Number of interaction blocks in the architecture.
        nr_filters : int, optional
            Number of filters; defines the dimensionality of the intermediate features (default is 0).
        cutoff : float, optional
            Cutoff value for the pairlist (default is 5.0 Angstrom).
        nr_of_embeddings: int, optional
            Number of embeddings (default is 100).
        """
        from .models import PairList  # Local import to avoid circular dependencies

        super().__init__()
        from .utils import EnergyReadout

        self.calculate_distances_and_pairlist = PairList(cutoff)
        self.representation = SchNETRepresentation(cutoff)
        self.readout = EnergyReadout(nr_atom_basis)
        self.embedding = nn.Embedding(nr_of_embeddings, nr_atom_basis, padding_idx=0)
        self.interactions = nn.ModuleList(
            [
                SchNETInteractionBlock(nr_atom_basis, nr_filters)
                for _ in range(nr_interactions)
            ]
        )

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

        # Initialize the feature representation using atomic numbers
        Z = inputs["Z"]
        mask = Z != 0
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])
        x = self.embedding(Z)  # shape (batch_size, n_atoms, n_atom_basis)

        # Compute the representation for each atom
        representation = self.representation(
            pairlist
        )  # shape (batch_size, n_atoms, n_atom_basis)

        # unpad

        # Iterate over interaction blocks to update features
        for interaction in self.interactions:
            v = interaction(
                x,
                representation["f_ij"],
                representation["idx_i"],
                representation["idx_j"],
                representation["rcut_ij"],
            )
            x = x + v

        # Pool over atoms to get molecular energies
        return self.readout(x)  # shape (batch_size,)


class SchNETInteractionBlock(nn.Module):
    def __init__(self, nr_atom_basis: int, nr_filters: int, nr_rbf: int = 20) -> None:
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        nr_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        nr_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        nr_rbf : int, optional
            Number of radial basis functions. Default is 20.
        """
        super().__init__()
        from .utils import ShiftedSoftplus, sequential_block

        # Initialize parameters
        self.nr_atom_basis = nr_atom_basis
        # Initialize layers
        self.intput_to_feature = nn.Linear(nr_atom_basis, nr_filters)
        self.feature_to_output = sequential_block(
            nr_filters, nr_atom_basis, ShiftedSoftplus
        )
        self.filter_network = sequential_block(nr_rbf, nr_filters, ShiftedSoftplus)
        self.nr_rbf = nr_rbf

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

        # Map input features to the filter space
        x = self.intput_to_feature(x)
        x = x.flatten(0, 1)  # shape (batch_size * n_atoms, nr_filters)

        # Generate interaction filters based on radial basis functions
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]  # Apply the cutoff
        Wij = Wij.to(dtype=x.dtype)

        # Perform continuous-filter convolution
        x_j = x[idx_j]  # Gather features of second atoms in each pair
        x_ij = x_j * Wij  # shape (n_pairs, nr_filters)

        # Initialize a tensor to gather the results
        shape = list(x.shape)  # note that we're using x.shape, not x_ij.shape
        x_native = torch.zeros(shape, dtype=x.dtype)

        # Prepare indices for scatter_add operation
        idx_i_expanded = idx_i.unsqueeze(1).expand_as(x_ij)

        # Sum contributions to update atom features
        x_native.scatter_add_(0, idx_i_expanded, x_ij)

        # Map back to the original feature space and reshape
        x = self.feature_to_output(x_native)
        x = x.reshape(batch_size, nr_of_atoms, self.nr_atom_basis)
        return x


class SchNETRepresentation(nn.Module):
    def __init__(self, cutoff: float = 5.0, n_rbf: int = 20):
        """
        Initialize the SchNet representation layer.

        Parameters
        ----------
        cutoff: float, optional
            Cutoff value for the pairlist. Default is 5.0.
        """
        from .utils import GaussianRBF

        super().__init__()

        self.radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    def forward(self, pairlist: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
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

        # Convert distances to radial basis functions
        f_ij, rcut_ij = _distance_to_radial_basis(d_ij, self.radial_basis)

        # Separate indices for atoms in each pair
        idx_i, idx_j = atom_index12[0], atom_index12[1]

        return {"f_ij": f_ij, "idx_i": idx_i, "idx_j": idx_j, "rcut_ij": rcut_ij}


class LightningSchNET(SchNET, LightningModuleMixin):
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

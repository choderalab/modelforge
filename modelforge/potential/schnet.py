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
        from .utils import EnergyReadout

        super().__init__(nr_of_embeddings, nr_atom_basis, cutoff)

        # Initialize representation, readout, and interaction layers
        self.representation = SchNETRepresentation(cutoff)
        self.readout = EnergyReadout(nr_atom_basis)
        self.interactions = nn.ModuleList(
            [
                SchNETInteractionBlock(nr_atom_basis, nr_filters)
                for _ in range(nr_interactions)
            ]
        )

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        atomic_numbers_embedding : torch.Tensor
            Atomic numbers embedding; shape (nr_systems, n_atoms, n_atom_basis).
        inputs : Dict[str, torch.Tensor]
        - pairlist:  shape (n_pairs, 2)
        - r_ij:  shape (n_pairs, 1)
        - d_ij:  shape (n_pairs, 3)
        - positions:  shape (nr_systems, n_atoms, 3)
        - atomic_numbers_embedding:  shape (nr_systems, n_atoms, n_atom_basis)


        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # Compute the representation for each atom
        representation = self.representation(
            inputs["d_ij"]
        )  # shape (n_pairs, n_atom_basis)

        x = inputs["atomic_numbers_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interactions:
            v = interaction(
                x,
                inputs["pair_indices"],
                representation["f_ij"],
                representation["rcut_ij"],
            )
            x = x + v  # Update atomic features

        # Pool over atoms to get molecular energies
        return self.readout(
            x, inputs["atomic_subsystem_indices"]
        )  # shape (batch_size,)


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

        self.nr_atom_basis = nr_atom_basis  # Initialize parameters
        self.intput_to_feature = nn.Linear(nr_atom_basis, nr_filters)
        self.feature_to_output = sequential_block(
            nr_filters, nr_atom_basis, ShiftedSoftplus
        )
        self.filter_network = sequential_block(nr_rbf, nr_filters, ShiftedSoftplus)
        self.nr_rbf = nr_rbf

    def forward(
        self,
        x: torch.Tensor,
        pairlist: torch.Tensor,  # shape [n_pairs, 2]
        f_ij: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_systems, nr_atoms, nr_atom_basis]
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

        # Map input features to the filter space
        x = self.intput_to_feature(x)
        # x = x.flatten(0, 1)  # shape (batch_size * n_atoms, nr_filters)

        # Generate interaction filters based on radial basis functions
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]  # Apply the cutoff
        Wij = Wij.to(dtype=x.dtype)

        idx_i, idx_j = pairlist[0], pairlist[1]

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

    def forward(self, d_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
        d_ij : Dict[str, torch.Tensor], Pairwise distances between atoms; shape [n_pairs]

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'f_ij': Radial basis functions for pairs of atoms; shape [n_pairs, n_rbf]
            - 'rcut_ij': Cutoff values for each pair; shape [n_pairs]
        """

        # Convert distances to radial basis functions
        f_ij, rcut_ij = _distance_to_radial_basis(d_ij, self.radial_basis)

        return {"f_ij": f_ij, "rcut_ij": rcut_ij}


class LightningSchNET(SchNET, LightningModuleMixin):
    def __init__(
        self,
        nr_atom_basis: int,
        nr_interactions: int,
        nr_filters: int = 0,
        cutoff: float = 5.0,
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        """
        PyTorch Lightning version of the SchNet model.

        Parameters
        ----------
        nr_atom_basis : int
            Dimensionality of the output features.
        nr_interactions : int
            Number of interaction blocks in the architecture.
        nr_filters : int, optional
            Dimensionality of the intermediate features (default is 0).
        cutoff : float, optional
            Cutoff value for the pairlist (default is 5.0).
        loss : Type[nn.Module], optional
            Loss function to use (default is nn.MSELoss).
        optimizer : Type[torch.optim.Optimizer], optional
            Optimizer to use (default is torch.optim.Adam).
        lr : float, optional
            Learning rate (default is 1e-3).
        """

        super().__init__(nr_atom_basis, nr_interactions, nr_filters, cutoff)
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

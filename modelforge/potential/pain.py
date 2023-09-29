import torch.nn as nn
from loguru import logger
from typing import Dict, Type, Tuple, Callable, Optional

from .models import BaseNNP, PairList, LighningModuleMixin
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    cosine_cutoff,
    scatter_add,
    sequential_block,
)
import torch
import torch.nn.functional as F
from .models import PairList
from modelforge.potential.utils import _distance_to_radial_basis, GaussianRBF
from torch.nn import SiLU


class PaiNN(BaseNNP):
    """PaiNN - polarizable interaction neural network

    References:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_rbf: int,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = SiLU,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """

        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.share_filters = shared_filters
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        if shared_filters:
            self.filter_net = sequential_block(n_rbf, 3 * n_atom_basis)
        else:
            self.filter_net = sequential_block(
                n_rbf, self.n_interactions * 3 * n_atom_basis
            )

        self.interactions = nn.ModuleList(
            PaiNNInteraction(n_atom_basis, activation=activation)
            for _ in range(n_interactions)
        )

        self.mixing = nn.ModuleList(
            PaiNNMixing(n_atom_basis, activation=activation, epsilon=epsilon)
            for _ in range(n_interactions)
        )

        self.calculate_distances_and_pairlist = PairList(self.cutoff)
        self.radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=float(self.cutoff))

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # calculate pairlist
        Z = inputs["Z"]
        mask = Z == -1
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])

        # extract properties from pairlist
        d_ij = pairlist["d_ij"]
        r_ij = pairlist["r_ij"]
        idx_i, idx_j = pairlist["atom_index12"][0], pairlist["atom_index12"][1]
        n_atoms = Z.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij

        f_ij, rcut_ij = _distance_to_radial_basis(d_ij, self.radial_basis)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(f_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(Z)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(
                q,
                mu,
                filter_list[i],
                dir_ij,
                idx_i,
                idx_j,
                Z.shape[0],
            )
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu
        return inputs


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis

        self.intra_atomic_net = nn.Sequential(
            sequential_block(n_atom_basis, n_atom_basis, activation_fct=activation),
            sequential_block(n_atom_basis, 3 * n_atom_basis),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        Wij: torch.Tensor,
        dir_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.intra_atomic_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis

        self.intra_atomic_net = nn.Sequential(
            sequential_block(2 * n_atom_basis, n_atom_basis, activation_fct=activation),
            sequential_block(n_atom_basis, 3 * n_atom_basis),
        )
        self.mu_channel_mix = sequential_block(
            n_atom_basis, 2 * n_atom_basis, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intratomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intra_atomic_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class LighningPaiNN(PaiNN, LighningModuleMixin):
    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_rbf: int,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = SiLU,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the PaiNN model."""

        super().__init__(
            n_atom_basis,
            n_interactions,
            n_rbf,
            cutoff_fn,
            activation,
            max_z,
            shared_interactions,
            shared_filters,
            epsilon,
        )
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

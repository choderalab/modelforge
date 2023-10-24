import torch.nn as nn
from loguru import logger
from typing import Dict, Type, Callable, Optional

from .models import BaseNNP, LightningModuleMixin
from .utils import (
    GaussianRBF,
    scatter_add,
    sequential_block,
)
import torch
import torch.nn.functional as F
from modelforge.potential.utils import _distance_to_radial_basis
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
        nr_of_embeddings: int = 100,
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

        from .utils import GaussianRBF

        super().__init__(nr_of_embeddings, n_atom_basis)

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = nn.Sequential(
                nn.Linear(n_rbf, 3 * n_atom_basis), nn.Identity()
            )
        else:
            self.filter_net = nn.Sequential(
                nn.Linear(n_rbf, self.n_interactions * 3 * n_atom_basis), nn.Identity()
            )
        self.interactions = nn.ModuleList(
            PaiNNInteraction(n_atom_basis, activation=activation)
            for _ in range(n_interactions)
        )
        self.mixing = nn.ModuleList(
            PaiNNMixing(n_atom_basis, activation=activation, epsilon=epsilon)
            for _ in range(n_interactions)
        )
        self.radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=float(self.cutoff))

    def _forward(
        self,
        pairlist: Dict[str, torch.Tensor],
        atomic_numbers_embedding: torch.Tensor,
        n_atoms: int,
    ):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # extract properties from pairlist
        d_ij = pairlist["d_ij"].unsqueeze(-1)  # n_pairs, 1
        r_ij = pairlist["r_ij"]
        # compute atom and pair features
        dir_ij = r_ij / d_ij

        f_ij, _ = _distance_to_radial_basis(d_ij, self.radial_basis)
        fcut = self.cutoff_fn(d_ij)  # n_pairs, 1

        filters = self.filter_net(f_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        qs = atomic_numbers_embedding.shape
        mu = torch.zeros(
            (qs[0] * qs[1], 3, qs[2]), device=atomic_numbers_embedding.device
        )  # nr_systems, 3, n_atom_basis

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            atomic_numbers_embedding, mu = interaction(
                atomic_numbers_embedding,
                mu,
                filter_list[i],
                dir_ij,
                pairlist,
                n_atoms,
            )
            atomic_numbers_embedding, mu = mixing(atomic_numbers_embedding, mu)

        atomic_numbers_embedding = atomic_numbers_embedding.squeeze(1)
        return {
            "scalar_representation": atomic_numbers_embedding,
            "vector_representation": mu,
        }


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
            sequential_block(n_atom_basis, n_atom_basis, activation),
            sequential_block(n_atom_basis, 3 * n_atom_basis),
        )

    def forward(
        self,
        atomic_numbers_embedding: torch.Tensor,  # shape [n_mols, n_atoms, n_atom_basis]
        mu: torch.Tensor,  # shape [n_mols, n_interactions, n_atom_basis]
        Wij: torch.Tensor,  # shape [n_interactions]
        dir_ij: torch.Tensor,
        pairlist: Dict[str, torch.Tensor],
        n_atoms: int,  # nr of unmasked atoms
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter

        Returns:
            atom features after interaction
        """
        import schnetpack.nn as snn

        # inter-atomic
        idx_i, idx_j = pairlist["pairlist"][0], pairlist["pairlist"][1]

        x = self.intra_atomic_net(atomic_numbers_embedding)
        n, m, k = atomic_numbers_embedding.shape  # [nr_systems,n_atoms,96]
        x = x.reshape(n * m, 1, 96)  # in: [nr_systems,n_atoms,96]; out:

        xj = x[idx_j]
        mu = mu  # [nr_systems*3*32] [6144]
        muj = mu[idx_j]
        x = Wij * xj
        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)

        ##############
        # Preparing for native scatter_add_
        # Expand the dimensions of idx_i to match that of dq
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(dq)

        # Create a zero tensor with appropriate shape
        dq_result_native = torch.zeros(n * m, 1, 32, dtype=dq.dtype, device=dq.device)

        # Use scatter_add_
        dq_result_native.scatter_add_(0, expanded_idx_i, dq)
        dq_result_custom = snn.scatter_add(dq, idx_i, dim_size=n * m)

        # The outputs should be the same
        assert torch.allclose(dq_result_custom, dq_result_native)

        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu_result_native = torch.zeros(
            n * m, 3, 32, dtype=dmu.dtype, device=dmu.device
        )
        expanded_idx_i_dmu = idx_i.view(-1, 1, 1).expand_as(dmu)
        dmu_result_native.scatter_add_(0, expanded_idx_i_dmu, dmu)
        dmu_results_custom = snn.scatter_add(dmu, idx_i, dim_size=n * m)
        assert torch.allclose(dmu_results_custom, dmu_result_native)

        q = x + dq_result_native.view(n, m)
        mu = mu + dmu_result_native

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


class LighningPaiNN(PaiNN, LightningModuleMixin):
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

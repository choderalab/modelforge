import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP, LightningModuleMixin
from .utils import (
    sequential_block,
)
import torch
from torch.nn import SiLU


class PaiNN(BaseNNP):
    """PaiNN - polarizable interaction neural network

    References:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        embedding: nn.Module,
        nr_interaction_blocks: int,
        radial_basis: nn.Module,
        cutoff: nn.Module,
        activation: Optional[Callable] = SiLU,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Parameters
            ----------
            embedding : torch.Module, contains atomic species embedding. 
                Embedding dimensions also define self.n_atom_basis.
            nr_interaction_blocks : int
                Number of interaction blocks.
            rbf : torch.Module
                radial basis functions.
            cutoff : torch.Module
                Cutoff function for the radial basis.
            activation : Callable, optional
                Activation function to use.
            shared_interactions : bool, optional
                Whether to share weights across interaction blocks (default is False).
            shared_filters : bool, optional
                Whether to share weights across filter-generating networks (default is False).
            epsilon : float, optional
                Stability constant to prevent numerical instabilities (default is 1e-8).

            References
            ----------
            Equivariant message passing for the prediction of tensorial properties and molecular spectra.
            ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
        """
        from .utils import EnergyReadout

        super().__init__(embedding)

        self.n_atom_basis = embedding.embedding_dim
        self.nr_interaction_blocks = nr_interaction_blocks
        self.radial_basis = radial_basis
        self.cutoff = cutoff
        self.share_filters = shared_filters
        self.readout = EnergyReadout(self.n_atom_basis)

        if shared_filters:
            self.filter_net = nn.Sequential(
                nn.Linear(self.radial_basis.n_rbf, 3 * self.n_atom_basis), nn.Identity()
            )
        else:
            self.filter_net = nn.Sequential(
                nn.Linear(
                    self.radial_basis.n_rbf,
                    self.nr_interaction_blocks * 3 * self.n_atom_basis,
                ),
                nn.Identity(),
            )
        self.interactions = nn.ModuleList(
            PaiNNInteraction(self.n_atom_basis, activation=activation)
            for _ in range(self.nr_interaction_blocks)
        )
        self.mixing = nn.ModuleList(
            PaiNNMixing(self.n_atom_basis, activation=activation, epsilon=epsilon)
            for _ in range(self.nr_interaction_blocks)
        )
        self.radial_basis = radial_basis

    def _forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        input : Dict[str, torch.Tensor]
            Dictionary containing pairlist information.
        atomic_numbers_embedding : torch.Tensor
            Tensor containing atomic number embeddings.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector representations.
        """
        from modelforge.potential.utils import _distance_to_radial_basis

        # extract properties from pairlist
        d_ij = inputs["d_ij"].unsqueeze(-1)  # n_pairs, 1
        r_ij = inputs["r_ij"]
        atomic_numbers_embedding = inputs["atomic_numbers_embedding"]
        qs = atomic_numbers_embedding.shape

        q = atomic_numbers_embedding.reshape(qs[0], 1, qs[1])
        # compute atom and pair features
        dir_ij = r_ij / d_ij
        # torch.Size([1150, 1, 32])
        f_ij, _ = _distance_to_radial_basis(d_ij, self.radial_basis)
        fcut = self.cutoff(d_ij)  # n_pairs, 1

        filters = self.filter_net(f_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.nr_interaction_blocks
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        mu = torch.zeros(
            (qs[0], 3, qs[1]), device=q.device
        )  # nr_of_systems * nr_of_atoms, 3, n_atom_basis

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(
                q,
                mu,
                filter_list[i],
                dir_ij,
                inputs["pair_indices"],
            )
            q, mu = mixing(q, mu)

        atomic_numbers_embedding = atomic_numbers_embedding.squeeze(1)

        # Use squeeze to remove dimensions of size 1
        q_ = q.squeeze(dim=1)

        _r = {
            "scalar_representation": q_,
            "vector_representation": mu,
        }
        return self.readout(
            _r["scalar_representation"], inputs["atomic_subsystem_indices"]
        )


class PaiNNInteraction(nn.Module):
    """
    PaiNN Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

    """

    def __init__(self, nr_atom_basis: int, activation: Callable):
        """
        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation : Callable
            Activation function to use.

        Attributes
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        intra_atomic_net : nn.Sequential
            Neural network for intra-atomic interactions.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # Initialize the intra-atomic neural network
        self.intra_atomic_net = nn.Sequential(
            sequential_block(nr_atom_basis, nr_atom_basis, activation),
            sequential_block(nr_atom_basis, 3 * nr_atom_basis),
        )

    def forward(
        self,
        q: torch.Tensor,  # shape [n_mols, n_atoms, n_atom_basis]
        mu: torch.Tensor,  # shape [n_mols, n_interactions, n_atom_basis]
        Wij: torch.Tensor,  # shape [n_interactions]
        dir_ij: torch.Tensor,
        pairlist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction output.

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values of shape [nr_of_atoms_in_systems, n_atom_basis].
        mu : torch.Tensor
            Vector input values of shape [n_mols, n_interactions, n_atom_basis].
        Wij : torch.Tensor
            Filter of shape [n_interactions].
        dir_ij : torch.Tensor
            Directional vector between atoms i and j.
        pairlist : torch.Tensor, shape (2, n_pairs)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        # inter-atomic
        idx_i, idx_j = pairlist[0], pairlist[1]

        x = self.intra_atomic_net(q)
        nr_of_atoms_in_all_systems, _, _ = q.shape  # [nr_systems,n_atoms,96]
        x = x.reshape(
            nr_of_atoms_in_all_systems, 1, 3 * self.nr_atom_basis
        )  # in: [nr_of_atoms_in_all_systems,96]; out:

        xj = x[idx_j]
        mu = mu  # [nr_systems*3*32] [6144]
        muj = mu[idx_j]
        x = Wij * xj
        dq, dmuR, dmumu = torch.split(x, self.nr_atom_basis, dim=-1)

        ##############
        # Preparing for native scatter_add_
        # Expand the dimensions of idx_i to match that of dq
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(dq)

        # Create a zero tensor with appropriate shape
        dq_result_native = torch.zeros(
            nr_of_atoms_in_all_systems,
            1,
            self.nr_atom_basis,
            dtype=dq.dtype,
            device=dq.device,
        )

        # Use scatter_add_
        dq_result_native.scatter_add_(0, expanded_idx_i, dq)
        # dq_result_custom = snn.scatter_add(
        #    dq, idx_i, dim_size=nr_of_atoms_in_all_systems
        # )

        # The outputs should be the same
        # assert torch.allclose(dq_result_custom, dq_result_native)

        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu_result_native = torch.zeros(
            nr_of_atoms_in_all_systems,
            3,
            self.nr_atom_basis,
            dtype=dmu.dtype,
            device=dmu.device,
        )
        expanded_idx_i_dmu = idx_i.view(-1, 1, 1).expand_as(dmu)
        dmu_result_native.scatter_add_(0, expanded_idx_i_dmu, dmu)
        # dmu_results_custom = snn.scatter_add(
        #    dmu, idx_i, dim_size=nr_of_atoms_in_all_systems
        # )
        # assert torch.allclose(dmu_results_custom, dmu_result_native)

        q = q + dq_result_native  # .view(nr_of_atoms_in_all_systems)
        mu = mu + dmu_result_native

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, nr_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        n_atom_basis : int
            Number of features to describe atomic environments.
        activation : Callable
            Activation function to use.
        epsilon : float, optional
            Stability constant added in norm to prevent numerical instabilities. Default is 1e-8.

        Attributes
        ----------
        n_atom_basis : int
            Number of features to describe atomic environments.
        intra_atomic_net : nn.Sequential
            Neural network for intra-atomic interactions.
        mu_channel_mix : nn.Sequential
            Neural network for mixing mu channels.
        epsilon : float
            Stability constant for numerical stability.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # initialize the intra-atomic neural network
        self.intra_atomic_net = nn.Sequential(
            sequential_block(
                2 * nr_atom_basis, nr_atom_basis, activation_fct=activation
            ),
            sequential_block(nr_atom_basis, 3 * nr_atom_basis),
        )
        # initialize the mu channel mixing network
        self.mu_channel_mix = sequential_block(
            nr_atom_basis, 2 * nr_atom_basis, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """
        compute intratomic mixing

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values.
        mu : torch.Tensor
            Vector input values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.nr_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intra_atomic_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.nr_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class LighningPaiNN(PaiNN, LightningModuleMixin):
    def __init__(
        self,
        embedding: nn.Module,
        nr_interaction_blocks: int,
        radial_basis: nn.Module,
        cutoff: nn.Module,
        activation: Optional[Callable] = SiLU,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the PaiNN model."""

        super().__init__(
            embedding=embedding,
            nr_interaction_blocks=nr_interaction_blocks,
            radial_basis=radial_basis,
            cutoff=cutoff,
            activation=activation,
            shared_interactions=shared_interactions,
            shared_filters=shared_filters,
            epsilon=epsilon,
        )
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

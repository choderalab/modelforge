from torch._tensor import Tensor
import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP, LightningModuleMixin
from .utils import Dense, scatter_softmax
import torch
import torch.nn.functional as F


class SAKE(BaseNNP):
    """SAKE - spatial attention kinetic networks with E(n) equivariance.

    References:
        https://openreview.net/pdf?id=3DIpIf3wQMC

    """

    def __init__(
            self,
            embedding_module: nn.Module,
            nr_interaction_blocks: int,
            radial_basis_module: nn.Module,
            cutoff_module: nn.Module,
            activation: Optional[Callable] = F.silu,
            epsilon: float = 1e-8
    ):
        """
        Parameters
            ----------
            embedding_module : torch.Module, contains atomic species embedding.
                Embedding dimensions also define self.nr_atom_basis.
            nr_interaction_blocks : int
                Number of interaction blocks.
            radial_basis_module : torch.Module
                radial basis functions.
            cutoff_module : torch.Module
                Cutoff function for the radial basis.
            activation : Callable, optional
                Activation function to use.
            epsilon : float, optional
                Stability constant to prevent numerical instabilities (default is 1e-8).
        """
        from .utils import EnergyReadout

        log.debug("Initializing SAKE model.")
        super().__init__(cutoff=cutoff_module.cutoff)
        self.nr_interaction_blocks = nr_interaction_blocks
        self.cutoff_module = cutoff_module
        self.radial_basis_module = radial_basis_module

        # initialize the energy readout
        self.nr_atom_basis = embedding_module.embedding_dim
        self.readout_module = EnergyReadout(self.nr_atom_basis)

        log.debug(
            f"Passed parameters to constructor: {self.nr_atom_basis=}, {nr_interaction_blocks=}, {cutoff_module=}"
        )
        log.debug(f"Initialized embedding: {embedding_module=}")

        # initialize the interaction networks
        self.interaction_modules = nn.ModuleList(
            SAKEInteraction(self.nr_atom_basis, activation=activation, radial_basis_module=self.radial_basis_module,
                            cutoff_module=self.cutoff_module, epsilon=epsilon)
            for _ in range(self.nr_interaction_blocks)
        )

        # save the embedding
        self.embedding_module = embedding_module

    def _readout(self, input: Dict[str, Tensor]):
        return self.readout_module(input)

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        inputs = self._prepare_inputs(inputs)
        return self._model_specific_input_preparation(inputs)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        from modelforge.potential.utils import embed_atom_features

        atomic_embedding = embed_atom_features(
            inputs["atomic_numbers"], self.embedding_module
        )
        inputs["atomic_embedding"] = atomic_embedding
        return inputs

    def _forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        inputs: Dict[str, torch.Tensor]
            Dictionary containing pairlist information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector representations.
        """

        # extract properties from pairlist
        h = inputs["atomic_embedding"]
        x = inputs["positions"]
        v = torch.zeros_like(x)
        for i, interaction_mod in enumerate(self.interaction_modules):
            h, x, v = interaction_mod(
                h,
                x,
                v,
                inputs["pair_indices"]
            )

        # Use squeeze to remove dimensions of size 1
        h = h.squeeze(dim=1)

        return {
            "scalar_representation": h,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


class SAKEInteraction(nn.Module):
    """
    SAKE Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

    """

    def __init__(self, nr_atom_basis: int, activation: Callable, radial_basis_module: nn.Module,
                 cutoff_module: nn.Module, epsilon: float, nr_heads: int = 7):
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
        """
        super().__init__()
        self.nr_atom_basis_in = nr_atom_basis
        self.nr_edge_basis = nr_atom_basis
        self.nr_edge_basis_hidden = nr_atom_basis
        self.nr_atom_basis_hidden = nr_atom_basis
        self.nr_atom_basis_out = nr_atom_basis
        self.nr_atom_basis_post_norm_hidden = nr_atom_basis
        self.nr_atom_basis_post_norm_out = nr_atom_basis
        self.nr_atom_basis_velocity = nr_atom_basis
        self.nr_heads = nr_heads
        self.nr_coefficients = nr_heads * self.nr_edge_basis
        self.epsilon = epsilon

        self.cutoff_module = cutoff_module

        self.node_mlp = nn.Sequential(
            Dense(self.nr_atom_basis_in + self.nr_coefficients + self.nr_edge_basis,
                  self.nr_atom_basis_hidden, activation=activation),
            Dense(self.nr_atom_basis_hidden, self.nr_atom_basis_out, activation=activation)
        )

        self.post_norm_mlp = nn.Sequential(
            Dense(self.nr_coefficients, self.nr_atom_basis_post_norm_hidden, activation=activation),
            Dense(self.nr_atom_basis_post_norm_hidden, self.nr_atom_basis_post_norm_out, activation=activation)
        )

        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.nr_atom_basis_in * 2,
                                                                       self.nr_edge_basis_hidden,
                                                                       self.nr_edge_basis,
                                                                       radial_basis_module,
                                                                       activation)

        self.semantic_attention_mlp = nn.Sequential(
            Dense(self.nr_edge_basis, self.nr_heads, activation=nn.CELU(alpha=2.0))
        )

        self.velocity_mlp = nn.Sequential(
            Dense(self.nr_atom_basis_in, self.nr_atom_basis_velocity, activation=activation),
            Dense(self.nr_atom_basis_velocity, 1, activation=lambda x: 2.0 * F.sigmoid(x), bias=False)
        )

        self.x_mixing_mlp = nn.Sequential(
            Dense(self.nr_coefficients, self.nr_coefficients, bias=False, activation=nn.Tanh()),
        )

        self.v_mixing = Dense(self.nr_coefficients, 1, bias=False)

    def update_node(self, h, h_i_semantic, h_i_spatial):
        return h + self.node_mlp(torch.cat([h, h_i_semantic, h_i_spatial], dim=-1))

    def update_velocity(self, v, h, combinations, idx_i, nr_atoms):
        v_ij = self.v_mixing(combinations.swapaxes(-1, -2)).squeeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(v_ij)
        dv = torch.zeros_like(v).scatter_reduce(0, expanded_idx_i, v_ij, "mean", include_self=False)
        return self.velocity_mlp(h) * v + dv

    def get_combinations(self, h_ij, dir_ij):
        return torch.einsum("px,pc->pcx", dir_ij, self.x_mixing_mlp(h_ij))

    def get_spatial_attention(self, combinations, idx_i, dir_ij, nr_atoms):
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(combinations)
        out_shape = (nr_atoms, self.nr_coefficients, dir_ij.shape[-1])
        zeros = torch.zeros(out_shape, dtype=combinations.dtype, device=combinations.device)
        combinations_mean = zeros.scatter_reduce(0, expanded_idx_i, combinations, "mean", include_self=False)
        combinations_norm_square = (combinations_mean ** 2).sum(dim=-1)
        return self.post_norm_mlp(combinations_norm_square)

    def aggregate(self, h_ij_semantic, idx_i, nr_atoms):
        expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_semantic)
        out_shape = (nr_atoms, self.nr_coefficients)
        zeros = torch.zeros(out_shape, dtype=h_ij_semantic.dtype, device=h_ij_semantic.device)
        return zeros.scatter_add(0, expanded_idx_i, h_ij_semantic)

    def get_semantic_attention(self, h_ij_edge, idx_i, d_ij, nr_atoms):
        h_ij_att_weights = self.semantic_attention_mlp(h_ij_edge)
        d_ij_att_weights = self.cutoff_module(d_ij)
        combined_ij_att_weights = torch.einsum("ph,p->ph", h_ij_att_weights, d_ij_att_weights)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(combined_ij_att_weights)
        combined_ij_att = scatter_softmax(combined_ij_att_weights, expanded_idx_i, dim=-2, dim_size=nr_atoms)
        return torch.reshape(torch.einsum("pf,ph->pfh", h_ij_edge, combined_ij_att),
                             (len(idx_i), self.nr_coefficients))

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            v: torch.Tensor,
            pairlist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interaction output.

        Parameters
        ----------
        h : torch.Tensor
            Scalar input values of shape [nr_of_atoms_in_systems, nr_atom_basis].
        x : torch.Tensor
            Equivariant position input values of shape [nr_of_atoms_in_systems, geometry_basis].
        v : torch.Tensor
            Equivariant velocity input values of shape [nr_of_atoms_in_systems, geometry_basis].
        pairlist : torch.Tensor, shape (2, nr_pairs)
        epsilon : float
            Stability constant to prevent numerical instabilities (default is 1e-8).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        idx_i, idx_j = pairlist[0], pairlist[1]
        nr_of_atoms_in_all_systems, _ = x.shape
        r_ij = x[idx_j] - x[idx_i]
        d_ij = torch.norm(r_ij, dim=-1)
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + self.epsilon)

        h_ij_edge = self.edge_model(h[idx_i], h[idx_j], d_ij)
        h_ij_semantic = self.get_semantic_attention(h_ij_edge, idx_i, d_ij, nr_of_atoms_in_all_systems)
        h_i_semantic = self.aggregate(h_ij_semantic, idx_i, nr_of_atoms_in_all_systems)
        combinations = self.get_combinations(h_ij_semantic, dir_ij)
        h_i_spatial = self.get_spatial_attention(combinations, idx_i, dir_ij, nr_of_atoms_in_all_systems)
        h_updated = self.update_node(h, h_i_semantic, h_i_spatial)
        v_updated = self.update_velocity(v, h, combinations, idx_i, nr_of_atoms_in_all_systems)
        x_updated = x + v_updated

        return h_updated, x_updated, v_updated


class LightningSAKE(SAKE, LightningModuleMixin):
    def __init__(
            self,
            embedding: nn.Module,
            nr_interaction_blocks: int,
            radial_basis: nn.Module,
            cutoff: nn.Module,
            activation: Optional[Callable] = F.silu,
            loss: Type[nn.Module] = nn.MSELoss(),
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the SAKE model."""

        super().__init__(
            embedding_module=embedding,
            nr_interaction_blocks=nr_interaction_blocks,
            radial_basis_module=radial_basis,
            cutoff_module=cutoff,
            activation=activation,
        )
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr


class ContinuousFilterConvolutionWithConcatenation(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, radial_basis_module, activation):
        super().__init__()
        self.rbf = radial_basis_module
        self.mlp_in = nn.Linear(in_features, radial_basis_module.n_rbf)
        self.mlp_out = nn.Sequential(
            Dense(in_features + radial_basis_module.n_rbf + 1, hidden_features, activation=activation),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, h_i, h_j, d_ij):
        h_ij_cat = torch.cat([h_i, h_j], dim=-1)
        h_ij_filtered = self.rbf(d_ij) * self.mlp_in(h_ij_cat)

        h_ij_out = self.mlp_out(
            torch.cat(
                [h_ij_cat, h_ij_filtered, d_ij.unsqueeze(-1)],
                dim=1
            )
        )

        return h_ij_out

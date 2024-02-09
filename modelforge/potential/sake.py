from torch._tensor import Tensor
import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP, LightningModuleMixin
from .utils import Dense, scatter_softmax, broadcast
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
            SAKEInteraction(self.nr_atom_basis, activation=activation, radial_basis_module=self.radial_basis_module)
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
        q = inputs["atomic_embedding"]
        mu = inputs["positions"]
        for i, interaction_mod in enumerate(self.interaction_modules):
            q, mu = interaction_mod(
                q,
                mu,
                inputs["pair_indices"]
            )

        # Use squeeze to remove dimensions of size 1
        q = q.squeeze(dim=1)

        return {
            "scalar_representation": q,
            "vector_representation": mu,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


class SAKEInteraction(nn.Module):
    """
    SAKE Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

    """

    def __init__(self, nr_atom_basis: int, activation: Callable, radial_basis_module: nn.Module, n_heads: int = 7):
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
        self.n_heads = n_heads
        self.n_coefficients = n_heads * self.nr_edge_basis

        self.node_mlp = nn.Sequential(
            Dense(self.nr_atom_basis_in + self.n_coefficients + self.nr_edge_basis,
                  self.nr_atom_basis_hidden, activation=activation),
            Dense(self.nr_atom_basis_hidden, self.nr_atom_basis_out, activation=activation)
        )

        self.post_norm_mlp = nn.Sequential(
            Dense(self.n_coefficients, self.nr_atom_basis_post_norm_hidden, activation=activation),
            Dense(self.nr_atom_basis_post_norm_hidden, self.nr_atom_basis_post_norm_out, activation=activation)
        )

        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.nr_atom_basis_in * 2,
                                                                       self.nr_edge_basis_hidden,
                                                                       self.nr_edge_basis,
                                                                       radial_basis_module,
                                                                       activation)
        
        self.semantic_attention_mlp = nn.Sequential(
            Dense(self.nr_edge_basis, self.n_heads, activation=nn.CELU(alpha=2.0))
        )
        
        self.x_mixing = nn.Sequential(
            Dense(self.n_coefficients, self.n_coefficients, bias=False, activation=nn.Tanh()),
        )

    def spatial_attention(self, q_ij_mtx, r_ij, d_ij, idx_j, n_atoms):
        # q_ij_mtx shape: (n_pairs,  n_coefficients)
        # coefficients shape: (n_pairs, n_coefficients)
        coefficients = self.x_mixing(q_ij_mtx)

        # d_ij shape: (n_pairs, 3)
        print("r_ij", r_ij.shape)
        print("d_ij", d_ij.shape)
        r_ij = torch.div(r_ij, (d_ij + 1e-5).unsqueeze(-1))

        # p: pair axis; x: position axis, c: coefficient axis
        combinations = torch.einsum("px,pc->pcx", r_ij, coefficients)
        broadcast_idx_j = broadcast(idx_j, torch.zeros(idx_j.shape[0], self.n_coefficients, 3), dim=0)
        out_shape = (n_atoms, self.n_coefficients, 3)
        combinations_sum = torch.zeros(out_shape).scatter_reduce(0,
                                                                 broadcast_idx_j,
                                                                 combinations,
                                                                 "mean",
                                                                 include_self=False
                                                                 )
        combinations_norm = (combinations_sum ** 2).sum(-1)
        q_combinations = self.post_norm_mlp(combinations_norm)
        return q_combinations

    def aggregate(self, q_ij_mtx, idx_j, n_atoms):
        broadcast_idx_j = broadcast(idx_j, torch.zeros(idx_j.shape[0], self.n_coefficients), dim=0)
        out_shape = (n_atoms, self.n_coefficients)
        return torch.zeros(out_shape).scatter_add(0, broadcast_idx_j, q_ij_mtx)

    def node_model(self, q, q_ij, q_combinations):
        print("q", q.shape)
        print("q_ij", q_ij.shape)
        print("q_combinations", q_combinations.shape)
        out = torch.cat([q, q_ij, q_combinations], dim=-1)
        out = self.node_mlp(out)
        out = q + out
        return out

    def semantic_attention(self, q_ij_mtx, idx_j, n_atoms):
        # att shape: (n_pairs, n_heads)
        att = self.semantic_attention_mlp(q_ij_mtx)
        semantic_attention = scatter_softmax(att, idx_j, dim=0, dim_size=n_atoms)
        return semantic_attention

    def combined_attention(self, d_ij, q_ij_mtx, idx_j, n_atoms):
        # semantic_attention shape: (n_pairs, n_heads)
        semantic_attention = self.semantic_attention(q_ij_mtx, idx_j, n_atoms)

        # combined_attention shape: (n_pairs, n_heads)
        combined_attention = semantic_attention
        # combined_attention_agg shape: (n_atoms, n_heads)
        print("combined_attention", combined_attention.shape)
        print("n_atoms", n_atoms)
        print("idx_j", idx_j.shape)
        broadcast_idx_j = broadcast(idx_j, torch.zeros(idx_j.shape[0], self.n_heads), dim=0)
        combined_attention_agg = torch.zeros(n_atoms, self.n_heads).scatter_add(0, broadcast_idx_j, combined_attention)
        combined_attention = combined_attention / combined_attention_agg[idx_j]

        return combined_attention
    def forward(
            self,
            q: torch.Tensor,  # shape [nr_of_atoms_in_batch, nr_atom_basis]
            mu: torch.Tensor,  # shape [n_mols, n_interactions, nr_atom_basis]
            pairlist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction output.

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values of shape [nr_of_atoms_in_systems, nr_atom_basis].
        mu : torch.Tensor
            Vector input values of shape [n_mols, n_interactions, nr_atom_basis].
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

        nr_of_atoms_in_all_systems, _ = q.shape

        # qi_cat_qj shape: (n_pairs, in_features * 2 [concatenated sender and receiver]) 
        qi_cat_qj = torch.cat([q[idx_i], q[idx_j]], dim=1)
        print("qi_cat_qj shape:", qi_cat_qj.shape)

        r_ij = q[idx_j] - q[idx_i]
        d_ij = torch.norm(r_ij, dim=-1)
        # q_ij_mtx shape: (n_pairs, nr_edge_basis)
        q_ij_mtx = self.edge_model(qi_cat_qj, d_ij)
        # combined_attention shape: (n_pairs, n_heads)
        combined_attention = self.combined_attention(d_ij, q_ij_mtx, idx_j, nr_of_atoms_in_all_systems)
        # p: pair axis; f: hidden feature axis; h: head axis
        q_ij_att = torch.einsum("pf,ph->pfh", q_ij_mtx, combined_attention)
        # q_ij_att shape before reshape: (n_pairs, hidden_features, n_heads)
        q_ij_att = torch.reshape(q_ij_att, q_ij_att.shape[:-2] + (-1,))
        # q_ij_att shape after reshape: (n_pairs, n_coefficients)
        q_combinations = self.spatial_attention(q_ij_att, r_ij, d_ij, idx_j, nr_of_atoms_in_all_systems)

        q_ij = self.aggregate(q_ij_att, idx_j, nr_of_atoms_in_all_systems)
        dq = self.node_model(q, q_ij, q_combinations)

        q = q + dq

        return q, mu


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

    def forward(self, qi_cat_qj, d_ij):
        q_ij = self.mlp_in(qi_cat_qj)
        q_ij_filtered = self.rbf(d_ij) * q_ij

        print("q_ij_filtered.shape", q_ij_filtered.shape)
        print("d_ij.shape", d_ij.shape)
        print("qi_cat_qj.shape", qi_cat_qj.shape)

        q_ij_out = self.mlp_out(
            torch.cat(
                [qi_cat_qj, q_ij_filtered, d_ij.unsqueeze(-1)],
                dim=1
            )
        )

        return q_ij_out

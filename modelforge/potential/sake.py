from torch._tensor import Tensor
import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP, LightningModuleMixin
from .postprocessing import PostprocessingPipeline, NoPostprocess
from .utils import Dense, scatter_softmax
import torch
import torch.nn.functional as F


class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

        self.out_features = self.num_rbf

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        return -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2


class SAKE(BaseNNP):
    """SAKE - spatial attention kinetic networks with E(n) equivariance.

    Reference:
        Wang, Yuanqing and Chodera, John D. ICLR 2023. https://openreview.net/pdf?id=3DIpIf3wQMC

    """

    def __init__(
            self,
            nr_atom_basis: int,
            nr_interaction_blocks: int,
            nr_heads: int,
            radial_basis_module: nn.Module = ExpNormalSmearing,
            cutoff_module: nn.Module = nn.Identity(),
            activation: Optional[Callable] = F.silu,
            epsilon: float = 1e-8
    ):
        """
        Parameters
            ----------
            nr_atom_basis : int
                Number of features in atomic embeddings. Must be at least the maximum atomic number.
            nr_interaction_blocks : int
                Number of interaction blocks.
            radial_basis_module : torch.Module
                radial basis functions.
            nr_heads: int
                Number of heads for spatial attention.
            cutoff_module : torch.Module
                Cutoff function for the radial basis.
            activation : Callable, optional
                Activation function to use.
            epsilon : float, optional
                Stability constant to prevent numerical instabilities (default is 1e-8).
        """
        from .utils import EnergyReadout

        log.debug("Initializing SAKE model.")
        super().__init__(cutoff=cutoff_module.cutoff, postprocessing=PostprocessingPipeline([NoPostprocess({})]))
        self.nr_interaction_blocks = nr_interaction_blocks
        self.nr_heads = nr_heads
        self.cutoff_module = cutoff_module
        self.radial_basis_module = radial_basis_module

        # initialize the energy readout
        self.nr_atom_basis = nr_atom_basis
        self.readout_module = EnergyReadout(self.nr_atom_basis)

        log.debug(
            f"Passed parameters to constructor: {self.nr_atom_basis=}, {nr_interaction_blocks=}, {cutoff_module=}"
        )

        # initialize the interaction networks
        self.interaction_modules = nn.ModuleList(
            SAKEInteraction(nr_atom_basis=self.nr_atom_basis,
                            nr_edge_basis=self.nr_atom_basis,
                            nr_edge_basis_hidden=self.nr_atom_basis,
                            nr_atom_basis_hidden=self.nr_atom_basis,
                            nr_atom_basis_spatial_hidden=self.nr_atom_basis,
                            nr_atom_basis_spatial=self.nr_atom_basis,
                            nr_atom_basis_velocity=self.nr_atom_basis,
                            nr_coefficients=(self.nr_heads * self.nr_atom_basis),
                            nr_heads=self.nr_heads,
                            activation=activation,
                            radial_basis_module=self.radial_basis_module,
                            cutoff_module=self.cutoff_module,
                            epsilon=epsilon)
            for _ in range(self.nr_interaction_blocks)
        )

    def _readout(self, input: Dict[str, Tensor]):
        return self.readout_module(input)

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        inputs = self._prepare_inputs(inputs)
        return self._model_specific_input_preparation(inputs)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        inputs["atomic_embedding"] = F.one_hot(inputs["atomic_numbers"].squeeze(-1), num_classes=self.nr_atom_basis).float()
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
    Spatial Attention Kinetic Networks Layer.

    Wang and Chodera (2023) Sec. 5 Algorithm 1.
    """

    def __init__(self,
                 nr_atom_basis: int,
                 nr_edge_basis: int,
                 nr_edge_basis_hidden: int,
                 nr_atom_basis_hidden: int,
                 nr_atom_basis_spatial_hidden: int,
                 nr_atom_basis_spatial: int,
                 nr_atom_basis_velocity: int,
                 nr_coefficients: int,
                 nr_heads: int,
                 activation: Callable,
                 radial_basis_module: nn.Module,
                 cutoff_module: nn.Module, epsilon: float):
        """
        Parameters
        ----------
        nr_atom_basis : int
            Number of features in semantic atomic embedding (h).
        nr_edge_basis : int
            Number of edge features after edge update.
        nr_edge_basis_hidden : int
            Number of edge features after hidden layer within edge update.
        nr_atom_basis_hidden : int
            Number of features after hidden layer within node update.
        nr_atom_basis_spatial_hidden : int
            Number of features after hidden layer within spatial attention.
        nr_atom_basis_spatial : int
            Number of features after spatial attention.
        nr_atom_basis_velocity : int
            Number of features after hidden layer within velocity update.
        nr_coefficients : int
            Number of coefficients for spatial attention.
        activation : Callable
            Activation function to use.

        Attributes
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis
        self.nr_edge_basis = nr_edge_basis
        self.nr_edge_basis_hidden = nr_edge_basis_hidden
        self.nr_atom_basis_hidden = nr_atom_basis_hidden
        self.nr_atom_basis_spatial_hidden = nr_atom_basis_spatial_hidden
        self.nr_atom_basis_spatial = nr_atom_basis_spatial
        self.nr_atom_basis_velocity = nr_atom_basis_velocity
        self.nr_coefficients = nr_coefficients
        self.nr_heads = nr_heads
        self.epsilon = epsilon
        self.radial_basis_module = radial_basis_module
        self.cutoff_module = cutoff_module

        self.node_mlp = nn.Sequential(
            Dense(self.nr_atom_basis + self.nr_heads * self.nr_edge_basis + self.nr_atom_basis_spatial,
                  self.nr_atom_basis_hidden, activation=activation),
            Dense(self.nr_atom_basis_hidden, self.nr_atom_basis, activation=activation)
        )

        self.post_norm_mlp = nn.Sequential(
            Dense(self.nr_coefficients, self.nr_atom_basis_spatial_hidden, activation=activation),
            Dense(self.nr_atom_basis_spatial_hidden, self.nr_atom_basis_spatial, activation=activation)
        )

        self.edge_mlp_in = nn.Linear(self.nr_atom_basis * 2, radial_basis_module.n_rbf)

        self.edge_mlp_out = nn.Sequential(
            Dense(self.nr_atom_basis * 2 + radial_basis_module.n_rbf + 1, self.nr_edge_basis_hidden,
                  activation=activation),
            nn.Linear(nr_edge_basis_hidden, nr_edge_basis),
        )

        self.semantic_attention_mlp = Dense(self.nr_edge_basis, self.nr_heads, activation=nn.CELU(alpha=2.0))

        self.velocity_mlp = nn.Sequential(
            Dense(self.nr_atom_basis, self.nr_atom_basis_velocity, activation=activation),
            Dense(self.nr_atom_basis_velocity, 1, activation=lambda x: 2.0 * F.sigmoid(x), bias=False)
        )

        self.x_mixing_mlp = Dense(self.nr_heads * self.nr_edge_basis, self.nr_coefficients, bias=False,
                                  activation=nn.Tanh())

        self.v_mixing = Dense(self.nr_coefficients, 1, bias=False)

    def update_edge(self, h_i_by_pair, h_j_by_pair, d_ij):
        """Compute intermediate edge features for semantic attention.

        Wang and Chodera (2023) Sec. 5 Eq. 7.

        Parameters
        ----------
        h_i_by_pair : torch.Tensor
            Node features of receivers, duplicated across pairs. Shape [nr_pairs, nr_atom_basis].
        h_j_by_pair : torch.Tensor
            Node features of senders, duplicated across pairs. Shape [nr_pairs, nr_atom_basis].
        d_ij : torch.Tensor
            Distance between senders and receivers. Shape [nr_pairs, ].

        Returns
        -------
        torch.Tensor
            Intermediate edge features. Shape [nr_pairs, nr_edge_basis].
        """
        h_ij_cat = torch.cat([h_i_by_pair, h_j_by_pair], dim=-1)
        h_ij_filtered = self.radial_basis_module(d_ij) * self.edge_mlp_in(h_ij_cat)
        return self.edge_mlp_out(
            torch.cat([h_ij_cat, h_ij_filtered, d_ij.unsqueeze(-1)], dim=-1)
        )

    def update_node(self, h, h_i_semantic, h_i_spatial):
        """Update node semantic features for the next layer.

        Wang and Chodera (2023) Sec. 2.2 Eq. 4.

        Parameters
        ----------
        h : torch.Tensor
            Input node semantic features. Shape [nr_of_atoms_in_systems, nr_atom_basis].
        h_i_semantic : torch.Tensor
            Node semantic attention. Shape [nr_atoms_in_systems, nr_heads * nr_edge_basis].
        h_i_spatial : torch.Tensor
            Node spatial attention. Shape [nr_atoms_in_systems, nr_atom_basis_spatial].

        Returns
        -------
        torch.Tensor
            Updated node features. Shape [nr_of_atoms_in_systems, nr_atom_basis].
        """

        return h + self.node_mlp(torch.cat([h, h_i_semantic, h_i_spatial], dim=-1))

    def update_velocity(self, v, h, combinations, idx_i):
        """Update node velocity features for the next layer.
        
        Wang and Chodera (2023) Sec. 5 Eq. 12.

        Parameters
        ----------
        v : torch.Tensor
            Input node velocity features. Shape [nr_of_atoms_in_systems, geometry_basis].
        h : torch.Tensor
            Input node semantic features. Shape [nr_of_atoms_in_systems, nr_atom_basis].
        combinations : torch.Tensor
            Linear combinations of mixed edge features. Shape [nr_pairs, nr_heads * nr_edge_basis].
        idx_i : torch.Tensor
            Indices of the receiver nodes. Shape [nr_pairs, ].

        Returns
        -------
        torch.Tensor
            Updated velocity features. Shape [nr_of_atoms_in_systems, geometry_basis].
        """
        v_ij = self.v_mixing(combinations.transpose(-1, -2)).squeeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(v_ij)
        dv = torch.zeros_like(v).scatter_reduce(0, expanded_idx_i, v_ij, "mean", include_self=False)
        return self.velocity_mlp(h) * v + dv

    def get_combinations(self, h_ij_semantic, dir_ij):
        """Compute linear combinations of mixed edge features.
        
        Summation term in Wang and Chodera (2023) Sec. 4 Eq. 6.

        Parameters
        ----------
        h_ij_semantic : torch.Tensor
            Edge semantic attention. Shape [nr_pairs, nr_heads * nr_edge_basis].
        dir_ij : torch.Tensor
            Normalized direction from receivers to senders. Shape [nr_pairs, geometry_basis].

        Returns
        -------
        torch.Tensor
            Linear combinations of mixed edge features. Shape [nr_pairs, nr_coefficients, geometry_basis].
        """
        return torch.einsum("px,pc->pcx", dir_ij, self.x_mixing_mlp(h_ij_semantic))

    def get_spatial_attention(self, combinations, idx_i, nr_atoms):
        """Compute spatial attention.

        Wang and Chodera (2023) Sec. 4 Eq. 6.

        Parameters
        ----------
        combinations : torch.Tensor
            Linear combinations of mixed edge features. Shape [nr_pairs, nr_coefficients, geometry_basis].
        idx_i : torch.Tensor
            Indices of the receiver nodes. Shape [nr_pairs, ].
        nr_atoms : in
            Number of atoms in all systems.

        Returns
        -------
        torch.Tensor
            Spatial attention. Shape [nr_atoms, nr_atom_basis_spatial].
        """
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(combinations)
        out_shape = (nr_atoms, self.nr_coefficients, len(idx_i))
        zeros = torch.zeros(out_shape, dtype=combinations.dtype, device=combinations.device)
        combinations_mean = zeros.scatter_reduce(0, expanded_idx_i, combinations, "mean", include_self=False)
        combinations_norm_square = (combinations_mean ** 2).sum(dim=-1)
        return self.post_norm_mlp(combinations_norm_square)

    def aggregate(self, h_ij_semantic, idx_i, nr_atoms):
        """Aggregate edge semantic attention over all senders connected to a receiver.

        Wang and Chodera (2023) Sec. 5 Algorithm 1,  step labelled "Neighborhood aggregation".

        Parameters
        ----------
        h_ij_semantic : torch.Tensor
            Edge semantic attention. Shape [nr_pairs, nr_heads * nr_edge_basis].
        idx_i : torch.Tensor
            Indices of the receiver nodes. Shape [nr_pairs, ].
        nr_atoms : int
            Number of atoms in all systems.

        Returns
        -------
        torch.Tensor
            Aggregated edge semantic attention. Shape [nr_atoms, nr_heads * nr_edge_basis].
        """
        expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_semantic)
        out_shape = (nr_atoms, self.nr_heads * self.nr_edge_basis)
        zeros = torch.zeros(out_shape, dtype=h_ij_semantic.dtype, device=h_ij_semantic.device)
        return zeros.scatter_add(0, expanded_idx_i, h_ij_semantic)

    def get_semantic_attention(self, h_ij_edge, idx_i, d_ij, nr_atoms):
        """Compute semantic attention. Softmax is over all senders connected to a receiver.

        Wang and Chodera (2023) Sec. 5 Eq. 9-10.

        Parameters
        ----------
        h_ij_edge : torch.Tensor
            Edge features. Shape [nr_pairs, nr_edge_basis].
        idx_i : torch.Tensor
            Indices of the receiver nodes. Shape [nr_pairs, ].
        d_ij : torch.Tensor
            Distance between senders and receivers. Shape [nr_pairs, ].
        nr_atoms : int
            Number of atoms in all systems.

        Returns
        -------
        torch.Tensor
            Semantic attention. Shape [nr_pairs, nr_heads * nr_edge_basis].
        """
        h_ij_att_weights = self.semantic_attention_mlp(h_ij_edge)
        d_ij_att_weights = self.cutoff_module(d_ij)
        combined_ij_att_weights = torch.einsum("ph,p->ph", h_ij_att_weights, d_ij_att_weights)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(combined_ij_att_weights)
        combined_ij_att = scatter_softmax(combined_ij_att_weights, expanded_idx_i, dim=-2, dim_size=nr_atoms)
        return torch.reshape(torch.einsum("pf,ph->pfh", h_ij_edge, combined_ij_att),
                             (len(idx_i), self.nr_edge_basis * self.nr_heads))

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            v: torch.Tensor,
            pairlist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute interaction layer output.

        Parameters
        ----------
        h : torch.Tensor
            Input semantic (invariant) atomic embeddings. Shape [nr_of_atoms_in_systems, nr_atom_basis].
        x : torch.Tensor
            Input position (equivariant) atomic embeddings. Shape [nr_of_atoms_in_systems, geometry_basis].
        v : torch.Tensor
            Input velocity (equivariant) atomic embeddings. Shape [nr_of_atoms_in_systems, geometry_basis].
        pairlist : torch.Tensor, shape (2, nr_pairs)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (h, x, v) with same shapes as input.
        """
        idx_i, idx_j = pairlist
        nr_of_atoms_in_all_systems, _ = x.shape
        r_ij = x[idx_j] - x[idx_i]
        d_ij = torch.norm(r_ij, dim=-1)
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + self.epsilon)

        h_ij_edge = self.update_edge(h[idx_i], h[idx_j], d_ij)
        h_ij_semantic = self.get_semantic_attention(h_ij_edge, idx_i, d_ij, nr_of_atoms_in_all_systems)
        h_i_semantic = self.aggregate(h_ij_semantic, idx_i, nr_of_atoms_in_all_systems)
        combinations = self.get_combinations(h_ij_semantic, dir_ij)
        h_i_spatial = self.get_spatial_attention(combinations, idx_i, nr_of_atoms_in_all_systems)
        h_updated = self.update_node(h, h_i_semantic, h_i_spatial)
        v_updated = self.update_velocity(v, h, combinations, idx_i)
        x_updated = x + v_updated

        return h_updated, x_updated, v_updated


class LightningSAKE(SAKE, LightningModuleMixin):
    def __init__(
            self,
            nr_atom_basis: int,
            nr_interaction_blocks: int,
            nr_heads: int,
            radial_basis: nn.Module,
            cutoff: nn.Module,
            activation: Optional[Callable] = F.silu,
            loss: Type[nn.Module] = nn.MSELoss(),
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the SAKE model."""

        super().__init__(
            nr_atom_basis=nr_atom_basis,
            nr_interaction_blocks=nr_interaction_blocks,
            nr_heads=nr_heads,
            radial_basis_module=radial_basis,
            cutoff_module=cutoff,
            activation=activation,
        )
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

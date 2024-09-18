"""
SAKE - Spatial Attention Kinetic Networks with E(n) Equivariance
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger as log

from .models import NNPInputTuple, PairlistData
from .utils import DenseWithCustomDist, PhysNetRadialBasisFunction, scatter_softmax


class MultiplySigmoid(nn.Module):
    """
    Custom activation module that multiplies the sigmoid output by a factor of 2.0.
    This module is compatible with TorchScript.
    """

    def __init__(self, factor: float = 2.0):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor * torch.sigmoid(x)


from typing import List, Tuple


class SAKECore(torch.nn.Module):

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_interaction_modules: int,
        number_of_spatial_attention_heads: int,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: float,
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[Tuple[str, str]],
        epsilon: float = 1e-8,
        potential_seed: int = -1,
    ):

        log.debug("Initializing the SAKE architecture.")
        super().__init__()

        self.activation_function = activation_function_parameter["activation_function"]

        self.nr_interaction_blocks = number_of_interaction_modules
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )
        self.nr_heads = number_of_spatial_attention_heads
        self.number_of_per_atom_features = number_of_per_atom_features
        # featurize the atomic input
        from modelforge.potential.utils import DenseWithCustomDist, FeaturizeInput

        self.featurize_input = FeaturizeInput(featurization)
        self.energy_layer = nn.Sequential(
            DenseWithCustomDist(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation_function=self.activation_function,
            ),
            DenseWithCustomDist(number_of_per_atom_features, 1),
        )
        # initialize the interaction networks
        self.interaction_modules = nn.ModuleList(
            SAKEInteraction(
                nr_atom_basis=number_of_per_atom_features,
                nr_edge_basis=number_of_per_atom_features,
                nr_edge_basis_hidden=number_of_per_atom_features,
                nr_atom_basis_hidden=number_of_per_atom_features,
                nr_atom_basis_spatial_hidden=number_of_per_atom_features,
                nr_atom_basis_spatial=number_of_per_atom_features,
                nr_atom_basis_velocity=number_of_per_atom_features,
                nr_coefficients=(self.nr_heads * number_of_per_atom_features),
                nr_heads=self.nr_heads,
                activation=self.activation_function,
                maximum_interaction_radius=maximum_interaction_radius,
                number_of_radial_basis_functions=number_of_radial_basis_functions,
                epsilon=epsilon,
                scale_factor=1.0,
            )
            for _ in range(self.nr_interaction_blocks)
        )

    def compute_properties(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute atomic properties.

        Parameters
        ----------
        data : SAKENeuralNetworkInput
            Input data for the SAKE neural network.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing per-atom energy predictions and atomic subsystem indices.
        """
        # extract properties from pairlist
        h = self.featurize_input(data)
        x = data.positions
        v = torch.zeros_like(x)

        for interaction_mod in self.interaction_modules:
            h, x, v = interaction_mod(h, x, v, pairlist_output.pair_indices)

        results = {
            "per_atom_scalar_representation": h,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the forward pass through the network.

        Parameters
        ----------
        data : NNPInput
            Contains input data for the batch obtained directly from the
            dataset, including atomic numbers, positions, and other relevant
            fields.
        pairlist_output : PairListOutputs
            Contains the indices for the selected pairs and their associated
            distances and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and other properties from the
            forward pass.
        """
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(data, pairlist_output)
        # add atomic numbers to the output
        outputs["atomic_numbers"] = data.atomic_numbers
        # FIXME:
        # Compute all specified outputs
        for output_name, output_layer in self.output_layers.items():
            results[output_name] = output_layer(h).squeeze(-1)

        return results


class SAKEInteraction(nn.Module):
    """
    Spatial Attention Kinetic Networks Layer.

    Wang and Chodera (2023) Sec. 5 Algorithm 1.
    """

    def __init__(
        self,
        nr_atom_basis: int,
        nr_edge_basis: int,
        nr_edge_basis_hidden: int,
        nr_atom_basis_hidden: int,
        nr_atom_basis_spatial_hidden: int,
        nr_atom_basis_spatial: int,
        nr_atom_basis_velocity: int,
        nr_coefficients: int,
        nr_heads: int,
        activation: nn.Module,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        epsilon: float,
        scale_factor: float,
    ):
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
        maximum_interaction_radius : unit.Quantity
            Distance parameter for setting scale factors in radial basis functions.
        number_of_radial_basis_functions: int
            Number of radial basis functions.
        epsilon : float
            Small constant to add for stability.
        scale_factor : unit.Quantity
            Factor with dimensions of length used to nondimensionalize distances before being
            passed into `edge_mlp_in`.
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
        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

        self.node_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis
                + self.nr_heads * self.nr_edge_basis
                + self.nr_atom_basis_spatial,
                self.nr_atom_basis_hidden,
                activation_function=activation,
            ),
            DenseWithCustomDist(
                self.nr_atom_basis_hidden,
                self.nr_atom_basis,
                activation_function=activation,
            ),
        )

        self.post_norm_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_coefficients,
                self.nr_atom_basis_spatial_hidden,
                activation_function=activation,
            ),
            DenseWithCustomDist(
                self.nr_atom_basis_spatial_hidden,
                self.nr_atom_basis_spatial,
                activation_function=activation,
            ),
        )

        self.edge_mlp_in = nn.Linear(
            self.nr_atom_basis * 2, number_of_radial_basis_functions
        )

        self.edge_mlp_out = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis * 2 + number_of_radial_basis_functions + 1,
                self.nr_edge_basis_hidden,
                activation_function=activation,
            ),
            nn.Linear(nr_edge_basis_hidden, nr_edge_basis),
        )

        self.semantic_attention_mlp = DenseWithCustomDist(
            self.nr_edge_basis, self.nr_heads, activation_function=nn.CELU(alpha=2.0)
        )

        self.velocity_mlp = nn.Sequential(
            DenseWithCustomDist(
                self.nr_atom_basis,
                self.nr_atom_basis_velocity,
                activation_function=activation,
            ),
            DenseWithCustomDist(
                self.nr_atom_basis_velocity,
                1,
                activation_function=MultiplySigmoid(factor=2.0),
                bias=False,
            ),
        )

        self.x_mixing_mlp = DenseWithCustomDist(
            self.nr_heads * self.nr_edge_basis,
            self.nr_coefficients,
            bias=False,
            activation_function=nn.Tanh(),
        )

        self.v_mixing_mlp = DenseWithCustomDist(self.nr_coefficients, 1, bias=False)

        self.scale_factor_in_nanometer = scale_factor

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
        h_ij_filtered = self.radial_symmetry_function_module(
            d_ij.unsqueeze(-1)
        ).squeeze(-2) * self.edge_mlp_in(h_ij_cat)
        return self.edge_mlp_out(
            torch.cat(
                [
                    h_ij_cat,
                    h_ij_filtered,
                    d_ij.unsqueeze(-1) / self.scale_factor_in_nanometer,
                ],
                dim=-1,
            )
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
        v_ij = self.v_mixing_mlp(combinations.transpose(-1, -2)).squeeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(v_ij)
        dv = torch.zeros_like(v).scatter_reduce(
            0, expanded_idx_i, v_ij, "mean", include_self=False
        )
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
        # p: nr_pairs, x: geometry_basis, c: nr_coefficients
        return torch.einsum("px,pc->pcx", dir_ij, self.x_mixing_mlp(h_ij_semantic))

    def get_spatial_attention(
        self, combinations: torch.Tensor, idx_i: torch.Tensor, nr_atoms: int
    ):
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
        out_shape = (nr_atoms, self.nr_coefficients, combinations.shape[-1])
        zeros = torch.zeros(
            out_shape, dtype=combinations.dtype, device=combinations.device
        )
        combinations_mean = zeros.scatter_reduce(
            0, expanded_idx_i, combinations, "mean", include_self=False
        )
        combinations_norm_square = (combinations_mean**2).sum(dim=-1)
        return self.post_norm_mlp(combinations_norm_square)

    def aggregate(
        self, h_ij_semantic: torch.Tensor, idx_i: torch.Tensor, nr_atoms: int
    ):
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
        zeros = torch.zeros(
            out_shape, dtype=h_ij_semantic.dtype, device=h_ij_semantic.device
        )
        return zeros.scatter_add(0, expanded_idx_i, h_ij_semantic)

    def get_semantic_attention(
        self,
        h_ij_edge: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        nr_atoms: int,
    ):
        """Compute semantic attention. Softmax is over all senders connected to a receiver.

        Wang and Chodera (2023) Sec. 5 Eq. 9-10.

        Parameters
        ----------
        h_ij_edge : torch.Tensor
            Edge features. Shape [nr_pairs, nr_edge_basis].
        idx_i : torch.Tensor
            Indices of the receiver nodes. Shape [nr_pairs, ].
        idx_j : torch.Tensor
            Indices of the sender nodes. Shape [nr_pairs, ].
        nr_atoms : int
            Number of atoms in all systems.

        Returns
        -------
        torch.Tensor
            Semantic attention. Shape [nr_pairs, nr_heads * nr_edge_basis].
        """
        h_ij_att_weights = self.semantic_attention_mlp(h_ij_edge) - (
            torch.eq(idx_i, idx_j) * 1e5
        ).unsqueeze(-1)
        expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_att_weights)
        combined_ij_att = scatter_softmax(
            h_ij_att_weights,
            expanded_idx_i,
            dim=0,
            dim_size=nr_atoms,
        )
        # p: nr_pairs, f: nr_edge_basis, h: nr_heads
        return torch.reshape(
            torch.einsum("pf,ph->pfh", h_ij_edge, combined_ij_att),
            (len(idx_i), self.nr_edge_basis * self.nr_heads),
        )

    def forward(
        self, h: torch.Tensor, x: torch.Tensor, v: torch.Tensor, pairlist: torch.Tensor
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
        idx_i, idx_j = pairlist.unbind(0)
        nr_of_atoms_in_all_systems = int(x.size(dim=0))
        r_ij = x[idx_j] - x[idx_i]
        d_ij = torch.sqrt((r_ij**2).sum(dim=1) + self.epsilon)
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + self.epsilon)

        h_ij_edge = self.update_edge(h[idx_j], h[idx_i], d_ij)
        h_ij_semantic = self.get_semantic_attention(
            h_ij_edge, idx_i, idx_j, nr_of_atoms_in_all_systems
        )
        del h_ij_edge
        h_i_semantic = self.aggregate(h_ij_semantic, idx_i, nr_of_atoms_in_all_systems)
        combinations = self.get_combinations(h_ij_semantic, dir_ij)
        del h_ij_semantic
        h_i_spatial = self.get_spatial_attention(
            combinations, idx_i, nr_of_atoms_in_all_systems
        )
        h_updated = self.update_node(h, h_i_semantic, h_i_spatial)
        del h, h_i_semantic, h_i_spatial
        v_updated = self.update_velocity(v, h_updated, combinations, idx_i)
        del v
        x_updated = x + v_updated

        return h_updated, x_updated, v_updated

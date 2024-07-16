from dataclasses import dataclass

import torch.nn as nn
from loguru import logger as log
from typing import Dict, Tuple
from openff.units import unit
from .models import InputPreparation, NNPInput, BaseNetwork, CoreNetwork

from .models import PairListOutputs
from .utils import (
    Dense,
    scatter_softmax,
    PhysNetRadialBasisFunction,
)
from modelforge.dataset.dataset import NNPInput
import torch
import torch.nn.functional as F


@dataclass
class SAKENeuralNetworkInput:
    """
    A dataclass designed to structure the inputs for SAKE neural network potentials, ensuring
    an efficient and structured representation of atomic systems for energy computation and
    property prediction within the SAKE framework.

    Attributes
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers for each atom in the system(s). Shape: [num_atoms].
    positions : torch.Tensor
        XYZ coordinates of each atom. Shape: [num_atoms, 3].
    atomic_subsystem_indices : torch.Tensor
        Maps each atom to its respective subsystem or molecule, useful for systems with multiple
        molecules. Shape: [num_atoms].
    pair_indices : torch.Tensor
        Indicates indices of atom pairs, essential for computing pairwise features. Shape: [2, num_pairs].
    number_of_atoms : int
        Total number of atoms in the batch, facilitating batch-wise operations.
    atomic_embedding : torch.Tensor
        Embeddings or features for each atom, potentially derived from atomic numbers or learned. Shape: [num_atoms, embedding_dim].

    Notes
    -----
    The `SAKENeuralNetworkInput` dataclass encapsulates essential inputs required by the SAKE neural network
    model for accurately predicting system energies and properties. It includes atomic positions, atomic types,
    and connectivity information, crucial for a detailed representation of atomistic systems.

    Examples
    --------
    >>> sake_input = SAKENeuralNetworkInput(
    ...     atomic_numbers=torch.tensor([1, 6, 6, 8]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ...     atomic_subsystem_indices=torch.tensor([0, 0, 0, 0]),
    ...     pair_indices=torch.tensor([[0, 1], [0, 2], [1, 2]]).T,
    ...     number_of_atoms=4,
    ...     atomic_embedding=torch.randn(4, 5)  # Example atomic embeddings
    ... )
    """

    pair_indices: torch.Tensor
    number_of_atoms: int
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    atomic_subsystem_indices: torch.Tensor
    atomic_embedding: torch.Tensor


class SAKECore(CoreNetwork):
    """SAKE - spatial attention kinetic networks with E(n) equivariance.

    Reference:
        Wang, Yuanqing and Chodera, John D. ICLR 2023. https://openreview.net/pdf?id=3DIpIf3wQMC

    """

    def __init__(
        self,
        max_Z: int = 100,
        number_of_atom_features: int = 64,
        number_of_interaction_modules: int = 6,
        number_of_spatial_attention_heads: int = 4,
        number_of_radial_basis_functions: int = 50,
        cutoff: unit.Quantity = 5.0 * unit.angstrom,
        epsilon: float = 1e-8,
    ):
        from .processing import FromAtomToMoleculeReduction

        log.debug("Initializing SAKE model.")
        super().__init__()
        self.nr_interaction_blocks = number_of_interaction_modules
        self.nr_heads = number_of_spatial_attention_heads
        self.max_Z = max_Z

        self.embedding = Dense(max_Z, number_of_atom_features)
        self.energy_layer = nn.Sequential(
            Dense(number_of_atom_features, number_of_atom_features),
            nn.SiLU(),
            Dense(number_of_atom_features, 1),
        )
        # initialize the interaction networks
        self.interaction_modules = nn.ModuleList(
            SAKEInteraction(
                nr_atom_basis=number_of_atom_features,
                nr_edge_basis=number_of_atom_features,
                nr_edge_basis_hidden=number_of_atom_features,
                nr_atom_basis_hidden=number_of_atom_features,
                nr_atom_basis_spatial_hidden=number_of_atom_features,
                nr_atom_basis_spatial=number_of_atom_features,
                nr_atom_basis_velocity=number_of_atom_features,
                nr_coefficients=(self.nr_heads * number_of_atom_features),
                nr_heads=self.nr_heads,
                activation=torch.nn.SiLU(),
                cutoff=cutoff,
                number_of_radial_basis_functions=number_of_radial_basis_functions,
                epsilon=epsilon,
            )
            for _ in range(self.nr_interaction_blocks)
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> SAKENeuralNetworkInput:
        # Perform atomic embedding

        number_of_atoms = data.atomic_numbers.shape[0]

        atomic_embedding = self.embedding(
            F.one_hot(data.atomic_numbers.long(), num_classes=self.max_Z).to(
                self.embedding.weight.dtype
            )
        )

        nnp_input = SAKENeuralNetworkInput(
            pair_indices=pairlist_output.pair_indices,
            number_of_atoms=number_of_atoms,
            positions=data.positions.to(self.embedding.weight.dtype),
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            atomic_embedding=atomic_embedding,
        )

        return nnp_input

    def compute_properties(self, data: SAKENeuralNetworkInput):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        data: SAKENeuralNetworkInput
            Dataclass containing atomic properties, embeddings, and pairlist.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing per-atom energy predictions and atomic subsystem indices.
        """

        # extract properties from pairlist
        h = data.atomic_embedding
        x = data.positions
        v = torch.zeros_like(x)

        for interaction_mod in self.interaction_modules:
            h, x, v = interaction_mod(h, x, v, data.pair_indices)

        # Use squeeze to remove dimensions of size 1
        E_i = self.energy_layer(h).squeeze(1)

        return {
            "per_atom_energy": E_i,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


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
        cutoff: float,
        number_of_radial_basis_functions: int,
        epsilon: float,
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
        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=cutoff,
            dtype=torch.float32,
        )

        self.node_mlp = nn.Sequential(
            Dense(
                self.nr_atom_basis
                + self.nr_heads * self.nr_edge_basis
                + self.nr_atom_basis_spatial,
                self.nr_atom_basis_hidden,
                activation=activation,
            ),
            Dense(self.nr_atom_basis_hidden, self.nr_atom_basis, activation=activation),
        )

        self.post_norm_mlp = nn.Sequential(
            Dense(
                self.nr_coefficients,
                self.nr_atom_basis_spatial_hidden,
                activation=activation,
            ),
            Dense(
                self.nr_atom_basis_spatial_hidden,
                self.nr_atom_basis_spatial,
                activation=activation,
            ),
        )

        self.edge_mlp_in = nn.Linear(
            self.nr_atom_basis * 2, number_of_radial_basis_functions
        )

        self.edge_mlp_out = nn.Sequential(
            Dense(
                self.nr_atom_basis * 2 + number_of_radial_basis_functions + 1,
                self.nr_edge_basis_hidden,
                activation=activation,
            ),
            nn.Linear(nr_edge_basis_hidden, nr_edge_basis),
        )

        self.semantic_attention_mlp = Dense(
            self.nr_edge_basis, self.nr_heads, activation=nn.CELU(alpha=2.0)
        )

        self.velocity_mlp = nn.Sequential(
            Dense(
                self.nr_atom_basis, self.nr_atom_basis_velocity, activation=activation
            ),
            Dense(
                self.nr_atom_basis_velocity,
                1,
                activation=lambda x: 2.0 * F.sigmoid(x),
                bias=False,
            ),
        )

        self.x_mixing_mlp = Dense(
            self.nr_heads * self.nr_edge_basis,
            self.nr_coefficients,
            bias=False,
            activation=nn.Tanh(),
        )

        self.v_mixing_mlp = Dense(self.nr_coefficients, 1, bias=False)

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
        h_ij_filtered = self.radial_symmetry_function_module(d_ij.unsqueeze(-1)).squeeze(-2) * self.edge_mlp_in(
            h_ij_cat
        )
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
        out_shape = (nr_atoms, self.nr_coefficients, combinations.shape[-1])
        zeros = torch.zeros(
            out_shape, dtype=combinations.dtype, device=combinations.device
        )
        combinations_mean = zeros.scatter_reduce(
            0, expanded_idx_i, combinations, "mean", include_self=False
        )
        combinations_norm_square = (combinations_mean**2).sum(dim=-1)
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
        zeros = torch.zeros(
            out_shape, dtype=h_ij_semantic.dtype, device=h_ij_semantic.device
        )
        return zeros.scatter_add(0, expanded_idx_i, h_ij_semantic)

    def get_semantic_attention(self, h_ij_edge, idx_i, idx_j, d_ij, nr_atoms):
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
        d_ij : torch.Tensor
            Distance between senders and receivers. Shape [nr_pairs, ].
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
            device=h_ij_edge.device,
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
        idx_i, idx_j = pairlist
        nr_of_atoms_in_all_systems, _ = x.shape
        r_ij = x[idx_j] - x[idx_i]
        d_ij = torch.sqrt((r_ij**2).sum(dim=1) + self.epsilon)
        dir_ij = r_ij / (d_ij.unsqueeze(-1) + self.epsilon)

        h_ij_edge = self.update_edge(h[idx_j], h[idx_i], d_ij)
        h_ij_semantic = self.get_semantic_attention(
            h_ij_edge, idx_i, idx_j, d_ij, nr_of_atoms_in_all_systems
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


from typing import Optional, List, Union


class SAKE(BaseNetwork):
    def __init__(
        self,
        max_Z: int,
        number_of_atom_features: int,
        number_of_interaction_modules: int,
        number_of_spatial_attention_heads: int,
        number_of_radial_basis_functions: int,
        cutoff: unit.Quantity,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
        )
        from modelforge.utils.units import _convert

        self.core_module = SAKECore(
            max_Z=max_Z,
            number_of_atom_features=number_of_atom_features,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_spatial_attention_heads=number_of_spatial_attention_heads,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            cutoff=_convert(cutoff),
            epsilon=epsilon,
        )

        self.only_unique_pairs = False  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=_convert(cutoff), only_unique_pairs=self.only_unique_pairs
        )

    def _config_prior(self):
        log.info("Configuring SAKE model hyperparameter prior distribution")
        from modelforge.utils.io import import_

        tune = import_("ray").tune
        # from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_modules": tune.randint(3, 8),
            "number_of_spatial_attention_heads": tune.randint(2, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
        }
        prior.update(shared_config_prior())
        return prior

    def combine_per_atom_properties(
        self, values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return values

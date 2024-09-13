"""
SAKE - Spatial Attention Kinetic Networks with E(n) Equivariance
"""

from dataclasses import dataclass

import torch.nn as nn
from loguru import logger as log
from typing import Dict, Tuple, Union, List, Type
from openff.units import unit
from .models import NNPInput, BaseNetwork, CoreNetwork, PairListOutputs
from .utils import (
    DenseWithCustomDist,
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
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_interaction_modules: int,
        number_of_spatial_attention_heads: int,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: unit.Quantity,
        activation_function: Type[torch.nn.Module],
        predicted_properties: List[Dict[str, str]],
        epsilon: float = 1e-8,
    ):
        """
        Initialize the SAKECore model.

        Parameters
        ----------
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for featurizing the atomic input.
        number_of_interaction_modules : int
            Number of interaction modules.
        number_of_spatial_attention_heads : int
            Number of spatial attention heads.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        maximum_interaction_radius : unit.Quantity
            Cutoff distance.
        activation_function : Type[torch.nn.Module]
            Activation function to use.
        epsilon : float, optional
            Small value to avoid division by zero, by default 1e-8.
        """
        log.debug("Initializing the SAKE architecture.")
        super().__init__(activation_function)
        self.nr_interaction_blocks = number_of_interaction_modules
        number_of_per_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )
        self.nr_heads = number_of_spatial_attention_heads
        self.number_of_per_atom_features = number_of_per_atom_features
        # featurize the atomic input
        from modelforge.potential.utils import FeaturizeInput, DenseWithCustomDist

        self.featurize_input = FeaturizeInput(featurization_config)

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
                scale_factor=(1.0 * unit.nanometer),  # TODO: switch to angstrom
            )
            for _ in range(self.nr_interaction_blocks)
        )

        # Initialize output layers based on configuration
        self.output_layers = nn.ModuleDict()
        for property in predicted_properties:
            output_name = property["name"]
            output_type = property["type"]
            output_dimension = (
                1 if output_type == "scalar" else 3
            )  # vector means 3D output

            self.output_layers[output_name] = nn.Sequential(
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                ),
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    output_dimension,
                ),
            )

    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist_output: Dict[str, PairListOutputs]
    ) -> SAKENeuralNetworkInput:
        """
        Prepare the model-specific input.

        Parameters
        ----------
        data : NNPInput
            Input data.
        pairlist_output : Dict[str,PairListOutputs]
            Pairlist output(s)

        Returns
        -------
        SAKENeuralNetworkInput
            Prepared input for the SAKE neural network.
        """
        # Perform atomic embedding

        number_of_atoms = data.atomic_numbers.shape[0]

        # Note, pairlist_output is a Dict where the key corresponds to the name of the cutoff parameter
        # e.g. "maximum_interaction_radius"

        pairlist_output = pairlist_output["maximum_interaction_radius"]

        nnp_input = SAKENeuralNetworkInput(
            pair_indices=pairlist_output.pair_indices,
            number_of_atoms=number_of_atoms,
            positions=data.positions,  # .to(self.embedding.weight.dtype),
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            atomic_embedding=self.featurize_input(data),
        )  # add per-atom properties and embedding,

        return nnp_input

    def compute_properties(self, data: SAKENeuralNetworkInput):
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
        h = data.atomic_embedding
        x = data.positions
        v = torch.zeros_like(x)

        for interaction_mod in self.interaction_modules:
            h, x, v = interaction_mod(h, x, v, data.pair_indices)

        results = {
            "per_atom_scalar_representation": h,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

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
        maximum_interaction_radius: unit.Quantity,
        number_of_radial_basis_functions: int,
        epsilon: float,
        scale_factor: unit.Quantity,
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
                activation_function=lambda x: 2.0 * F.sigmoid(x),
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

        self.scale_factor_in_nanometer = scale_factor.m_as(unit.nanometer)

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

    def get_semantic_attention(self, h_ij_edge, idx_i, idx_j, nr_atoms):
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


from typing import Optional, List, Union


class SAKE(BaseNetwork):
    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_interaction_modules: int,
        number_of_spatial_attention_heads: int,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: unit.Quantity,
        activation_function_parameter: Dict,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        predicted_properties: List[Dict[str, str]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-8,
        potential_seed: Optional[int] = None,
    ):
        from modelforge.utils.units import _convert_str_to_unit

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            potential_seed=potential_seed,
        )
        activation_function = activation_function_parameter["activation_function"]

        self.core_module = SAKECore(
            featurization_config=featurization,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_spatial_attention_heads=number_of_spatial_attention_heads,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            activation_function=activation_function,
            predicted_properties=predicted_properties,
            epsilon=epsilon,
        )

    def _config_prior(self):
        log.info("Configuring SAKE model hyperparameter prior distribution")
        from modelforge.utils.io import import_

        from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_modules": tune.randint(3, 8),
            "number_of_spatial_attention_heads": tune.randint(2, 5),
            "maximum_interaction_radius": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
        }
        prior.update(shared_config_prior())
        return prior

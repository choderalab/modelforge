"""
PaiNN - polarizable interaction neural network
"""

from typing import Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from .models import NNPInputTuple, PairlistData
from .utils import DenseWithCustomDist


class PaiNNCore(torch.nn.Module):

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: float,
        number_of_interaction_modules: int,
        shared_interactions: bool,
        shared_filters: bool,
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[str],
        predicted_dim: List[int],
        epsilon: float = 1e-8,
        potential_seed: int = -1,
    ):
        """
        Core PaiNN architecture for modeling polarizable molecular interactions.

        Parameters
        ----------
        featurization : Dict[str, Dict[str, int]]
            Configuration for atom featurization, including number of features
            per atom.
        number_of_radial_basis_functions : int
            Number of radial basis functions for the PaiNN representation.
        maximum_interaction_radius : float
            Maximum interaction radius for atom pairs.
        number_of_interaction_modules : int
            Number of interaction modules to apply.
        shared_interactions : bool
            Whether to share weights across all interaction modules.
        shared_filters : bool
            Whether to share filters across blocks.
        activation_function_parameter : Dict[str, str]
            Dictionary containing the activation function to use.
        predicted_properties : List[str]
            List of properties to predict.
        predicted_dim : List[int]
            List of dimensions for each predicted property.
        epsilon : float, optional
            Small constant for numerical stability (default is 1e-8).
        potential_seed : int, optional
            Seed for random number generation (default is -1).
        """

        from modelforge.utils.misc import seed_random_number

        if potential_seed != -1:
            seed_random_number(potential_seed)

        super().__init__()
        log.debug("Initializing the PaiNN architecture.")
        self.activation_function = activation_function_parameter["activation_function"]

        self.number_of_interaction_modules = number_of_interaction_modules

        # Featurize the atomic input
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )
        # initialize representation block
        self.representation_module = PaiNNRepresentation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            number_of_interaction_modules,
            number_of_per_atom_features,
            shared_filters,
            featurization_config=featurization,
        )

        # initialize the interaction and mixing networks
        if shared_interactions:
            self.message_function = nn.ModuleList(
                [
                    Message(
                        number_of_per_atom_features,
                        activation_function=self.activation_function,
                    )
                ]
                * number_of_interaction_modules
            )
        else:
            self.message_function = nn.ModuleList(
                [
                    Message(
                        number_of_per_atom_features,
                        activation_function=self.activation_function,
                    )
                    for _ in range(number_of_interaction_modules)
                ]
            )

        self.update_function = nn.ModuleList(
            [
                Update(
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                    epsilon=epsilon,
                )
                for _ in range(number_of_interaction_modules)
            ]
        )

        # Initialize output layers based on configuration
        self.output_layers = nn.ModuleDict()
        for property, dim in zip(predicted_properties, predicted_dim):
            self.output_layers[property] = nn.Sequential(
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function=self.activation_function,
                ),
                DenseWithCustomDist(
                    number_of_per_atom_features,
                    int(dim),
                ),
            )

    def compute_properties(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute atomic representations and embeddings using PaiNN.

        Parameters
        ----------
        data : NNPInputTuple
            The input data containing atomic numbers, positions, etc.
        pairlist_output : PairlistData
            The output from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector atomic representations.
        """
        # Compute filters, scalar features (q), and vector features (mu)
        transformed_input = self.representation_module(data, pairlist_output)

        filter_list = transformed_input["filters"]
        per_atom_scalar_feature = transformed_input["per_atom_scalar_feature"]
        per_atom_vector_feature = transformed_input["per_atom_vector_feature"]
        dir_ij = transformed_input["dir_ij"]

        # Apply interaction and mixing modules
        for i, (interaction_mod, mixing_mod) in enumerate(
            zip(self.message_function, self.update_function)
        ):
            per_atom_scalar_feature, per_atom_vector_feature = interaction_mod(
                per_atom_scalar_feature,
                per_atom_vector_feature,
                filter_list[i],
                dir_ij,
                pairlist_output.pair_indices,
            )
            per_atom_scalar_feature, per_atom_vector_feature = mixing_mod(
                per_atom_scalar_feature, per_atom_vector_feature
            )

        return {
            "per_atom_scalar_representation": per_atom_scalar_feature.squeeze(1),
            "per_atom_vector_representation": per_atom_vector_feature,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the PaiNN model.

        Parameters
        ----------
        data : NNPInputTuple
            Input data including atomic numbers, positions, etc.
        pairlist_output : PairlistData
            Pair indices and distances from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            The predicted properties from the forward pass.
        """
        # Compute properties using the core PaiNN modules
        results = self.compute_properties(data, pairlist_output)
        # Apply output layers to the atomic embedding
        atomic_embedding = results["per_atom_scalar_representation"]
        for output_name, output_layer in self.output_layers.items():
            results[output_name] = output_layer(atomic_embedding).squeeze(-1)

        return results


class PaiNNRepresentation(nn.Module):

    def __init__(
        self,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        nr_interaction_blocks: int,
        nr_atom_basis: int,
        shared_filters: bool,
        featurization_config: Dict[str, Union[List[str], int]],
    ):
        """
        PaiNN representation module for generating scalar and vector atomic embeddings.

        Parameters
        ----------
        maximum_interaction_radius : float
            Maximum interaction radius for atomic pairs in nanometer.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        nr_interaction_blocks : int
            Number of interaction blocks.
        nr_atom_basis : int
            Number of features to describe atomic environments.
        shared_filters : bool
            Whether to share filters across blocks.
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for atom featurization.
        """
        from modelforge.potential import CosineAttenuationFunction, FeaturizeInput

        from .representation import SchnetRadialBasisFunction

        super().__init__()

        self.featurize_input = FeaturizeInput(featurization_config)

        # Initialize the cutoff function and radial symmetry functions
        self.cutoff_module = CosineAttenuationFunction(maximum_interaction_radius)

        self.radial_symmetry_function_module = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

        # initialize the filter network
        if shared_filters:
            filter_net = DenseWithCustomDist(
                in_features=number_of_radial_basis_functions,
                out_features=3 * nr_atom_basis,
            )

        else:
            filter_net = DenseWithCustomDist(
                in_features=number_of_radial_basis_functions,
                out_features=nr_interaction_blocks * nr_atom_basis * 3,
            )

        self.filter_net = filter_net

        self.shared_filters = shared_filters
        self.nr_interaction_blocks = nr_interaction_blocks
        self.nr_atom_basis = nr_atom_basis

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Generate atomic embeddings and filters for PaiNN.

        Parameters
        ----------
        data : NNPInputTuple
            The input data containing atomic numbers, positions, etc.
        pairlist_output : PairlistData
            The output from the pairlist module, containing pair indices and distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the transformed input tensors.
        """
        # compute normalized pairwise distances
        d_ij = pairlist_output.d_ij
        dir_ij = pairlist_output.r_ij / d_ij

        # featurize pairwise distances using radial basis functions (RBF)
        f_ij = self.radial_symmetry_function_module(d_ij)
        f_ij_cut = self.cutoff_module(d_ij)
        # Apply the filter network and cutoff function
        filters = torch.mul(self.filter_net(f_ij), f_ij_cut)

        # depending on whether we share filters or not filters have different
        # shape at dim=1 (dim=0 is always the number of atom pairs) if we share
        # filters, we copy the filters and use the same filters for all blocks
        if self.shared_filters:
            filter_list = torch.stack([filters] * self.nr_interaction_blocks, dim=0)
        # otherwise we index into subset of the calculated filters and provide
        # each block with its own set of filters
        else:
            filter_list = torch.stack(
                torch.split(filters, 3 * self.nr_atom_basis, dim=-1), dim=0
            )

        # Initialize scalar and vector features
        per_atom_scalar_feature = self.featurize_input(data).unsqueeze(
            1
        )  # nr_of_atoms, 1, nr_atom_basis
        atomic_embedding_shape = per_atom_scalar_feature.shape
        per_atom_vector_feature = torch.zeros(
            (atomic_embedding_shape[0], 3, atomic_embedding_shape[2]),
            device=per_atom_scalar_feature.device,
            dtype=per_atom_scalar_feature.dtype,
        )  # nr_of_atoms, 3, nr_atom_basis

        return {
            "filters": filter_list,
            "dir_ij": dir_ij,
            "per_atom_scalar_feature": per_atom_scalar_feature,
            "per_atom_vector_feature": per_atom_vector_feature,
        }


class Message(nn.Module):

    def __init__(
        self,
        nr_atom_basis: int,
        activation_function: torch.nn.Module,
    ):
        """
        PaiNN message block for modeling scalar and vector interactions between atoms.

        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation_function : Type[torch.nn.Module]
            Activation function to use in the interaction block.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # Initialize the interatomic network
        self.interatomic_net = nn.Sequential(
            DenseWithCustomDist(
                nr_atom_basis, nr_atom_basis, activation_function=activation_function
            ),
            DenseWithCustomDist(nr_atom_basis, 3 * nr_atom_basis),
        )

    def forward(
        self,
        per_atom_scalar_representation: torch.Tensor,
        per_atom_vector_representation: torch.Tensor,
        W_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        pairlist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the message  block.

        Parameters
        ----------
        per_atom_scalar_representation : torch.Tensor
            Scalar input values (Shape: [nr_atoms, 1, nr_atom_basis]).
        per_atom_vector_representation : torch.Tensor
            Vector input values (Shape: [nr_atoms, 3, nr_atom_basis]).
        W_ij : torch.Tensor
            Interaction filters (Shape: [nr_pairs, 1, nr_interactions]).
        dir_ij : torch.Tensor
            Direction vectors between atoms i and j.
        pairlist : torch.Tensor

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        # perform the scalar operations (same as in SchNet)
        idx_i, idx_j = pairlist[0], pairlist[1]

        # Compute scalar interactions (q)
        transformed_per_atom_scalar_representation = self.interatomic_net(
            per_atom_scalar_representation
        )  # per atom
        s_j = transformed_per_atom_scalar_representation[idx_j]  # per pair
        weighted_s_j = W_ij.unsqueeze(1) * s_j  # per_pair

        # split the output into 3x per_pair_ds to exchange information between the scalar and vector outputs
        per_pair_ds1, per_pair_ds2, per_pair_ds3 = torch.split(
            weighted_s_j, self.nr_atom_basis, dim=-1
        )

        # Update scalar feature
        ds_i = torch.zeros_like(per_atom_scalar_representation)
        # Expand idx_i to match the shape of dq for scatter_add operation
        expanded_idx_i = idx_i.unsqueeze(-1).expand(-1, per_pair_ds1.size(2))
        ds_i.scatter_add_(0, expanded_idx_i.unsqueeze(1), per_pair_ds1)
        per_atom_scalar_representation = per_atom_scalar_representation + ds_i

        # ----------------- vector output -----------------
        # Compute vector interactions (dv_i)

        v_j = per_atom_vector_representation[idx_j]
        dmu_per_pair = per_pair_ds2 * dir_ij.unsqueeze(-1) + per_pair_ds3 * v_j
        # Create a tensor to store the result, matching the size of `mu`
        dv_i = torch.zeros_like(
            per_atom_vector_representation
        )  # Shape: (nr_of_atoms, 3, nr_atom_basis)
        # Expand idx_i to match the shape of dmu for scatter_add operation
        expanded_idx_i = (
            idx_i.unsqueeze(-1).unsqueeze(-1).expand(-1, dv_i.size(1), dv_i.size(2))
        )
        # Perform scatter_add_ operation
        dv_i.scatter_add_(0, expanded_idx_i, dmu_per_pair)

        per_atom_vector_representation = per_atom_vector_representation + dv_i

        return per_atom_scalar_representation, per_atom_vector_representation


class Update(nn.Module):

    def __init__(
        self,
        nr_atom_basis: int,
        activation_function: torch.nn.Module,
        epsilon: float = 1e-8,
    ):
        """
        PaiNN update block.

        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation_function : torch.nn.Module
            Activation function to use.
        epsilon : float, optional
            Stability constant added to prevent numerical instabilities (default is 1e-8).
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # initialize the intra-atomic neural network
        self.intra_atomic_net = nn.Sequential(
            DenseWithCustomDist(
                2 * nr_atom_basis,
                nr_atom_basis,
                activation_function=activation_function,
            ),
            DenseWithCustomDist(nr_atom_basis, 3 * nr_atom_basis),
        )
        # Initialize the channel mixing network for mu
        self.linear_transformation = DenseWithCustomDist(
            nr_atom_basis, 2 * nr_atom_basis, bias=False
        )
        self.epsilon = epsilon

    def forward(
        self, scalar_message: torch.Tensor, vector_message: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through node update.

        Parameters
        ----------
        scalar_message : torch.Tensor
            Scalar input values.
        vector_message : torch.Tensor
            Vector input values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations .
        """
        vector_meassge_transformed = self.linear_transformation(vector_message)

        v_V, v_U = torch.split(vector_meassge_transformed, self.nr_atom_basis, dim=-1)

        L2_norm_v_V = torch.sqrt(torch.sum(v_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([scalar_message, L2_norm_v_V], dim=-1)
        transformed_scalar_message = self.intra_atomic_net(ctx)

        a_ss, a_vv, a_sv = torch.split(
            transformed_scalar_message, self.nr_atom_basis, dim=-1
        )

        a_vv = a_vv * v_U
        a_sv = a_sv * torch.sum(v_V * v_U, dim=1, keepdim=True)

        scalar_message = scalar_message + a_ss + a_sv
        vector_message = vector_message + a_vv
        return scalar_message, vector_message

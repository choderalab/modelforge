import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from loguru import logger as log

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.aimnet2 import (
    AimNet2Core,
    AIMNet2Representation,
    AIMNet2InteractionModule,
    mlp_init,
    FukuiEquilibration,
)
from modelforge.potential.multipolenet import (
    MultipoleNetCore,
    MultipoleInteractionModule,
    MultipoleRepresentation,
    MultipoleReadout,
)
from modelforge.potential.neighbors import PairlistData


class AimNet2MPCore(torch.nn.Module):

    # per-atom properties computed by the core (charges are equilibrated);
    # readout heads must not overwrite them
    core_output_properties = frozenset(
        {"per_atom_scalar_representation", "per_atom_charge"}
    )

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_vector_features: int,
        number_of_interaction_modules: int,
        interaction_module_hidden_layers: List[List[int]],
        output_module_hidden_layers: List[int],
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[str],
        predicted_dim: List[int],
        maximum_interaction_radius: float,
        number_of_charge_channels: int,
        charge_equilibration_layer_size: int,
        number_of_radial_basis_functions_mp: int,
        number_of_interaction_modules_mp: int,
        number_of_monopole_dimensions: int,
        number_of_dipole_dimensions: int,
        number_of_quadrupole_dimensions: int,
        maximum_interaction_radius_mp: float,
        activation_function_parameter_mp: Dict[str, str],
        number_of_radial_basis_module_dimensions: int,
    ) -> None:
        """
        Core architecture of the AimNet2 model for molecular property
        prediction.

        Parameters
        ----------
        featurization : Dict[str, Dict[str, int]]
            Configuration dictionary specifying feature details for atomic
            embeddings.
        number_of_radial_basis_functions : int
            Number of radial basis functions used in the radial symmetry
            function.
        number_of_vector_features : int
            Number of vector features used in the model, which determines the
            dimensionality of the angular symmetry functions.
        number_of_interaction_modules : int
            Number of interaction modules in the model, determining the depth of
            message passing.
        interaction_module_hidden_layers : List[List[int]]
            List of hidden layer sizes for each interaction module. Note, this requires at least one value to be defined
            as this specifies the output size of the first layer in the interaction module and input size of the final layer
        output_module_hidden_layers : List[int]
            List of hidden layer sizes for the output modules, which are used to
            compute the final predictions from the atomic embeddings.
        activation_function_parameter : Dict[str, str]
            Configuration of activation functions used across the model.
        predicted_properties : List[str]
            List of properties that the model is predicting (e.g., energy,
            forces).
        predicted_dim : List[int]
            The dimensionality of each predicted property.
        maximum_interaction_radius : float
            The cutoff radius for atomic interactions in the model.
        number_of_charge_channels : int
            The number of charge channels in the model.
        charge_equilibration_layer_size:
            The size of hidden layers in the charge equilibration

        """

        super().__init__()

        log.debug("Initializing the AimNet2 architecture.")
        self.model_name = "aimnet2"
        self.activation_function = activation_function_parameter["activation_function"]

        self.number_of_charge_channels = number_of_charge_channels

        # Initialize representation block
        self.multipolenet_representation_module = MultipoleNetCore(
            number_of_radial_basis_functions=number_of_radial_basis_functions_mp,
            number_of_interaction_modules=number_of_interaction_modules_mp,
            number_of_monopole_dimensions=number_of_monopole_dimensions,
            number_of_dipole_dimensions=number_of_dipole_dimensions,
            number_of_quadrupole_dimensions=number_of_quadrupole_dimensions,
            maximum_interaction_radius=maximum_interaction_radius_mp,
            number_of_radial_basis_module_dimensions=number_of_radial_basis_module_dimensions,
            activation_function_parameter=activation_function_parameter_mp,
            readout_hidden_features=64,  # Not relevant when MultipoleNet is used as a representation module
        )
        self.aimnet2_representation_module = AIMNet2Representation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization,
        )
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )

        # scale by 1/sqrt(F_atom * G), the fan-in of the einsum contraction,
        # so vector features start on the same scale as radial features
        self.agh = nn.Parameter(
            torch.randn(
                number_of_per_atom_features,  # F_atom
                number_of_radial_basis_functions,  # G
                number_of_vector_features,  # H
            )
            / math.sqrt(number_of_per_atom_features * number_of_radial_basis_functions)
        )
        # shape(nr_of_angular_symmetry_functions,nr_of_radial_symmetry_functions,nr_of_vector_features)

        # Define interaction modules for message passing
        self.interaction_modules = torch.nn.ModuleList(
            [
                AIMNet2InteractionModule(
                    number_of_per_atom_features=number_of_per_atom_features,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    number_of_vector_features=number_of_vector_features,
                    number_of_charge_channels=self.number_of_charge_channels,
                    hidden_layers=interaction_module_hidden_layers[i],
                    activation_function=self.activation_function,
                    is_first_module=(i == 0),
                )
                for i in range(number_of_interaction_modules)
            ]
        )
        # Define output layers to calculate per-atom predictions
        overlap = self.core_output_properties.intersection(predicted_properties)
        if overlap:
            raise ValueError(
                f"predicted_properties {sorted(overlap)} are computed by the "
                "AimNet2 core (equilibrated charges) and will not be overwritten "
                "by a readout head. Remove them from predicted_properties; the "
                "core always returns them."
            )
        self.output_layers = nn.ModuleDict()
        for property, dim in zip(predicted_properties, predicted_dim):
            self.output_layers[property] = mlp_init(
                n_in_features=number_of_per_atom_features,
                n_out_features=int(dim),
                activation_function=self.activation_function,
                hidden_layers=output_module_hidden_layers,
            )

        self.charge_conservation = FukuiEquilibration(
            number_of_per_atom_features=number_of_per_atom_features,
            number_of_charge_channels=number_of_charge_channels,
            hidden_dim=charge_equilibration_layer_size,
        )

    def compute_properties(
        self,
        data: NNPInput,
        pairlist: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the requested properties for a given input batch.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist: PairlistData
            The output from the pairlist module.
        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom scalar representations and atomic subsystem
            indices.
        """

        rep = self.multipolenet_representation_module(data, pairlist)

        aimnet2_rep = self.aimnet2_representation_module(data, pairlist)
        rep["f_ij"] = aimnet2_rep["f_ij"]
        rep["f_cutoff"] = aimnet2_rep["f_cutoff"]

        atomic_embedding = rep["per_atom_scalar_representation"]
        r_ij, d_ij, f_ij, f_cutoff = (
            pairlist.r_ij,
            pairlist.d_ij,
            rep["f_ij"],
            rep["f_cutoff"],
        )
        # Scalar Gaussian expansion for radial terms
        gs = f_ij * f_cutoff  # Shape: (number_of_pairs, G)
        # Unit direction vectors
        u_ij = r_ij / d_ij
        # Compute gv with shape (number_of_pairs, 3, G)
        gv = u_ij.unsqueeze(-1) * gs.unsqueeze(1)  # Broadcasting over G

        # Atomic embedding "a" Eqn. (3)
        partial_charge_channels = torch.zeros(
            (atomic_embedding.shape[0], self.number_of_charge_channels),
            dtype=atomic_embedding.dtype,
            device=atomic_embedding.device,
        )

        per_system_total_charge = data.per_system_total_charge
        if per_system_total_charge.shape[0] == 0:
            n_systems = (
                int(data.atomic_subsystem_indices.max().item()) + 1
                if data.atomic_subsystem_indices.numel() > 0
                else 0
            )

            per_system_total_charge = torch.zeros(
                n_systems,
                1,
                device=atomic_embedding.device,
                dtype=atomic_embedding.dtype,
            )

        per_system_spin_multiplicity = data.per_system_spin_multiplicity
        if per_system_spin_multiplicity.shape[0] == 0:

            per_system_spin_multiplicity = torch.ones_like(per_system_total_charge)

        if self.number_of_charge_channels == 2:
            half_spin = 0.5 * (per_system_spin_multiplicity - 1)
            half_q = 0.5 * per_system_total_charge

            temp_sum = half_q + half_spin
            temp_diff = half_q - half_spin

            reference_charge = torch.stack(
                [temp_sum.squeeze(), temp_diff.squeeze()], dim=-1
            )
        else:
            reference_charge = per_system_total_charge

        # Perform message passing using interaction modules
        for i, interaction in enumerate(self.interaction_modules):

            delta_a, delta_q, f = interaction(
                atomic_embedding,
                partial_charge_channels,
                pairlist.pair_indices,
                gs,
                gv,
                self.agh,
            )

            # Update atomic embeddings
            atomic_embedding = atomic_embedding + delta_a

            # Apply scaling factor `f` to `delta_q`
            scaled_delta_q = f * delta_q

            # Update partial charges
            partial_charge_channels = scaled_delta_q  # Initialize charges

            partial_charge_channels = self.charge_conservation(
                atomic_embedding=atomic_embedding,
                per_atom_charge=partial_charge_channels,
                atomic_subsystem_indices=data.atomic_subsystem_indices.to(
                    dtype=torch.int64
                ),
                per_system_target=reference_charge.to(dtype=atomic_embedding.dtype),
            )

        # if we have two charge channels we need to sum them up to get partial charge
        if self.number_of_charge_channels == 2:
            partial_charges = partial_charge_channels.sum(dim=-1).unsqueeze(-1)
        else:
            partial_charges = partial_charge_channels
        # check that none of the tensors are NaN
        if torch.isnan(atomic_embedding).any():
            raise ValueError("NaN values detected in atomic embeddings.")
        if torch.isnan(partial_charges).any():
            raise ValueError("NaN values detected in partial charges.")

        if self.number_of_charge_channels == 1:
            return {
                "per_atom_scalar_representation": atomic_embedding,
                "atomic_subsystem_indices": data.atomic_subsystem_indices,
                "atomic_numbers": data.atomic_numbers,
                "per_atom_charge": partial_charges,
            }
        elif self.number_of_charge_channels == 2:
            return {
                "per_atom_scalar_representation": atomic_embedding,
                "atomic_subsystem_indices": data.atomic_subsystem_indices,
                "atomic_numbers": data.atomic_numbers,
                "per_atom_charge": partial_charges,
                "per_atom_charge_channels": partial_charge_channels,
            }

    def forward(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the forward pass through the network.

        Parameters
        ----------
        data : NNPInput
            Contains input data for the batch obtained directly from the
            dataset, including atomic numbers, positions, and other relevant
            fields.
        pairlist_output : PairListData
            Contains the indices for the selected pairs and their associated
            distances and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and other properties from the
            forward pass.
        """
        # perform the forward pass implemented in the subclass
        results = self.compute_properties(data, pairlist_output)
        atomic_embedding = results["per_atom_scalar_representation"]

        # The way this is setup, we will always return per_atom_charge and electrostatic_energy
        # as those come directly from the compute_properties method.

        # Compute all specified outputs
        for output_name, output_layer in self.output_layers.items():
            output = output_layer(atomic_embedding)
            results[output_name] = output

        return results

# class AimNet2MPCore(torch.nn.Module):
#     def __init__(
#             self,
#             featurization: Dict[str, Dict[str, int]],
#             number_of_radial_basis_functions: int,
#             number_of_vector_features: int,
#             number_of_interaction_modules: int,
#             interaction_module_hidden_layers: List[List[int]],
#             output_module_hidden_layers: List[int],
#             activation_function_parameter: Dict[str, str],
#             predicted_properties: List[str],
#             predicted_dim: List[int],
#             maximum_interaction_radius: float,
#             number_of_radial_basis_functions_mp: int,
#             number_of_interaction_modules_mp: int,
#             number_of_monopole_dimensions: int,
#             number_of_dipole_dimensions: int,
#             number_of_quadrupole_dimensions: int,
#             maximum_interaction_radius_mp: float,
#             activation_function_parameter_mp: Dict[str, str],
#             number_of_radial_basis_module_dimensions: int,
#     ) -> None:
#         """
#         Core architecture of the AimNet2 model for molecular property
#         prediction.
#
#         Parameters
#         ----------
#         featurization : Dict[str, Dict[str, int]]
#             Configuration dictionary specifying feature details for atomic
#             embeddings.
#         number_of_radial_basis_functions : int
#             Number of radial basis functions used in the radial symmetry
#             function.
#         number_of_vector_features : int
#             Number of vector features used in the model, which determines the
#             dimensionality of the angular symmetry functions.
#         number_of_interaction_modules : int
#             Number of interaction modules in the model, determining the depth of
#             message passing.
#         interaction_module_hidden_layers : List[List[int]]
#             List of hidden layer sizes for each interaction module. Note, this requires at least one value to be defined
#             as this specifies the output size of the first layer in the interaction module and input size of the final layer
#         output_module_hidden_layers : List[int]
#             List of hidden layer sizes for the output modules, which are used to
#             compute the final predictions from the atomic embeddings.
#         activation_function_parameter : Dict[str, str]
#             Configuration of activation functions used across the model.
#         predicted_properties : List[str]
#             List of properties that the model is predicting (e.g., energy,
#             forces).
#         predicted_dim : List[int]
#             The dimensionality of each predicted property.
#         maximum_interaction_radius : float
#             The cutoff radius for atomic interactions in the model.
#         """
#
#         super().__init__()
#
#         log.debug("Initializing the AimNet2 architecture with MultiPoleNet representation.")
#         self.model_name = "aimnet2_mp"
#
#         self.activation_function = activation_function_parameter["activation_function"]
#         self.number_of_charge_channels = number_of_charge_channels
#
#
#         # Initialize representation block
#         self.multipolenet_representation_module = MultipoleNetCore(
#             number_of_radial_basis_functions=number_of_radial_basis_functions_mp,
#             number_of_interaction_modules=number_of_interaction_modules_mp,
#             number_of_monopole_dimensions=number_of_monopole_dimensions,
#             number_of_dipole_dimensions=number_of_dipole_dimensions,
#             number_of_quadrupole_dimensions=number_of_quadrupole_dimensions,
#             maximum_interaction_radius=maximum_interaction_radius_mp,
#             number_of_radial_basis_module_dimensions=number_of_radial_basis_module_dimensions,
#             activation_function_parameter=activation_function_parameter_mp,
#             readout_hidden_features=64,  # Not relevant when MultipoleNet is used as a representation module
#         )
#         self.aimnet2_representation_module = AIMNet2Representation(
#             maximum_interaction_radius,
#             number_of_radial_basis_functions,
#             featurization_config=featurization,
#         )
#         number_of_per_atom_features = int(
#             featurization["atomic_number"]["number_of_per_atom_features"]
#         )
#
#         self.agh = nn.Parameter(
#             torch.randn(
#                 number_of_per_atom_features,  # F_atom
#                 number_of_radial_basis_functions,  # G
#                 number_of_vector_features,  # H
#             )
#         )
#         # shape(nr_of_angular_symmetry_functions,nr_of_radial_symmetry_functions,nr_of_vector_features)
#
#         # Define interaction modules for message passing
#         self.interaction_modules = torch.nn.ModuleList(
#             [
#                 AIMNet2InteractionModule(
#                     number_of_per_atom_features=number_of_per_atom_features,
#                     number_of_radial_basis_functions=number_of_radial_basis_functions,
#                     number_of_vector_features=number_of_vector_features,
#                     hidden_layers=interaction_module_hidden_layers[i],
#                     activation_function=self.activation_function,
#                     is_first_module=(i == 0),
#                 )
#                 for i in range(number_of_interaction_modules)
#             ]
#         )
#         # Define output layers to calculate per-atom predictions
#         self.output_layers = nn.ModuleDict()
#         for property, dim in zip(predicted_properties, predicted_dim):
#             self.output_layers[property] = mlp_init(
#                 n_in_features=number_of_per_atom_features,
#                 n_out_features=int(dim),
#                 activation_function=self.activation_function,
#                 hidden_layers=output_module_hidden_layers,
#             )
#
#         from modelforge.potential.processing import ChargeConservation
#
#         self.charge_conservation = ChargeConservation()
#
#         from modelforge.potential.processing import CoulombPotential
#
#         # note this will need to change when we switch to supporting multiple cutoffs
#         self.coulomb_potential = CoulombPotential(cutoff=maximum_interaction_radius)
#
#     def compute_properties(
#             self,
#             data: NNPInput,
#             pairlist: PairlistData,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Calculate the requested properties for a given input batch.
#
#         Parameters
#         ----------
#         data : NNPInput
#             The input data for the model.
#         pairlist: PairlistData
#             The output from the pairlist module.
#         Returns
#         -------
#         Dict[str, torch.Tensor]
#             The calculated per-atom scalar representations and atomic subsystem
#             indices.
#         """
#
#         rep = self.multipolenet_representation_module(data, pairlist)
#
#         aimnet2_rep = self.aimnet2_representation_module(data, pairlist)
#         rep["f_ij"] = aimnet2_rep["f_ij"]
#         rep["f_cutoff"] = aimnet2_rep["f_cutoff"]
#
#         atomic_embedding = rep["per_atom_scalar_representation"]
#         r_ij, d_ij, f_ij, f_cutoff = (
#             pairlist.r_ij,
#             pairlist.d_ij,
#             rep["f_ij"],
#             rep["f_cutoff"],
#         )
#         # Scalar Gaussian expansion for radial terms
#         gs = f_ij * f_cutoff  # Shape: (number_of_pairs, G)
#         # Unit direction vectors
#         u_ij = r_ij / d_ij
#         # Compute gv with shape (number_of_pairs, 3, G)
#         gv = u_ij.unsqueeze(-1) * gs.unsqueeze(1)  # Broadcasting over G
#
#         # Atomic embedding "a" Eqn. (3)
#         partial_charges = torch.zeros(
#             (atomic_embedding.shape[0], 1), device=atomic_embedding.device
#         )
#
#         # Perform message passing using interaction modules
#         for i, interaction in enumerate(self.interaction_modules):
#
#             delta_a, delta_q, f = interaction(
#                 atomic_embedding,
#                 partial_charges,
#                 pairlist.pair_indices,
#                 gs,
#                 gv,
#                 self.agh,
#             )
#
#             # Update atomic embeddings
#             atomic_embedding = atomic_embedding + delta_a
#
#             # Apply scaling factor `f` to `delta_q`
#             scaled_delta_q = f * delta_q
#
#             # Update partial charges
#             if i == 0:
#                 partial_charges = scaled_delta_q  # Initialize charges
#             else:
#                 partial_charges = partial_charges + scaled_delta_q  # Incremental update
#
#             partial_charges = self.charge_conservation(
#                 {
#                     "per_atom_charge": partial_charges,
#                     "per_system_total_charge": data.per_system_total_charge.to(
#                         dtype=torch.float32
#                     ),
#                     "atomic_subsystem_indices": data.atomic_subsystem_indices.to(
#                         dtype=torch.int64
#                     ),
#                 }
#             )["per_atom_charge"]
#
#         # check that none of the tensors are NaN
#         if torch.isnan(atomic_embedding).any():
#             raise ValueError("NaN values detected in atomic embeddings.")
#         if torch.isnan(partial_charges).any():
#             raise ValueError("NaN values detected in partial charges.")
#
#         return {
#             "per_atom_scalar_representation": atomic_embedding,
#             "atomic_subsystem_indices": data.atomic_subsystem_indices,
#             "atomic_numbers": data.atomic_numbers,
#             "per_atom_charge": partial_charges,
#         }
#
#     def forward(
#             self,
#             data: NNPInput,
#             pairlist_output: PairlistData,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Implements the forward pass through the network.
#
#         Parameters
#         ----------
#         data : NNPInput
#             Contains input data for the batch obtained directly from the
#             dataset, including atomic numbers, positions, and other relevant
#             fields.
#         pairlist_output : PairListOutputs
#             Contains the indices for the selected pairs and their associated
#             distances and displacement vectors.
#
#         Returns
#         -------
#         Dict[str, torch.Tensor]
#             The calculated per-atom properties and other properties from the
#             forward pass.
#         """
#         # perform the forward pass implemented in the subclass
#         results = self.compute_properties(data, pairlist_output)
#         atomic_embedding = results["per_atom_scalar_representation"]
#
#         # The way this is setup, we will always return per_atom_charge and electrostatic_energy
#         # as those come directly from the compute_properties method.
#
#         # Compute all specified outputs
#         for output_name, output_layer in self.output_layers.items():
#             output = output_layer(atomic_embedding)
#             results[output_name] = output
#
#         return results
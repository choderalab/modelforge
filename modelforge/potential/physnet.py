"""
Implementation of the PhysNet neural network potential.
"""

from typing import Dict

import torch
from loguru import logger as log
from torch import nn

from modelforge.utils.prop import NNPInput
from modelforge.potential.neighbors import PairlistData
from .utils import Dense


class PhysNetRepresentation(nn.Module):
    def __init__(
        self,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        """
        Representation module for PhysNet, generating radial basis functions
        (RBFs) and atomic embeddings with a cutoff for atomic interactions.

        Parameters
        ----------
        maximum_interaction_radius : float
            The cutoff distance for interactions.
        number_of_radial_basis_functions : int
            Number of radial basis functions to use.
        featurization_config : Dict[str, Dict[str, int]]
            Configuration for atomic feature generation.
        """

        super().__init__()

        # Initialize the cutoff function and radial basis function modules
        from modelforge.potential import (
            CosineAttenuationFunction,
            PhysNetRadialBasisFunction,
            FeaturizeInput,
        )

        self.cutoff_module = CosineAttenuationFunction(maximum_interaction_radius)
        self.featurize_input = FeaturizeInput(featurization_config)

        # Radial symmetry function using PhysNet radial basis expansion
        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the representation module, generating RBFs and
        atomic embeddings.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic positions, atomic numbers, etc.
        pairlist_output : PairlistData
            Output from the pairlist module containing distances and pair indices.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary with RBFs and atomic embeddings.
        """
        # Generate radial basis function expansion and apply cutoff
        f_ij = self.radial_symmetry_function_module(pairlist_output.d_ij).squeeze()
        f_ij = torch.mul(f_ij, self.cutoff_module(pairlist_output.d_ij))

        return {
            "f_ij": f_ij,
            "atomic_embedding": self.featurize_input(data),
        }


class PhysNetResidual(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function: torch.nn.Module,
    ):
        """
        Residual block for PhysNet, refining atomic feature vectors by adding
        a residual component.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature vector.
        output_dim : int
            Dimensionality of the output feature vector, which typically matches the
            input dimension.
        activation_function : Type[torch.nn.Module]
            The activation function to be used in the residual block.
        """
        super().__init__()

        # Define the dense layers and residual connection with activation
        self.dense = nn.Sequential(
            activation_function,
            Dense(input_dim, output_dim, activation_function),
            Dense(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying residual connection.
        """
        return x + self.dense(x)


class PhysNetInteractionModule(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        activation_function: torch.nn.Module,
    ):
        """
        Module for computing interaction terms based on atomic distances and features.

        Parameters
        ----------
        number_of_per_atom_features : int
            Dimensionality of the atomic embeddings.
        number_of_radial_basis_functions : int
            Number of radial basis functions for the interaction.
        number_of_interaction_residual : int
            Number of residual blocks in the interaction module.
        activation_function : torch.nn.Module
            The activation function to be used in the interaction module.
        """

        super().__init__()
        from .utils import DenseWithCustomDist

        # Initialize activation function
        self.activation_function = activation_function

        # Initialize attention mask
        self.attention_mask = DenseWithCustomDist(
            number_of_radial_basis_functions,
            number_of_per_atom_features,
            bias=False,
            weight_init=torch.nn.init.zeros_,
        )

        # Initialize networks for processing atomic embeddings of i and j atoms
        self.interaction_i = Dense(
            number_of_per_atom_features,
            number_of_per_atom_features,
            activation_function=activation_function,
        )
        self.interaction_j = Dense(
            number_of_per_atom_features,
            number_of_per_atom_features,
            activation_function=activation_function,
        )

        # Initialize processing network
        self.process_v = Dense(number_of_per_atom_features, number_of_per_atom_features)

        # Initialize residual blocks
        self.residuals = nn.ModuleList(
            [
                PhysNetResidual(
                    input_dim=number_of_per_atom_features,
                    output_dim=number_of_per_atom_features,
                    activation_function=activation_function,
                )
                for _ in range(number_of_interaction_residual)
            ]
        )

        # Gating and dropout layers
        self.gate = nn.Parameter(torch.ones(number_of_per_atom_features))
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the interaction module.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data including pairwise distances, pair indices, and atomic
            embeddings.

        Returns
        -------
        torch.Tensor
            Updated atomic embeddings after interaction computation.
        """

        idx_i, idx_j = data["pair_indices"].unbind()

        # Apply activation to atomic embeddings
        # first term in equation 6 in the PhysNet paper
        embedding_atom_i = self.activation_function(
            self.interaction_i(data["atomic_embedding"])
        )  # shape (nr_of_atoms_in_batch, atomic_embedding_dim)

        # second term in equation 6 in the PhysNet paper
        # apply attention mask G to radial basis functions f_ij
        g = self.attention_mask(
            data["f_ij"]
        )  # shape (nr_of_atom_pairs_in_batch, atomic_embedding_dim)
        # calculate the updated embedding for atom j
        # NOTE: this changes the 2nd dimension from number_of_radial_basis_functions to atomic_embedding_dim
        embedding_atom_j = self.activation_function(
            self.interaction_j(data["atomic_embedding"])[
                idx_j
            ]  # NOTE this is the same as the embedding_atom_i, but then we are selecting the embedding of atom j
            # shape (nr_of_atom_pairs_in_batch, atomic_embedding_dim)
        )
        updated_embedding_atom_j = torch.mul(
            g, embedding_atom_j
        )  # element-wise multiplication

        # Sum over contributions from atom j as function of embedding of atom i
        # and attention mask G(f_ij)
        embedding_atom_i.scatter_add_(
            0,
            idx_i.unsqueeze(-1).expand(-1, updated_embedding_atom_j.shape[-1]),
            updated_embedding_atom_j,
        )

        # apply residual blocks
        for residual in self.residuals:
            embedding_atom_i = residual(
                embedding_atom_i
            )  # shape (nr_of_atoms_in_batch, number_of_radial_basis_functions)

        # Apply dropout to the embedding after the residuals
        embedding_atom_i = self.dropout(embedding_atom_i)

        # eqn 5 in the PhysNet paper
        embedding_atom_i = self.gate * data["atomic_embedding"] + self.process_v(
            self.activation_function(embedding_atom_i)
        )
        return embedding_atom_i


class PhysNetOutput(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_atomic_properties: int,
        number_of_residuals_in_output: int,
        activation_function: torch.nn.Module,
    ):
        """
        Output module for the PhysNet model, responsible for generating predictions
        from atomic embeddings.

        Parameters
        ----------
        number_of_per_atom_features : int
            Dimensionality of the atomic embeddings.
        number_of_atomic_properties : int
            Number of atomic properties to predict.
        number_of_residuals_in_output : int
            Number of residual blocks in the output module.
        activation_function : torch.nn.Module
            Activation function to apply in the output module.
        """
        from .utils import DenseWithCustomDist

        super().__init__()
        # Initialize residual blocks
        self.residuals = nn.Sequential(
            *[
                PhysNetResidual(
                    number_of_per_atom_features,
                    number_of_per_atom_features,
                    activation_function,
                )
                for _ in range(number_of_residuals_in_output)
            ]
        )
        # Output layer for predicting atomic properties
        self.output = DenseWithCustomDist(
            number_of_per_atom_features,
            number_of_atomic_properties,
            weight_init=torch.nn.init.zeros_,  # NOTE: the result of this initialization is that before the first parameter update the output is zero
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the output module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing atomic feature vectors.

        Returns
        -------
        torch.Tensor
            Predicted atomic properties.
        """
        x = self.output(self.residuals(x))
        return x


class PhysNetModule(nn.Module):

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        activation_function: torch.nn.Module,
        number_of_residuals_in_output: int,
        number_of_atomic_properties: int,
    ):
        """
        Wrapper for the PhysNet interaction and output modules.

        Parameters
        ----------
        number_of_per_atom_features : int
            Dimensionality of the atomic embeddings.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        number_of_interaction_residual : int
            Number of residual blocks in the interaction module.
        activation_function : torch.nn.Module
            Activation function to apply in the modules.
        number_of_residuals_in_output : int
            Number of residual blocks in the output module.
        number_of_atomic_properties : int
            Number of atomic properties to predict.
        """

        super().__init__()

        # Initialize interaction module
        self.interaction = PhysNetInteractionModule(
            number_of_per_atom_features=number_of_per_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_residual=number_of_interaction_residual,
            activation_function=activation_function,
        )
        # Initialize output module
        self.output = PhysNetOutput(
            number_of_per_atom_features=number_of_per_atom_features,
            number_of_atomic_properties=number_of_atomic_properties,
            number_of_residuals_in_output=number_of_residuals_in_output,
            activation_function=activation_function,
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the PhysNet module.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data containing atomic features and pairwise information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing predictions and updated embeddings.
        """

        # Update embeddings via interaction
        updated_embedding = self.interaction(data)

        # Generate atomic property predictions
        prediction = self.output(updated_embedding)
        return {
            "prediction": prediction,
            "updated_embedding": updated_embedding,  # input for next module
        }


from typing import List


class PhysNetCore(torch.nn.Module):

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        number_of_modules: int,
        activation_function_parameter: Dict[str, str],
        predicted_properties: List[str],
        predicted_dim: List[int],
        potential_seed: int = -1,
    ) -> None:
        """
        Core implementation of PhysNet, combining multiple PhysNet modules.

        Parameters
        ----------
        featurization : Dict[str, Dict[str, int]]
            Configuration for atomic feature generation.
        maximum_interaction_radius : float
            Cutoff distance for atomic interactions.
        number_of_radial_basis_functions : int
            Number of radial basis functions for interaction computation.
        number_of_interaction_residual : int
            Number of residual blocks in the interaction modules.
        number_of_modules : int
            Number of PhysNet modules to stack.
        activation_function_parameter : Dict[str, str]
            Configuration for the activation function.
        predicted_properties : List[str]
            List of properties to predict.
        predicted_dim : List[int]
            List of dimensions corresponding to the predicted properties.
        potential_seed : int, optional
            Seed for random number generation, by default -1.
        """
        from modelforge.utils.misc import seed_random_number

        if potential_seed != -1:
            seed_random_number(potential_seed)

        super().__init__()
        self.activation_function = activation_function_parameter["activation_function"]

        log.debug("Initializing the PhysNet architecture.")

        # Initialize atomic feature dimensions and representation module
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )
        self.physnet_representation_module = PhysNetRepresentation(
            maximum_interaction_radius=maximum_interaction_radius,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            featurization_config=featurization,
        )

        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList

        self.output_dim = int(sum(predicted_dim))
        # Stack multiple PhysNet modules
        self.physnet_module = ModuleList(
            [
                PhysNetModule(
                    number_of_per_atom_features,
                    number_of_radial_basis_functions,
                    number_of_interaction_residual,
                    number_of_residuals_in_output=2,
                    number_of_atomic_properties=self.output_dim,
                    activation_function=self.activation_function,
                )
                for _ in range(number_of_modules)
            ]
        )

        # Define learnable atomic shift and scale per atomic property
        maximum_atomic_number = int(
            featurization["atomic_number"]["maximum_atomic_number"]
        )

        self.atomic_scale = nn.Parameter(
            torch.ones(
                maximum_atomic_number,
                len(predicted_properties),
            )
        )
        self.atomic_shift = nn.Parameter(
            torch.zeros(
                maximum_atomic_number,
                len(predicted_properties),
            )
        )

        self.predicted_properties = predicted_properties
        self.predicted_dim = predicted_dim

    def compute_properties(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties for a given input batch.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic features and pairwise information.
        pairlist_output : PairlistData
            Output from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated atomic properties.
        """

        # Compute representations for the input data
        representation = self.physnet_representation_module(data, pairlist_output)

        # Initialize tensor to store accumulated property predictions
        nr_of_atoms_in_batch = data.atomic_numbers.shape[0]
        per_atom_property_prediction = torch.zeros(
            (nr_of_atoms_in_batch, self.output_dim),
            device=data.atomic_numbers.device,
        )

        # Pass through stacked PhysNet modules
        module_data: Dict[str, torch.Tensor] = {
            "pair_indices": pairlist_output.pair_indices,
            "f_ij": representation["f_ij"],
            "atomic_embedding": representation["atomic_embedding"],
        }

        for module in self.physnet_module:
            module_output = module(module_data)
            # accumulate output for atomic properties
            per_atom_property_prediction = (
                per_atom_property_prediction + module_output["prediction"]
            )
            # update embedding for next module
            module_data["atomic_embedding"] = module_output["updated_embedding"]

        # Return computed properties and representations
        return {
            "per_atom_scalar_representation": module_output["updated_embedding"],
            "per_atom_prediction": per_atom_property_prediction,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

    def _aggregate_results(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate atomic property predictions into the final results.

        Parameters
        ----------
        per_atom_property_prediction : torch.Tensor
            Tensor of predicted per-atom properties.
        data : NNPInput
            Input data containing atomic numbers, etc.

        Returns
        -------
        Dict[str, torch.Tensor]
            Aggregated results containing per-atom predictions and other properties.
        """
        per_atom_prediction = outputs.pop("per_atom_prediction")
        # Apply atomic-specific scaling and shifting to the predicted properties
        atomic_numbers = outputs["atomic_numbers"]
        per_atom_prediction = (
            self.atomic_shift[atomic_numbers]
            + per_atom_prediction * self.atomic_scale[atomic_numbers]
        )  # NOTE: Questions: is this appropriate for partial charges?

        # Split predictions for each property
        split_tensors = torch.split(per_atom_prediction, self.predicted_dim, dim=1)
        outputs.update(
            {
                label: tensor
                for label, tensor in zip(self.predicted_properties, split_tensors)
            }
        )
        return outputs

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire PhysNet architecture.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic features and pairwise information.
        pairlist_output : PairlistData
            Pairwise information from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the predicted atomic properties.
        """
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(data, pairlist_output)
        # Aggregate and return the results
        return self._aggregate_results(outputs)

"""
Implementation of the PhysNet neural network potential.
"""

from typing import Dict

import torch
from loguru import logger as log
from torch import nn

from .models import NNPInputTuple, PairlistData
from .utils import Dense


class PhysNetRepresentation(nn.Module):
    def __init__(
        self,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        """
        Representation module for the PhysNet potential, handling the generation
        of the radial basis functions (RBFs) with a cutoff and atom number embedding.

        Parameters
        ----------
        maximum_interaction_radius : openff.units.unit.Quantity
            The cutoff distance for interactions.
        number_of_radial_basis_functions : int
            Number of radial basis functions to use.
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for atomic feature generation.
        """

        super().__init__()

        # Initialize cutoff module
        from modelforge.potential import CosineAttenuationFunction
        from modelforge.potential.utils import FeaturizeInput
        from .utils import PhysNetRadialBasisFunction

        self.cutoff_module = CosineAttenuationFunction(maximum_interaction_radius)

        # Initialize radial symmetry function module
        from modelforge.potential.utils import FeaturizeInput

        from .utils import PhysNetRadialBasisFunction

        self.featurize_input = FeaturizeInput(featurization_config)

        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the representation module.

        Parameters
        ----------
        data : PhysnetNeuralNetworkData
            pairwise distances between atoms, shape (n_pairs).

        Returns
        -------
        Dict[str, torch.Tensor]
            The radial basis function expansion applied to the input distances,
            shape (n_pairs, n_gaussians), after applying the cutoff function.
        """
        # calculate radial symmetry function
        f_ij = self.radial_symmetry_function_module(pairlist_output.d_ij).squeeze()
        # apply cutoff function
        f_ij = torch.mul(f_ij, self.cutoff_module(pairlist_output.d_ij))

        return {
            "f_ij": f_ij,
            "atomic_embedding": self.featurize_input(data),
        }


class PhysNetResidual(nn.Module):
    """
    Implements a preactivation residual block as described in Equation 4 of the
    PhysNet paper.

    The block refines atomic feature vectors by adding a residual component
    computed through two linear transformations and a non-linear activation
    function (Softplus). This setup enhances gradient flow and supports
    effective deep network training by employing a preactivation scheme.

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

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation_function: torch.nn.Module,
    ):
        super().__init__()
        # Initialize dense layers and residual connection

        self.dense = nn.Sequential(
            activation_function,
            Dense(input_dim, output_dim, activation_function),
            Dense(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing feature vectors of atoms.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the residual block operations.
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
        Module to compute interaction terms based on atomic distances and features.

        Parameters
        ----------
        number_of_per_atom_features : int
            Dimensionality of the atomic embeddings.
        number_of_radial_basis_functions : int
            Number of radial basis functions for the interaction.
        number_of_interaction_residual : int
            Number of residual blocks in the interaction module.
        activation_function : Type[torch.nn.Module]
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

        # Initialize gating and dropout
        self.gate = nn.Parameter(torch.ones(number_of_per_atom_features))
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Processes input tensors through the interaction module, applying
        Gaussian Logarithm Attention to modulate the influence of pairwise
        distances on the interaction features, followed by aggregation to update
        atomic embeddings.

        Parameters
        ----------
        data : PhysNetNeuralNetworkData
            Input data containing pair indices, distances, and atomic
            embeddings.

        Returns
        -------
        torch.Tensor
            Updated atomic feature representations incorporating interaction
            information.
        """

        # extract relevant variables
        idx_i, idx_j = data["pair_indices"].unbind()  # (nr_of_pairs, 2)

        # # Apply activation to atomic embeddings
        per_atom_embedding = self.activation_function(
            data["atomic_embedding"]
        )  # (nr_of_atoms_in_batch, number_of_per_atom_features)

        # calculate attention weights and transform to
        # input shape: (number_of_pairs, number_of_radial_basis_functions)
        # output shape: (number_of_pairs, number_of_per_atom_features)
        g = self.attention_mask(data["f_ij"])

        # Calculate contribution of central atom i
        per_atom_updated_embedding = self.interaction_i(per_atom_embedding)

        # Calculate contribution of neighbor atom
        per_interaction_embededding_for_atom_j = (
            self.interaction_j(per_atom_embedding[idx_j]) * g
        )

        per_atom_updated_embedding.scatter_add_(
            0,
            idx_i.unsqueeze(-1).expand(
                -1, per_interaction_embededding_for_atom_j.shape[-1]
            ),
            per_interaction_embededding_for_atom_j,
        )

        # apply residual blocks
        for residual in self.residuals:
            per_atom_updated_embedding = residual(
                per_atom_updated_embedding
            )  # shape (nr_of_atoms_in_batch, number_of_radial_basis_functions)

        per_atom_updated_embedding = self.activation_function(
            per_atom_updated_embedding
        )

        per_atom_embedding = self.gate * per_atom_embedding + self.process_v(
            per_atom_updated_embedding
        )
        return per_atom_embedding


class PhysNetOutput(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_atomic_properties: int,
        number_of_residuals_in_output: int,
        activation_function: torch.nn.Module,
    ):
        """
        Output module for the PhysNet model.

        Parameters
        ----------
        number_of_per_atom_features : int
            Dimensionality of the atomic embeddings.
        number_of_atomic_properties : int
            Number of atomic properties to predict.
        number_of_residuals_in_output : int
            Number of residual blocks in the output module.
        activation_function : Type[torch.nn.Module]
            The activation function to be used in the output module.
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
        # Initialize output layer
        self.output = DenseWithCustomDist(
            number_of_per_atom_features,
            number_of_atomic_properties,
            weight_init=torch.nn.init.zeros_,  # NOTE: the result of this initialization is that before the first parameter update the output is zero
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
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
    """
    Wrapper module that combines the PhysNetInteraction, PhysNetResidual, and PhysNetOutput classes into a single module.

    Parameters
    ----------
    number_of_per_atom_features : int
        Dimensionality of the atomic embeddings.
    number_of_radial_basis_functions : int
        Number of radial basis functions for the interaction.
    number_of_interaction_residual : int
        Number of residual blocks in the interaction module.
    activation_function : Type[torch.nn.Module]
        The activation function to be used in the modules.
    """

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        activation_function: torch.nn.Module,
        number_of_residuals_in_output: int,
        number_of_atomic_properties: int,
    ):

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
        data : PhysNetNeuralNetworkData
            Input data containing atomic features and pairwise information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing predictions and updated embeddings.
        """

        # calculate the interaction
        updated_embedding = self.interaction(data)

        # calculate the module output
        prediction = self.output(updated_embedding)
        return {
            "prediction": prediction,
            "updated_embedding": updated_embedding,  # input for next module
        }


from typing import Tuple, List


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

        from modelforge.utils.misc import seed_random_number

        if potential_seed != -1:
            seed_random_number(potential_seed)

        super().__init__()
        self.activation_function = activation_function_parameter["activation_function"]

        log.debug("Initializing the PhysNet architecture.")

        # featurize the atomic input
        number_of_per_atom_features = int(
            featurization["atomic_number"]["number_of_per_atom_features"]
        )
        maximum_atomic_number = int(
            featurization["atomic_number"]["maximum_atomic_number"]
        )
        self.physnet_representation_module = PhysNetRepresentation(
            maximum_interaction_radius=maximum_interaction_radius,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            featurization_config=featurization,
        )

        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList

        self.output_dim = int(sum(predicted_dim))
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

        # learnable shift and bias that is applied per-element to ech atomic energy
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

    def aggregate_results(
        self,
        per_atom_property_prediction: torch.Tensor,
        data: NNPInputTuple,
    ) -> Dict[str, torch.Tensor]:

        # predictions are shifted and scaled as a function of atomic number
        atomic_numbers = data.atomic_numbers
        linear_transformation_of_results = (
            self.atomic_shift[atomic_numbers]
            + per_atom_property_prediction * self.atomic_scale[atomic_numbers]
        )  # NOTE: Questions: is this appropriate for partial charges?

        # initialize the results dictionary
        results = {
            "per_atom_scalar_representation": per_atom_property_prediction,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }
        split_tensors = torch.split(
            linear_transformation_of_results, self.predicted_dim, dim=1
        )
        results.update(
            {
                label: tensor
                for label, tensor in zip(self.predicted_properties, split_tensors)
            }
        )

        # add user requested properties
        return results

    def compute_properties(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties for a given input batch.

        Parameters
        ----------
        data : PhysNetNeuralNetworkData
            Input data containing atomic features and pairwise information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated properties including per-atom energies.
        """

        # Computed representation
        representation = self.physnet_representation_module(data, pairlist_output)

        # the per atom predictions are accumulated in
        # per_atom_property_prediction
        nr_of_atoms_in_batch = data.atomic_numbers.shape[0]
        per_atom_property_prediction = torch.zeros(
            (nr_of_atoms_in_batch, self.output_dim),
            device=data.atomic_numbers.device,
        )

        information_to_pass: Dict[str, torch.Tensor] = {
            "pair_indices": pairlist_output.pair_indices,
            "f_ij": representation["f_ij"],
            "atomic_embedding": representation["atomic_embedding"],
        }

        for module in self.physnet_module:
            output_of_module = module(information_to_pass)
            # accumulate output for atomic properties
            per_atom_property_prediction = (
                per_atom_property_prediction + output_of_module["prediction"]
            )
            # update embedding for next module
            information_to_pass["atomic_embedding"] = output_of_module[
                "updated_embedding"
            ]

        return {
            "per_atom_scalar_representation": output_of_module["updated_embedding"],
            "per_atom_prediction": per_atom_property_prediction,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

    def _aggregate_results(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        per_atom_prediction = outputs.pop("per_atom_prediction")
        split_tensors = torch.split(per_atom_prediction, self.predicted_dim, dim=1)
        outputs.update(
            {
                label: tensor.squeeze(1)
                for label, tensor in zip(self.predicted_properties, split_tensors)
            }
        )
        return outputs

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
        # extract predictions per property
        return self._aggregate_results(outputs)

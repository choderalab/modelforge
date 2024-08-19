"""
Implementation of the PhysNet neural network potential.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

import torch
from loguru import logger as log
from openff.units import unit
from torch import nn

from modelforge.potential.utils import NeuralNetworkData, shared_config_prior
from modelforge.utils.io import import_
from modelforge.utils.units import _convert_str_to_unit

from .models import BaseNetwork, CoreNetwork, NNPInput, PairListOutputs
from .utils import Dense


@dataclass
class PhysNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs for PhysNet-based neural network
    potentials, facilitating the efficient and structured representation of
    atomic systems for energy computation and property prediction within the
    PhysNet framework.

    Attributes
    ----------
    atomic_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived
        from atomic numbers or other properties. Shape: [num_atoms,
        embedding_dim].
    f_ij : Optional[torch.Tensor]
        A tensor representing the radial basis function (RBF) expansion applied
        to distances between atom pairs, capturing the local chemical
        environment. Will be added after initialization. Shape: [num_pairs,
        num_rbf].
    """

    atomic_embedding: Optional[torch.Tensor] = field(default=None)
    f_ij: Optional[torch.Tensor] = field(default=None)


class PhysNetRepresentation(nn.Module):
    def __init__(
        self,
        maximum_interaction_radius: unit.Quantity,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Union[List[str], int]],
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

    def forward(self, data: Type[PhysNetNeuralNetworkData]) -> Dict[str, torch.Tensor]:
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
        f_ij = self.radial_symmetry_function_module(data.d_ij).squeeze()
        return {
            "f_ij": f_ij,
            "f_ij_cutoff": self.cutoff_module(data.d_ij),
            "atomic_embedding": self.featurize_input(
                data
            ),  # add per-atom properties and embedding
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
        activation_function: Type[torch.nn.Module],
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
        activation_function: Type[torch.nn.Module],
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

    def forward(self, data: PhysNetNeuralNetworkData) -> torch.Tensor:
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
        idx_i, idx_j = data.pair_indices  # (nr_of_pairs, 2)
        f_ij = data.f_ij  # (nr_of_pairs, number_of_radial_basis_functions)

        # # Apply activation to atomic embeddings
        per_atom_embedding = self.activation_function(
            data.atomic_embedding
        )  # (nr_of_atoms_in_batch, number_of_per_atom_features)

        # calculate attention weights and transform to
        # input shape: (number_of_pairs, number_of_radial_basis_functions)
        # output shape: (number_of_pairs, number_of_per_atom_features)
        g = self.attention_mask(f_ij)

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
        activation_function: Type[torch.nn.Module],
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
            weight_init=torch.nn.init.zeros_,
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
        activation_function: Type[torch.nn.Module],
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
            number_of_atomic_properties=2,
            number_of_residuals_in_output=2,
            activation_function=activation_function,
        )

    def forward(self, data: PhysNetNeuralNetworkData) -> Dict[str, torch.Tensor]:
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
        # The PhysNet module is a sequence of interaction modules and residual modules.
        #              x_1, ..., x_N
        #                     |
        #                     v
        #               ┌─────────────┐
        #               │ interaction │ <-- g(d_ij)
        #               └─────────────┘
        #                     │
        #                     v
        #                ┌───────────┐
        #                │  residual │
        #                └───────────┘
        #                ┌───────────┐
        #                │  residual │
        #                └───────────┘
        # ┌───────────┐      │
        # │   output  │<-----│
        # └───────────┘      │
        #                    v

        # calculate the interaction
        v = self.interaction(data)

        # calculate the module output
        prediction = self.output(v)
        return {
            "prediction": prediction,
            "updated_embedding": v,  # input for next module
        }


class PhysNetCore(CoreNetwork):
    """
    Core network class for the PhysNet neural network potential.

    Parameters
    ----------
    featurization_config : Dict[str, Union[List[str], int]]
        Configuration for atomic feature generation.
    maximum_interaction_radius : openff.units.unit.Quantity
        The cutoff distance for interactions.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    number_of_interaction_residual : int
        Number of interaction residual blocks.
    number_of_modules : int
        Number of PhysNet modules.
    activation_function : Type[torch.nn.Module]
        The activation function to use.
    """

    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        maximum_interaction_radius: unit.Quantity,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        number_of_modules: int,
        activation_function: Type[torch.nn.Module],
    ) -> None:

        log.debug("Initializing the PhysNet architecture.")
        super().__init__(activation_function)

        # featurize the atomic input
        number_of_per_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )
        maximum_atomic_number = int(featurization_config["maximum_atomic_number"])
        self.physnet_representation_module = PhysNetRepresentation(
            maximum_interaction_radius=maximum_interaction_radius,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )

        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList

        self.physnet_module = ModuleList(
            [
                PhysNetModule(
                    number_of_per_atom_features,
                    number_of_radial_basis_functions,
                    number_of_interaction_residual,
                    activation_function=self.activation_function,
                )
                for _ in range(number_of_modules)
            ]
        )

        # learnable shift and bias that is applied per-element to ech atomic energy
        self.atomic_scale = nn.Parameter(torch.ones(maximum_atomic_number, 2))
        self.atomic_shift = nn.Parameter(torch.zeros(maximum_atomic_number, 2))

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> PhysNetNeuralNetworkData:
        """
        Prepare model-specific input data.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic information.
        pairlist_output : PairListOutputs
            Output from the pairlist calculation.

        Returns
        -------
        PhysNetNeuralNetworkData
            Prepared input data for the PhysNet model.
        """
        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = PhysNetNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            f_ij=None,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
        )

        return nnp_input

    def compute_properties(
        self, data: PhysNetNeuralNetworkData
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
        representation = self.physnet_representation_module(data)
        nr_of_atoms_in_batch = data.number_of_atoms

        data.atomic_embedding = representation["atomic_embedding"]
        data.f_ij = torch.mul(representation["f_ij"], representation["f_ij_cutoff"])

        #         d_i, ..., d_N
        #
        #             │
        #             V
        #        ┌────────────┐
        #        │    RBF     │
        #        └────────────┘

        # see https://doi.org/10.1021/acs.jctc.9b00181
        # in the following we are implementing the calculations analoguous
        # to the modules outlined in Figure 1

        # NOTE: both embedding and f_ij (the output of the Radial Symmetry
        # Function) are stored in `inputs` inputs are the embedding vectors and
        # f_ij the embedding vector will get updated in each pass through the
        # modules

        #             ┌────────────┐         ┌────────────┐
        #             │ embedding  │         │    RBF     │
        #             └────────────┘         └────────────┘
        #                        |                   │
        #                       ┌───────────────┐    │
        #                 | <-- |   module 1    │ <--│
        #                 |     └────────────---┘    │
        #                 |            |             │
        #  E_1, ..., E_N (+)           V             │
        #                 |     ┌───────────────┐    │
        #                 | <-- |   module 2    │ <--│
        #                       └────────────---┘

        # the atomic energies are accumulated in per_atom_energies
        prediction_i = torch.zeros(
            (nr_of_atoms_in_batch, 2),
            device=data.d_ij.device,
        )

        for module in self.physnet_module:
            output_of_module = module(data)
            # accumulate output for atomic properties
            prediction_i += output_of_module["prediction"]
            # update embedding for next module
            data.atomic_embedding = output_of_module["updated_embedding"]

        prediction_i_shifted_scaled = (
            self.atomic_shift[data.atomic_numbers]
            + prediction_i * self.atomic_scale[data.atomic_numbers]
        )

        # sum over atom features
        E_i = prediction_i_shifted_scaled[:, 0]  # shape(nr_of_atoms, 1)
        q_i = prediction_i_shifted_scaled[:, 1]  # shape(nr_of_atoms, 1)

        return {
            "per_atom_energy": E_i.contiguous(),  # reshape memory mapping for JAX/dlpack
            "per_atom_charge": q_i.contiguous(),
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }


class PhysNet(BaseNetwork):
    """
    Implementation of the PhysNet neural network potential.

    Parameters
    ----------
    featurization : Dict[str, Union[List[str], int]]
        Configuration for atomic feature generation.
    maximum_interaction_radius : Union[unit.Quantity, str]
        The cutoff distance for interactions.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    number_of_interaction_residual : int
        Number of interaction residual blocks.
    number_of_modules : int
        Number of PhysNet modules.
    activation_function_parameter : Dict
        Dict that contains keys: activation_function_name [str], activation_function_arguments [Dict],
        and activation_function [Type[torch.nn.Module]].
    postprocessing_parameter : Dict[str, Dict[str, bool]]
        Configuration for postprocessing parameters.
    dataset_statistic : Optional[Dict[str, float]], optional
        Statistics of the dataset, by default None.
    potential_seed : Optional[int], optional
        Seed for the random number generator, default None.
    """

    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        maximum_interaction_radius: Union[unit.Quantity, str],
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        number_of_modules: int,
        activation_function_parameter: Dict,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        potential_seed: Optional[int] = None,
    ) -> None:

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            potential_seed=potential_seed,
        )
        activation_function = activation_function_parameter["activation_function"]

        self.core_module = PhysNetCore(
            featurization_config=featurization,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_residual=number_of_interaction_residual,
            number_of_modules=number_of_modules,
            activation_function=activation_function,
        )

    def _config_prior(self):
        """
        Configure the hyperparameter prior distribution for the PhysNet model.

        Returns
        -------
        dict
            The hyperparameter prior distribution.
        """
        log.info("Configuring PhysNet model hyperparameter prior distribution")

        from ray import tune

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_modules": tune.randint(2, 8),
            "number_of_interaction_residual": tune.randint(2, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
        }
        prior.update(shared_config_prior())
        return prior

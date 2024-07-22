from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from loguru import logger as log
from openff.units import unit
from torch import nn
from .models import InputPreparation, NNPInput, BaseNetwork, CoreNetwork

from modelforge.potential.utils import NeuralNetworkData

if TYPE_CHECKING:
    from modelforge.dataset.dataset import NNPInput

    from .models import PairListOutputs


@dataclass
class PhysNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs for PhysNet-based neural network potentials,
    facilitating the efficient and structured representation of atomic systems for
    energy computation and property prediction within the PhysNet framework.

    Attributes
    ----------
    f_ij : Optional[torch.Tensor]
        A tensor representing the radial basis function (RBF) expansion applied to distances between atom pairs,
        capturing the local chemical environment. Will be added after initialization. Shape: [num_pairs, num_rbf].
    number_of_atoms : int
        An integer indicating the number of atoms in the batch.
    atomic_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived from atomic numbers or other properties.
        Shape: [num_atoms, embedding_dim].

    Notes
    -----
    The `PhysNetNeuralNetworkInput` class encapsulates essential geometric and chemical information required by
    the PhysNet model to predict system energies and properties. It includes information on atomic positions, types,
    and connectivity, alongside derived features such as radial basis functions (RBF) for detailed representation
    of atomic environments. This structured input format ensures that all relevant data is readily available for
    the PhysNet model, supporting its complex network architecture and computation requirements.

    Examples
    --------
    >>> physnet_input = PhysNetNeuralNetworkInput(
    ...     atomic_numbers=torch.tensor([1, 6, 6, 8]),
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ...     atomic_subsystem_indices=torch.tensor([0, 0, 0, 0]),
    ...     total_charge=torch.tensor([0.0]),
    ...     pair_indices=torch.tensor([[0, 1], [0, 2], [1, 2]]),
    ...     d_ij=torch.tensor([1.0, 1.0, 1.0]),
    ...     r_ij=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ...     f_ij=torch.randn(3, 4),  # Radial basis function expansion
    ...     number_of_atoms=torch.tensor([4]),
    ...     atomic_embedding=torch.randn(4, 5)  # Example atomic embeddings/features
    ... )
    """

    atomic_embedding: torch.Tensor
    f_ij: Optional[torch.Tensor] = field(default=None)


class PhysNetRepresentation(nn.Module):
    def __init__(
        self,
        cutoff: unit = 5 * unit.angstrom,
        number_of_radial_basis_functions: int = 16,
    ):
        """
        Representation module for the PhysNet potential, handling the generation of
        the radial basis functions (RBFs) with a cutoff.

        Parameters
        ----------
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        number_of_radial_basis_functions : int, default=16
            Number of radial basis functions to use.
        """

        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff)

        # radial symmetry function
        from .utils import PhysNetRadialBasisFunction

        self.radial_symmetry_function_module = PhysNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=cutoff,
            dtype=torch.float32,
        )

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the representation module.

        Parameters
        ----------
        d_ij : torch.Tensor
            pairwise distances between atoms, shape (n_pairs).

        Returns
        -------
        torch.Tensor
            The radial basis function expansion applied to the input distances,
            shape (n_pairs, n_gaussians), after applying the cutoff function.
        """

        f_ij = self.radial_symmetry_function_module(d_ij).squeeze()
        cutoff = self.cutoff_module(d_ij)
        f_ij = torch.mul(f_ij, cutoff)
        return f_ij


class GatingModule(nn.Module):
    def __init__(self, number_of_atom_basis: int):
        """
        Initializes a gating module that
        optionally applies a sigmoid gating mechanism to input features.

        Parameters:
        -----------
        input_dim : int
            The dimensionality of the input (and output) features.
        """
        super().__init__()
        self.gate = nn.Parameter(torch.ones(number_of_atom_basis))

    def forward(self, x: torch.Tensor, activation_fn: bool = False) -> torch.Tensor:
        """
        Apply gating to the input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor to gate.

        Returns:
        --------
        torch.Tensor
            The gated input tensor.
        """
        gating_signal = torch.sigmoid(self.gate)
        return gating_signal * x


from .utils import ShiftedSoftplus, Dense


class PhysNetResidual(nn.Module):
    """
    Implements a preactivation residual block as described in Equation 4 of the PhysNet paper.

    The block refines atomic feature vectors by adding a residual component computed through
    two linear transformations and a non-linear activation function (Softplus). This setup
    enhances gradient flow and supports effective deep network runtime_defaults by employing a
    preactivation scheme.

    Parameters:
    -----------
    input_dim: int
        Dimensionality of the input feature vector.
    output_dim: int
        Dimensionality of the output feature vector, which typically matches the input dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = Dense(input_dim, output_dim, activation=ShiftedSoftplus())
        self.residual = Dense(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor containing feature vectors of atoms.

        Returns:
        --------
        torch.Tensor
            Output tensor after applying the residual block operations.
        """
        # update x with residual
        return x + self.residual(self.dense(x))


class PhysNetInteractionModule(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 16,
        number_of_interaction_residual: int = 3,
    ):
        """
        Module to compute interaction terms based on atomic distances and features.

        Parameters
        ----------
        number_of_atom_features : int, default=64
            Dimensionality of the atomic embeddings.
        number_of_radial_basis_functions : int, default=16
            Specifies the number of basis functions for the Gaussian Logarithm Attention,
            essentially defining the output feature dimension for attention-weighted interactions.
        """

        super().__init__()
        from .utils import ShiftedSoftplus, Dense

        self.attention_mask = Dense(
            number_of_radial_basis_functions,
            number_of_atom_features,
            bias=False,
            weight_init=torch.nn.init.zeros_,
        )
        self.activation_function = ShiftedSoftplus()

        # Networks for processing atomic embeddings of i and j atoms
        self.interaction_i = Dense(
            number_of_atom_features,
            number_of_atom_features,
            activation=self.activation_function,
        )
        self.interaction_j = Dense(
            number_of_atom_features,
            number_of_atom_features,
            activation=self.activation_function,
        )

        self.process_v = Dense(number_of_atom_features, number_of_atom_features)

        # Residual block
        self.residuals = nn.ModuleList(
            [
                PhysNetResidual(number_of_atom_features, number_of_atom_features)
                for _ in range(number_of_interaction_residual)
            ]
        )

        # Gating
        self.gate = nn.Parameter(torch.ones(number_of_atom_features))
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, data: PhysNetNeuralNetworkData) -> torch.Tensor:
        """
        Processes input tensors through the interaction module, applying
        Gaussian Logarithm Attention to modulate the influence of pairwise distances
        on the interaction features, followed by aggregation to update atomic embeddings.

        Parameters
        ----------
        inputs : PhysNetNeuralNetworkInput

        Returns
        -------
        torch.Tensor
            Updated atomic feature representations incorporating interaction information.
        """
        # Equation 6: Formation of the Proto-Message ṽ_i for an Atom i
        # ṽ_i = σ(Wl_I * x_i^l + bl_I) + Σ_j (G_g * Wl * (σ(σl_J * x_j^l + bl_J)) * g(r_ij))
        # Equation 6 implementation overview:
        # ṽ_i = x_i_prime + sum_over_j(x_j_prime * f_ij_prime)
        # where:
        # - x_i_prime and x_j_prime are the features of atoms i and j, respectively, processed through separate networks.
        # - f_ij_prime represents the modulated radial basis functions (f_ij) by the Gaussian Logarithm Attention weights.

        # extract relevant variables
        idx_i, idx_j = data.pair_indices
        f_ij = data.f_ij
        x = data.atomic_embedding

        # # Apply activation to atomic embeddings
        xa = self.dropout(self.activation_function(x))

        # calculate attention weights and
        # transform to
        # input shape: (number_of_pairs, number_of_radial_basis_functions)
        # output shape: (number_of_pairs, number_of_atom_features)
        g = self.attention_mask(f_ij)

        # Calculate contribution of central atom
        x_i = self.interaction_i(xa)
        # Calculate contribution of neighbor atom
        x_j = self.interaction_j(xa)
        # Gather the results according to idx_j
        x_j = x_j[idx_j]
        # Multiply the gathered features by g
        x_j_modulated = x_j * g
        # Aggregate modulated contributions for each atom i
        x_j_prime = torch.zeros_like(x_i)
        x_j_prime.scatter_add_(
            0, idx_i.unsqueeze(-1).expand(-1, x_j_modulated.size(-1)), x_j_modulated
        )

        # Draft proto message v_tilde
        m = x_i + x_j_prime
        # shape of m (nr_of_atoms_in_batch, 1)
        # Equation 4: Preactivation Residual Block Implementation
        # xl+2_i = xl_i + Wl+1 * sigma(Wl * xl_i + bl) + bl+1
        for residual in self.residuals:
            m = residual(
                m
            )  # shape (nr_of_atoms_in_batch, number_of_radial_basis_functions)
        m = self.activation_function(m)
        x = self.gate * x + self.process_v(m)
        return x


class PhysNetOutput(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int,
        number_of_atomic_properties: int = 2,
        number_of_residuals_in_output: int = 2,
    ):
        from .utils import Dense

        super().__init__()
        self.residuals = nn.Sequential(
            *[
                PhysNetResidual(number_of_atom_features, number_of_atom_features)
                for _ in range(number_of_residuals_in_output)
            ]
        )
        self.output = Dense(
            number_of_atom_features,
            number_of_atomic_properties,
            weight_init=torch.nn.init.zeros_,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output(self.residuals(x))
        return x


class PhysNetModule(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 16,
        number_of_interaction_residual: int = 2,
    ):
        """
        Wrapper module that combines the PhysNetInteraction, PhysNetResidual, and
        PhysNetOutput classes into a single module. This serves as the building
        block for the PhysNet model.

        This is a skeletal implementation that needs to be expanded upon.
        """

        super().__init__()

        # this class combines the PhysNetInteraction, PhysNetResidual and
        # PhysNetOutput class

        self.interaction = PhysNetInteractionModule(
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_residual=number_of_interaction_residual,
        )
        self.output = PhysNetOutput(
            number_of_atom_features=number_of_atom_features,
            number_of_atomic_properties=2,
        )

    def forward(self, data: PhysNetNeuralNetworkData) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the PhysNet module.
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
    def __init__(
        self,
        max_Z: int,
        cutoff: unit.Quantity,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        number_of_modules: int,
    ) -> None:
        """
        Implementation of the PhysNet neural network potential.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        number_of_modules : int, default=2(
        """

        log.debug("Initializing PhysNet model.")
        super().__init__()

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, number_of_atom_features)

        self.physnet_representation_module = PhysNetRepresentation(
            cutoff=cutoff,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
        )

        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList

        self.physnet_module = ModuleList(
            [
                PhysNetModule(
                    number_of_atom_features,
                    number_of_radial_basis_functions,
                    number_of_interaction_residual,
                )
                for _ in range(number_of_modules)
            ]
        )

        self.atomic_scale = nn.Parameter(torch.ones(max_Z, 2))
        self.atomic_shift = nn.Parameter(torch.zeros(max_Z, 2))

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> PhysNetNeuralNetworkData:
        # Perform atomic embedding
        atomic_embedding = self.embedding_module(data.atomic_numbers)
        #         Z_i, ..., Z_N
        #
        #             │
        #             ∨
        #        ┌────────────┐
        #        │ embedding  │
        #        └────────────┘

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
            atomic_embedding=atomic_embedding,  # atom embedding
        )

        return nnp_input

    def compute_properties(
        self, data: PhysNetNeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the energy for a given input batch.
        Parameters
        ----------
        inputs : PhysNetNeutralNetworkInput

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # Computed representation
        data.f_ij = self.physnet_representation_module(data.d_ij).squeeze(
            1
        )  # shape: (n_pairs, number_of_radial_basis_functions)
        nr_of_atoms_in_batch = data.number_of_atoms

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

        # NOTE: both embedding and f_ij (the output of the Radial Symmetry Function) are
        # stored in `inputs`
        # inputs are the embedding vectors and f_ij
        # the embedding vector will get updated in each pass through the modules

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
            # accumulate output for atomic energies
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

        output = {
            "per_atom_energy": E_i.contiguous(),  # reshape memory mapping for JAX/dlpack
            "q_i": q_i.contiguous(),
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

        return output


from .models import InputPreparation, NNPInput, BaseNetwork
from typing import List


class PhysNet(BaseNetwork):
    def __init__(
        self,
        max_Z: int,
        cutoff: Union[unit.Quantity, str],
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_residual: int,
        number_of_modules: int,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies,
        Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).


        """
        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
        )
        from modelforge.utils.units import _convert

        self.core_module = PhysNetCore(
            max_Z=max_Z,
            cutoff=_convert(cutoff),
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_residual=number_of_interaction_residual,
            number_of_modules=number_of_modules,
        )
        self.only_unique_pairs = False  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=_convert(cutoff), only_unique_pairs=self.only_unique_pairs
        )

    def _config_prior(self):
        log.info("Configuring SchNet model hyperparameter prior distribution")
        from modelforge.utils.io import import_

        tune = import_("ray").tune
        # from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_modules": tune.randint(2, 8),
            "number_of_interaction_residual": tune.randint(2, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
        }
        prior.update(shared_config_prior())
        return prior

    def combine_per_atom_properties(
        self, values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return values

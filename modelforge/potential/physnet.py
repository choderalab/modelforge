from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import torch
from loguru import logger as log
from openff.units import unit
from torch import nn
from torch_scatter import scatter_add

from modelforge.potential.utils import NeuralNetworkData

from .models import BaseNeuralNetworkPotential

if TYPE_CHECKING:
    from modelforge.potential.utils import NNPInput

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
        number_of_gaussians : int, default=16
            Number of Gaussian functions to use in the radial basis function.
        """

        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff)

        # radial symmetry function
        from .utils import SchnetRadialSymmetryFunction

        self.radial_symmetry_function_module = SchnetRadialSymmetryFunction(
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

        rbf = self.radial_symmetry_function_module(d_ij)
        cutoff = self.cutoff_module(d_ij)
        f_ij = torch.mul(rbf, cutoff)
        return f_ij


class GatingModule(nn.Module):
    def __init__(self, number_of_atom_basis: int):
        """
        Initializes a gating module that applies a sigmoid gating mechanism to input features.

        Parameters:
        -----------
        input_dim : int
            The dimensionality of the input (and output) features.
        """
        super().__init__()
        self.gate = nn.Parameter(torch.randn(number_of_atom_basis))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class AttentionMask(nn.Module):
    def __init__(
        self, number_of_radial_basis_functions: int, number_of_atom_features: int
    ):
        super().__init__()

        # Learnable matrix for the attention mask
        self.G = nn.Parameter(
            torch.randn(number_of_atom_features, number_of_radial_basis_functions)
        )

    def forward(self, f_ij: torch.Tensor) -> torch.Tensor:
        """
        Apply the attention mask to the radial symmetry function outputs.
        f_ij: Tensor of shape (number_of_pairs, number_of_radial_basis_functions) containing RBF applied distances.
        """
        # Apply the attention mask
        attention_output = torch.matmul(f_ij, self.G.t())
        return attention_output


class PhysNetResidual(nn.Module):
    """
    Implements a preactivation residual block as described in Equation 4 of the PhysNet paper.

    The block refines atomic feature vectors by adding a residual component computed through
    two linear transformations and a non-linear activation function (Softplus). This setup
    enhances gradient flow and supports effective deep network training by employing a
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
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.activation = nn.Softplus()

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
        # Apply the first linear transformation and activation
        residual = self.activation(self.linear1(x))
        # Apply the second linear transformation
        residual = self.linear2(residual)
        # Add the input x (identity) to the output of the second linear layer
        out = x + residual
        return out


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
        from .utils import ShiftedSoftplus

        self.attention_mask = AttentionMask(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_atom_features=number_of_atom_features,
        )
        self.activation_function = ShiftedSoftplus()

        # Networks for processing atomic embeddings of i and j atoms
        self.interaction_i = nn.Sequential(
            nn.Linear(number_of_atom_features, number_of_atom_features),
            nn.Softplus(),
        )
        self.interaction_j = nn.Sequential(
            nn.Linear(number_of_atom_features, number_of_atom_features),
            nn.Softplus(),
        )

        self.process_v = nn.Sequential(
            nn.Softplus(),
            nn.Linear(number_of_atom_features, number_of_atom_features),
        )

        # Residual block
        self.residuals = nn.ModuleList(
            [
                PhysNetResidual(number_of_atom_features, number_of_atom_features)
                for _ in range(number_of_interaction_residual)
            ]
        )

        # Gating Module
        self.gating_module = GatingModule(number_of_atom_features)

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

        # extract relevant variables
        idx_i, idx_j = data.pair_indices
        f_ij = data.f_ij
        d_ij = data.d_ij
        atom_embedding = data.atomic_embedding

        # Apply activation to atomic embeddings
        x = self.activation_function(atom_embedding)

        # extract all interaction partners for atom i
        x_j_embedding = x[idx_j]  # shape (nr_of_pairs, number_of_atom_features)
        # extract all atoms i
        x_i_embedding = x  # shape (nr_of_atoms, number_of_atom_features)

        # Equation 6: Formation of the Proto-Message ṽ_i for an Atom i
        # ṽ_i = σ(Wl_I * x_i^l + bl_I) + Σ_j (G_g * Wl * (σ(σl_J * x_j^l + bl_J)) * g(r_ij))
        # Equation 6 implementation overview:
        # ṽ_i = x_i_prime + sum_over_j(x_j_prime * f_ij_prime)
        # where:
        # - x_i_prime and x_j_prime are the features of atoms i and j, respectively, processed through separate networks.
        # - f_ij_prime represents the modulated radial basis functions (f_ij) by the Gaussian Logarithm Attention weights.

        # Draft proto message v_tilde
        x_i_prime = self.interaction_i(x_i_embedding)
        x_j_prime = self.interaction_j(x_j_embedding)

        f_ij_prime = self.attention_mask(
            f_ij
        )  # shape: (number_of_pairs, number_of_atom_features)

        # Compute sum_over_j(x_j_prime * f_ij_prime)
        x_j_modulated = torch.mul(x_j_prime, f_ij_prime)
        # Aggregate modulated contributions for each atom i
        summed_contributions = scatter_add(
            x_j_modulated, idx_i, dim=0, dim_size=atom_embedding.shape[0]
        )
        v_tilde = x_i_prime + summed_contributions
        # shape of v_tilde (nr_of_atoms_in_batch, 1)
        # Equation 4: Preactivation Residual Block Implementation
        # xl+2_i = xl_i + Wl+1 * sigma(Wl * xl_i + bl) + bl+1
        for residual in self.residuals:
            v_tilde = residual(
                v_tilde
            )  # shape (nr_of_atoms_in_batch, number_of_radial_basis_functions)

        # FIXME: add gated vector
        v = self.process_v(
            v_tilde
        )  # shape (nr_of_atoms_in_batch, number_of_atom_features)
        return v


class PhysNetOutput(nn.Module):

    def __init__(
        self, number_of_atom_features: int, number_of_atomic_properties: int = 2
    ):
        from .utils import ShiftedSoftplus

        super().__init__()
        self.residual = PhysNetResidual(
            number_of_atom_features, number_of_atom_features
        )
        self.output = nn.Sequential(
            nn.Linear(number_of_atom_features, number_of_atomic_properties),
            ShiftedSoftplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output(self.residual(x))
        return x


class PhysNetModule(nn.Module):

    def __init__(
        self,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 16,
        number_of_interaction_residual: int = 3,
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
        self.residuals = nn.ModuleList(
            [
                PhysNetResidual(number_of_atom_features, number_of_atom_features)
                for _ in range(number_of_interaction_residual)
            ]
        )
        self.output = PhysNetOutput(
            number_of_atom_features=number_of_atom_features,
            number_of_atomic_properties=2,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the PhysNet module. Currently, a placeholder that
        needs further implementation.
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
        v = self.interaction(inputs)

        # add the atomic residual blocks
        for residual in self.residuals:
            v = residual(v)

        # calculate the module output
        module_output = self.output(v)
        return {
            "prediction": module_output,
            "updated_embedding": v,  # input for next module
        }


class PhysNet(BaseNeuralNetworkPotential):
    def __init__(
        self,
        max_Z: int = 100,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 16,
        number_of_interaction_residual: int = 3,
        number_of_modules: int = 5,
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
        number_of_modules : int, default=2
            Number of PhysNet modules to be stacked in the network.
        """

        log.debug("Initializing PhysNet model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff, only_unique_pairs=self.only_unique_pairs)

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
        self.atomic_shift = nn.Parameter(torch.ones(max_Z, 2))

    def _config_prior(self):
        log.info("Configuring SchNet model hyperparameter prior distribution")
        from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_modules": tune.randint(3, 8),
            "number_of_interaction_residual": tune.randint(2, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
        }
        prior.update(shared_config_prior())
        return prior

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

    def _forward(self, data: PhysNetNeuralNetworkData) -> Dict[str, torch.Tensor]:
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
            self.atomic_shift[data.atomic_numbers] * prediction_i
            + self.atomic_scale[data.atomic_numbers]
        )

        # sum over atom features
        E_i = prediction_i_shifted_scaled[:, 0]  # shape(nr_of_atoms, 1)
        q_i = prediction_i_shifted_scaled[:, 1]  # shape(nr_of_atoms, 1)

        output = {
            "E_i": E_i,
            "q_i": q_i,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

        return output

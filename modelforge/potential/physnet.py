from .models import BaseNeuralNetworkPotential
from loguru import logger as log
from openff.units import unit
import torch
from typing import Dict
from torch import nn
from torch_scatter import scatter_add


class PhysNetRepresentation(nn.Module):

    def __init__(
        self,
        cutoff: unit = 5 * unit.angstrom,
        number_of_gaussians: int = 16,
        device: torch.device = torch.device("cpu"),
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
        device : torch.device, default=torch.device("cpu")
            The device on which to perform the computations.
        """

        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff, device)

        # radial symmetry function
        from .utils import RadialSymmetryFunction

        self.radial_symmetry_function_module = RadialSymmetryFunction(
            number_of_gaussians=number_of_gaussians,
            radial_cutoff=cutoff,
            ani_style=False,
            dtype=torch.float32,
        )

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the representation module.

        Parameters
        ----------
        d_ij : torch.Tensor
            pairwise distances between atoms, shape (n_pairs, 1).

        Returns
        -------
        torch.Tensor
            The radial basis function expansion applied to the input distances,
            shape (n_pairs, 1, n_gaussians), after applying the cutoff function.
        """

        rbf = self.radial_symmetry_function_module(d_ij).squeeze(1)
        cutoff = self.cutoff_module(d_ij)
        f_ij = torch.mul(rbf, cutoff)
        return f_ij.unsqueeze(1)


class GatingModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Initialize learnable parameters for the gating vector
        self.u = nn.Parameter(torch.randn(feature_dim))

    def forward(self, v):
        # Apply gating to the input vector v
        gated_v = torch.sigmoid(self.u) * v
        return gated_v


class GaussianLogarithmAttention(nn.Module):
    def __init__(self, input_features: int = 1, output_features: int = 64):
        super().__init__()
        # This layer approximates the Gaussian Logarithm function for attention
        self.linear = nn.Linear(input_features, output_features)
        self.activation = nn.Softplus()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # Apply the GloG function
        # distances shape: (nr_of_atoms_in_batch, 1)

        logits = self.linear(distances)
        attention_weights = self.activation(logits)
        return attention_weights


class ResidualBlock(nn.Module):
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

    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):
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


class PhysNetInteraction(nn.Module):

    def __init__(
        self,
        embedding_dimensions: int = 64,
        number_of_atom_basis: int = 16,
    ):
        """
        Defines the interaction module within PhysNet, responsible for capturing the
        effects of atomic interactions through embedded nuclear charges and distance-based
        modulations.

        The module utilizes Gaussian Logarithm Attention to weigh radial basis function (RBF)
        expansions based on pairwise distances, integrating these weights with atomic embeddings
        to generate interaction messages. These messages are aggregated to update atomic feature
        representations, contributing to the model's ability to predict molecular properties.

        Parameters
        ----------
        embedding_dimensions : int, default=64
            Dimensionality of the atomic embeddings.
        number_of_atom_basis : int, default=16
            Specifies the number of basis functions for the Gaussian Logarithm Attention,
            essentially defining the output feature dimension for attention-weighted interactions.

        Attributes
        ----------
        glog_attention : GaussianLogarithmAttention
            The attention mechanism based on pairwise distances between atoms.
        activation_function : ShiftedSoftplus
            Non-linear activation function applied to atomic embeddings.
        interaction_i : nn.Sequential
            A neural network sequence processing features of atom i.
        interaction_j : nn.Sequential
            A neural network sequence processing features of atom j, parallel to interaction_i.
        """

        super().__init__()
        from .utils import ShiftedSoftplus

        self.glog_attention = GaussianLogarithmAttention(
            output_features=number_of_atom_basis
        )
        self.activation_function = ShiftedSoftplus()

        # Networks for processing atomic embeddings of i and j atoms
        self.interaction_i = nn.Sequential(
            nn.Linear(embedding_dimensions, 2 * embedding_dimensions),
            nn.Softplus(),
            nn.Linear(2 * embedding_dimensions, 1),
        )
        self.interaction_j = nn.Sequential(
            nn.Linear(embedding_dimensions, 2 * embedding_dimensions),
            nn.Softplus(),
            nn.Linear(2 * embedding_dimensions, 1),
        )

        self.process_v = nn.Sequential(
            nn.Softplus(),
            nn.Linear(2 * embedding_dimensions, 1),
        )

        # Residual block
        self.residual_block = ResidualBlock(embedding_dimensions, embedding_dimensions)

        # Gating Module
        self.gating_module = GatingModule(embedding_dimensions)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Processes input tensors through the interaction module, applying
        Gaussian Logarithm Attention to modulate the influence of pairwise distances
        on the interaction features, followed by aggregation to update atomic embeddings.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Contains the tensors necessary for interaction calculations:
            - "pair_indices": Indices of atom pairs (shape: [2, n_pairs]).
            - "f_ij": Radial basis function expansions (shape: [n_pairs, n_gaussians]).
            - "d_ij": Pairwise distances between atoms (shape: [n_pairs, 1]).
            - "atomic_embedding": Atomic embeddings (shape: [nr_of_atoms_in_batch, embedding_dim]).

        Returns
        -------
        torch.Tensor
            Updated atomic feature representations incorporating interaction information.
        """

        # extract relevant variables
        idx_i, idx_j = inputs["pair_indices"]  # shape: (2, n_pairs)
        f_ij = inputs["f_ij"].squeeze(1)  # shape: (n_pairs, n_gaussians)
        d_ij = inputs["d_ij"]  # shape: (n_pairs, 1)
        atom_embedding = inputs[
            "atomic_embedding"
        ]  # shape (nr_of_atoms_in_batch, embedding dim)

        # Apply activation to atomic embeddings
        x = self.activation_function(atom_embedding)

        x_i = x[idx_i]  # embedding for atoms i
        x_j = x[idx_j]  # embedding for atoms j

        # Equation 6: Formation of the Proto-Message ṽ_i for an Atom i
        # ṽ_i = σ(Wl_I * x_i^l + bl_I) + Σ_j (G_g * Wl * (σ(σl_J * x_j^l + bl_J)) * g(r_ij))
        # Equation 6 implementation overview:
        # ṽ_i = x_i_prime + sum_over_j(x_j_prime * f_ij_prime)
        # where:
        # - x_i_prime and x_j_prime are the features of atoms i and j, respectively, processed through separate networks.
        # - f_ij_prime represents the modulated radial basis functions (f_ij) by the Gaussian Logarithm Attention weights.

        # Obtain attention mask G(r_ij)
        # Get attention weights from distances
        attention_weights = self.glog_attention(
            d_ij
        )  # shape: (n_pairs, embedding_dim) # NOTE: are we sure about the dimensions?

        # Draft proto message v_tilde
        x_i_prime = self.interaction_i(x_i)
        x_j_prime = self.interaction_j(x_j)
        f_ij_prime = torch.mul(f_ij, attention_weights)

        # Compute sum_over_j(x_j_prime * f_ij_prime)
        x_j_modulated = torch.mul(x_j_prime, f_ij_prime)
        # Aggregate modulated contributions for each atom i
        summed_contributions = scatter_add(
            x_j_modulated, idx_i, dim=0, dim_size=atom_embedding.shape[0]
        )
        v_tilde = x_i_prime + summed_contributions
        # Equation 4: Preactivation Residual Block Implementation
        # xl+2_i = xl_i + Wl+1 * sigma(Wl * xl_i + bl) + bl+1
        v = self.residual_block(v_tilde)
        v = self.self.process_v(v)

        # add gated


class PhysNetResidual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return inputs


class PhysNetOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return inputs


class PhysNetModule(nn.Module):

    def __init__(self):
        """
        Wrapper module that combines the PhysNetInteraction, PhysNetResidual, and
        PhysNetOutput classes into a single module. This serves as the building
        block for the PhysNet model.

        This is a skeletal implementation that needs to be expanded upon.
        """

        super().__init__()

        # this class combines the PhysNetInteraction, PhysNetResidual and
        # PhysNetOutput class

        self.interaction = PhysNetInteraction()
        self.residula = PhysNetResidual()
        self.output = PhysNetOutput()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the PhysNet module. Currently, a placeholder that
        needs further implementation.
        """
        # Placeholder for actual module operations.
        return inputs


class PhysNet(BaseNeuralNetworkPotential):
    def __init__(
        self,
        max_Z: int = 100,
        embedding_dimensions: int = 64,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        nr_of_modules: int = 2,
    ) -> None:
        """
        Implementation of the PhysNet neural network potential.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        embedding_dimensions : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        nr_of_modules : int, default=2
            Number of PhysNet modules to be stacked in the network.
        """

        log.debug("Initializing PhysNET model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.nr_atom_basis = embedding_dimensions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, embedding_dimensions)

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff, self.device)

        # initialize the energy readout
        from .utils import FromAtomToMoleculeReduction

        self.readout_module = FromAtomToMoleculeReduction(embedding_dimensions)

        number_of_gaussian = 16
        self.physnet_representation_module = PhysNetRepresentation(
            cutoff=cutoff, number_of_gaussians=number_of_gaussian
        )

        self.physnet_module_tmp = PhysNetInteraction(
            embedding_dimensions=embedding_dimensions,
            number_of_atom_basis=number_of_gaussian,
        )
        # initialize the PhysNetModule building blocks
        from torch.nn import ModuleList

        self.physnet_module = ModuleList(
            [PhysNetModule() for module_idx in range(nr_of_modules)]
        )

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        # return self.readout_module(inputs)
        return inputs

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        inputs["atomic_embedding"] = self.embedding_module(inputs["atomic_numbers"])
        return inputs

    def _forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.
        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
        - atomic_embedding : torch.Tensor
            Atomic numbers embedding; shape (nr_of_atoms_in_systems, 1, nr_atom_basis).
        - pairlist:  shape (2, n_pairs)
        - d_ij:  shape (n_pairs, 1)
        - atomic_embedding:  shape (nr_of_atoms_in_systems, nr_atom_basis)
        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # Computed representation
        inputs["f_ij"] = self.physnet_representation_module(inputs["d_ij"])
        self.physnet_module_tmp(inputs)
        for module in self.physnet_module:
            output = module(inputs)

        return torch.rand_like(inputs["d_ij"])

        # return self._readout(output)

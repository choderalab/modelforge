from .models import BaseNeuralNetworkPotential
from loguru import logger as log
from openff.units import unit
import torch
from typing import Dict
from torch import nn


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

        rbf = self.radial_symmetry_function_module(d_ij)
        cutoff = self.cutoff_module(d_ij)
        f_ij = torch.mul(rbf, cutoff)
        return f_ij


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


class PhysNetInteraction(nn.Module):

    def __init__(self, number_of_atom_basis: int = 16):
        """
        Interaction module for PhysNet, which is responsible for processing the
        embedded nuclear charges and the expanded radial basis functions.

        This module currently acts as a placeholder to be expanded upon.
        """
        from .utils import ShiftedSoftplus

        super().__init__()
        self.glog_attention = GaussianLogarithmAttention(
            output_features=number_of_atom_basis
        )
        self.activation_function = ShiftedSoftplus()
        self.interaction_i = nn.Sequential(
            nn.Linear(number_of_atom_basis, 2 * number_of_atom_basis),
            nn.Softplus(),
            nn.Linear(2 * number_of_atom_basis, 1),
        )
        self.interaction_j = nn.Sequential(
            nn.Linear(number_of_atom_basis, 2 * number_of_atom_basis),
            nn.Softplus(),
            nn.Linear(2 * number_of_atom_basis, 1),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Forward pass of the interaction module. Currently, a placeholder that
        needs further implementation.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            A dictionary containing the input tensors. Expected tensors include
            "f_ij" for radial basis function expansions, "atomic_embedding" for
            embeddings of atomic numbers, and "pairlist" for the pair lists.

        Returns
        -------
        """

        # extract relevant variables
        idx_i, idx_j = inputs["pairlist"]
        f_ij = inputs["f_ij"]  # shape: (n_pairs, 1, n_gaussians)
        d_ij = inputs["d_ij"]  # shape: (n_pairs, 1)
        atom_embedding = inputs["atomic_embedding"] #shape (nr_of_atoms_in_batch, embedding dim)
        
        x_i = x[idx_i]
        x_j = x[idx_j]

        # Start with embedded features
        x = self.activation_function(atom_embedding)

        # this function implements equation 6 from the PhysNet manuscript
    
        # Equation 6 implementation overview:
        # v_tilde_i = x_i_prime + sum_over_j(x_j_prime * f_ij_prime)
        # where:
        # - x_i_prime and x_j_prime are the features of atoms i and j, respectively, processed through separate networks.
        # - f_ij_prime represents the modulated radial basis functions (f_ij) by the Gaussian Logarithm Attention weights.

        # Obtain attention mask G(r_ij)
        # Get attention weights from distances
        attention_weights = self.glog_attention(d_ij)

        # Draft proto message v_tilde
        # x_i_prime
        x_i_prime = self.interaction_i(x_i)
        # x_j_prime
        x_j_prime = self.interaction_j(x_j)
        # f_ij_prime
        f_ij_prime = torch.mul(f_ij, attention_weights)

        # Compute the interaction for x_j
        x_j_prime = torch.mul(x_j_prime, f_ij_prime)

        # gating vectors
        # u =


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

        self.physnet_representation_module = PhysNetRepresentation(
            cutoff=cutoff, number_of_gaussians=16
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
        representation = self.physnet_representation_module(inputs["d_ij"])
        inputs["representation"] = representation

        for module in self.physnet_module:
            output = module(inputs)

        return torch.rand_like(inputs["d_ij"])

        # return self._readout(output)

""" 
This module contains the dimenet++ implementation based on
"Directional Message Passing for Molecular Graphs" (ICLR 2020) 
and "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" (NeurIPS-W 2020) 
"""

import torch
import torch.nn as nn
from loguru import logger as log

from typing import Dict, List

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.neighbors import PairlistData


class EmbeddingBlock(nn.Module):
    """
    Embedding block for the DimeNet++ model.

    Parameters
    ----------
    embedding_size : int
        Embedding size.
    activation_function : torch.nn.Module

    Notes
    -----
    This module computes the embedding for atom pairs based on their atomic
    numbers and radial basis functions. It uses trainable embeddings for atomic
    numbers up to 94 (Plutonium) and applies two dense layers.
    """

    def __init__(
        self,
        embedding_size: int,
        number_of_radial_bessel_functions: int,
        activation_function: torch.nn.Module,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        import math
        from modelforge.potential.utils import Dense

        num_embeddings = 95  # Elements up to atomic number 94 (Pu)

        # Initialize embeddings with Uniform(-sqrt(3), sqrt(3))
        self.embeddings = nn.Embedding(num_embeddings, embedding_size)
        emb_init_range = math.sqrt(3)
        nn.init.uniform_(self.embeddings.weight, -emb_init_range, emb_init_range)

        # Dense layer for radial basis functions
        self.dense_rbf = Dense(
            number_of_radial_bessel_functions,
            embedding_size,
            bias=True,
            activation_function=activation_function,
        )

        # Final dense layer
        self.dense = Dense(
            3 * embedding_size,
            embedding_size,
            bias=True,
            activation_function=activation_function,
        )

    def forward(
        self,
        inputs: NNPInput,
        pairlist_output: PairlistData,
        f_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the EmbeddingBlock.

        Parameters
        ----------
        inputs : NNPInput
            Input data including atomic numbers, positions, etc.
        pairlist_output : PairlistData
            Output from the pairlist module, containing pair indices and
            distances.
        f_ij : torch.Tensor
        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (nr_of_pairs, emb_size).
        """

        # Transform radial basis functions
        # rbf: (nr_of_pairs, num_radial) -> (nr_of_pairs, emb_size)
        rbf = self.activation(self.dense_rbf(f_ij))

        # Gather atomic numbers for neighbor pairs
        # Z_i and Z_j have shape (nr_of_pairs)
        Z_i = inputs.atomic_numbers[pairlist_output.pair_indices[0]]
        Z_j = inputs.atomic_numbers[pairlist_output.pair_indices[1]]

        # Get embeddings for atomic numbers
        # x_i and x_j have shape (E, emb_size)
        x_i = self.embeddings(Z_i)
        x_j = self.embeddings(Z_j)

        # Concatenate embeddings and transformed rbf
        # x has shape (E, 3 * emb_size)
        x = torch.cat([x_i, x_j, rbf], dim=-1)

        # Final transformation
        # x: (E, 3 * emb_size) -> (E, emb_size)
        x = self.dense(x)

        return x


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff.
    """

    def __init__(self, exponent: int):
        super().__init__()
        self.exponent = exponent

        # Precompute constants
        p = torch.tensor(exponent + 1, dtype=torch.int32)
        self.register_buffer("p", p)
        self.register_buffer("a", -((p + 1) * (p + 2)) / 2)
        self.register_buffer("b", p * (p + 2))
        self.register_buffer("c", -p * (p + 1) / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Compute powers efficiently
        inputs_p_minus1 = torch.pow(inputs, self.p - 1)
        inputs_p = inputs_p_minus1 * inputs  # inputs ** self.p
        inputs_p_plus1 = inputs_p * inputs  # inputs ** (self.p + 1)

        # Envelope function divided by r
        env_val = (
            (1.0 / inputs)
            + self.a * inputs_p_minus1
            + self.b * inputs_p
            + self.c * inputs_p_plus1
        )

        # Apply cutoff
        env_val = torch.where(inputs < 1.0, env_val, torch.zeros_like(env_val))

        return env_val


class BesselBasisLayer(nn.Module):
    """
    Bessel Basis Layer as used in DimeNet++.
    """

    def __init__(
        self,
        number_of_radial_bessel_functions: int,
        radial_cutoff: float,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        self.number_of_radial_bessel_functions = number_of_radial_bessel_functions
        self.register_buffer(
            "inv_cutoff", torch.tensor(1.0 / radial_cutoff, dtype=torch.float32)
        )
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        frequencies = torch.pi * torch.arange(
            1, number_of_radial_bessel_functions + 1, dtype=torch.float32
        )
        self.frequencies = nn.Parameter(frequencies)  # Trainable parameter

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        # d_ij: Pairwise distances between atoms. Shape: (nr_pairs, 1)

        # Scale distances
        d_scaled = d_ij * self.inv_cutoff  # Shape: (nr_pairs, 1)

        # Apply envelope
        d_cutoff = self.envelope(d_scaled)  # Shape: (nr_pairs, 1)

        # Compute Bessel basis
        basis = d_cutoff * torch.sin(
            self.frequencies * d_scaled
        )  # Shape: nr_pairs, num_radial)
        return basis


class DimeNetCore(torch.nn.Module):
    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_blocks: int,
        dimension_of_bilinear_layer: int,
        number_of_spherical_harmonics: int,
        number_of_radial_bessel_functions: int,
        maximum_interaction_radius: float,
        envelope_exponent: int,
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

        log.debug("Initializing the DimeNet architecture.")

        self.representation_module = Representation(
            number_of_radial_bessel_functions=number_of_radial_bessel_functions,
            radial_cutoff=maximum_interaction_radius,
            number_of_spherical_harmonics=number_of_spherical_harmonics,
            envelope_exponent=envelope_exponent,
            activation_function=self.activation_function,
            embedding_size=32,
        )

    def compute_properties(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties based on the input data and pair list.

        Parameters
        ----------
        data : NNPInput
            Input data including atomic numbers, positions, etc.
        pairlist_output: PairlistData
            Output from the pairlist module, containing pair indices and
            distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the computed properties for each atom.
        """

        # Compute the atomic representation, which includes
        # - radial/angular bessel function
        # - embedding of pairwise distances

        representation = self.representation_module(
            data, pairlist_output
        )  # includes 'm_ij', 'radial_bessel', 'angular_bessel'

        # Apply interaction modules to update the atomic embedding

        return None

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the DimeNet model.

        Parameters
        ----------
        data : NNPInput
            Input data including atomic numbers, positions, and relevant fields.
        pairlist_output : PairlistData
            Pair indices and distances from the pairlist module.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of calculated properties from the forward pass.
        """
        # Compute properties using the core method
        results = self.compute_properties(data, pairlist_output)
        atomic_embedding = results["per_atom_scalar_representation"]

        # Apply output layers to the atomic embedding
        for output_name, output_layer in self.output_layers.items():
            results[output_name] = output_layer(atomic_embedding).squeeze(-1)

        return results


class Representation(nn.Module):

    def __init__(
        self,
        radial_cutoff: float,
        number_of_radial_bessel_functions: int,
        number_of_spherical_harmonics: int,
        envelope_exponent: int,
        activation_function: torch.nn.Module,
        embedding_size: int,
    ):
        """
        Initialize the representation module.
        """
        super().__init__()

        # The representation part of DimeNet++ includes
        # - radial bessel basis (invariant representation/featurization of
        #   distances)
        # - angular bessel basis (equivariant representation/featurization of
        #   pairwise direction (distance vector))
        # - embedding of pairwise distances
        self.radial_bessel_function = self._setup_radial_bessel_basis(
            radial_cutoff,
            number_of_radial_bessel_functions,
            envelope_exponent,
        )

        self.angular_bessel_function = self._setup_angular_bessel_basis(
            number_of_spherical_harmonics,
            envelope_exponent,
        )

        self.embedding = EmbeddingBlock(
            embedding_size=embedding_size,
            number_of_radial_bessel_functions=number_of_radial_bessel_functions,
            activation_function=activation_function,
        )

    def _setup_radial_bessel_basis(
        self,
        radial_cutoff: float,
        number_of_radial_bessel_functions: int,
        envelope_exponent: int,
    ):
        radial_symmetry_function = BesselBasisLayer(
            number_of_radial_bessel_functions=number_of_radial_bessel_functions,
            radial_cutoff=radial_cutoff,
            envelope_exponent=envelope_exponent,
        )
        return radial_symmetry_function

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate the radial symmetry representation of pairwise
        distances.

        Parameters
        ----------
        data : NNPInput
            Input data containing atomic numbers and positions.
        pairlist_output : PairlistData
            Output from the pairlist module, containing pair indices and distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing radial/angular bessel basis and first message.
        """

        # Convert distances to radial bessel functions
        radial_bessel = self.radial_bessel_function(pairlist_output.d_ij)

        # convert distances to angular bessel functions
        angular_bessel = self.angular_bessel_function(
            pairlist_output.d_ij, pairlist_output.d_ij
        )

        # generate first message
        m_ij = self.embedding(data, pairlist_output, radial_bessel)

        return {
            "m_ij": m_ij,
            "radial_bessel": radial_bessel,
            "angular_bessel": angular_bessel,
        }

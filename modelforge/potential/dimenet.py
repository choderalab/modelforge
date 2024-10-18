""" 
This module contains the dimenet++ implementation based on
"Directional Message Passing for Molecular Graphs" (ICLR 2020) 
and "Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules" (NeurIPS-W 2020) 
"""

import torch
import torch.nn as nn
from loguru import logger as log

from typing import Dict, List

from modelforge.dataset.dataset import NNPInputTuple
from modelforge.potential.neighbors import PairlistData


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
        from modelforge.potential.utils import Dense

        self.representation_module = Representation(...)

    def compute_properties(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties based on the input data and pair list.

        Parameters
        ----------
        data : NNPInputTuple
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
        # - radial bessel basis
        # - spherical harmonics

        representation = self.representation_module(data, pairlist_output)

        # embedding block
        embedding = self.embedding_block(
            data.atomic_numbers,
            representation["radial_bessel_basis"],
            pairlist_output.pair_indices,
        )

        # Apply interaction modules to update the atomic embedding

        return {
            "per_atom_scalar_representation": None,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the DimeNet model.

        Parameters
        ----------
        data : NNPInputTuple
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
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        """
        SchNet representation module to generate the radial symmetry
        representation of pairwise distances.

        Parameters
        ----------
        radial_cutoff : float
            The cutoff distance for interactions in nanometer.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        featurization_config : Dict[str, Dict[str, int]]
            Configuration for atom featurization.
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        # Initialize cutoff module
        from modelforge.potential import CosineAttenuationFunction, FeaturizeInput

        self.featurize_input = FeaturizeInput(featurization_config)
        self.cutoff_module = CosineAttenuationFunction(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: float, number_of_radial_basis_functions: int
    ):
        from modelforge.potential import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate the radial symmetry representation of pairwise
        distances.

        Parameters
        ----------
        data : NNPInputTuple
            Input data containing atomic numbers and positions.
        pairlist_output : PairlistData
            Output from the pairlist module, containing pair indices and distances.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing radial basis functions, cutoff values, and atomic embeddings.
        """

        # Convert distances to radial bessel functions
        f_ij = self.radial_bessel_function(pairlist_output.d_ij)
        # generate triple indices
        triple_indices = self.calculate_triplets(
            pairlist_output
        )  # Shape: [number_of_triplets (number_of_triples<<number_of_atoms^3), 3]

        sbf = self.spherical_bessel_function()

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(data),
        }

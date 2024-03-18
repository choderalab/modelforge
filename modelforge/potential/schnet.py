from typing import Dict

import torch
from loguru import logger as log
import torch.nn as nn

from .models import BaseNeuralNetworkPotential
from openff.units import unit


class SchNet(BaseNeuralNetworkPotential):
    def __init__(
        self,
        max_Z: int = 100,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 16,
        nr_interaction_modules: int = 2,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        nr_filters: int = None,
        shared_interactions: bool = False,
    ) -> None:
        """
        Initialize the SchNet class.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        nr_interaction_modules : int, default=2
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """
        from .utils import ShiftedSoftplus, Dense

        log.debug("Initializing SchNet model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.number_of_atom_features = number_of_atom_features
        self.nr_filters = nr_filters or self.number_of_atom_features
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, number_of_atom_features)

        # initialize the energy readout
        from .utils import FromAtomToMoleculeReduction

        self.readout_module = FromAtomToMoleculeReduction()

        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            cutoff, number_of_radial_basis_functions, self.device
        )
        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SchNETInteractionModule(
                    self.number_of_atom_features,
                    self.nr_filters,
                    number_of_radial_basis_functions,
                )
                for _ in range(nr_interaction_modules)
            ]
        )

        # final output layer
        self.energy_layer = nn.Sequential(
            Dense(
                number_of_atom_features,
                number_of_atom_features,
                activation=ShiftedSoftplus(),
            ),
            Dense(
                number_of_atom_features,
                1,
            ),
        )

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
        - pairlist:  shape (n_pairs, 2)
        - r_ij:  shape (n_pairs, 3)
        - d_ij:  shape (n_pairs, 1)
        - positions:  shape (nr_of_atoms_per_molecules, 3)
        - atomic_embedding:  shape (nr_of_atoms_in_systems, nr_atom_basis)


        Returns
        -------
        torch.Tensor
            Calculated energies; shape (nr_systems,).
        """

        # Compute the representation for each atom (transform to radial basis set, multiply by cutoff)
        representation = self.schnet_representation_module(inputs["d_ij"])
        x = inputs["atomic_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                x,
                inputs["pair_indices"],
                representation["f_ij"],
                representation["f_cutoff"],
            )
            x = x + v  # Update atomic features

        E_i = self.energy_layer(x).squeeze(1)

        return {
            "E_i": E_i,
            "q": x,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


class SchNETInteractionModule(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int,
        nr_filters: int,
        number_of_radial_basis_functions: int,
    ) -> None:
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        number_of_atom_features : int
            Number of atom ffeatures, defines the dimensionality of the embedding.
        nr_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        """
        super().__init__()
        from .utils import ShiftedSoftplus, Dense

        assert (
            number_of_radial_basis_functions > 4
        ), "Number of radial basis functions must be larger than 10."
        assert nr_filters > 1, "Number of filters must be larger than 1."
        assert (
            number_of_atom_features > 10
        ), "Number of atom basis must be larger than 10."

        self.number_of_atom_features = number_of_atom_features  # Initialize parameters
        self.intput_to_feature = Dense(
            number_of_atom_features, nr_filters, bias=False, activation=None
        )
        self.feature_to_output = nn.Sequential(
            Dense(nr_filters, number_of_atom_features, activation=ShiftedSoftplus()),
            Dense(number_of_atom_features, number_of_atom_features, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(
                number_of_radial_basis_functions,
                nr_filters,
                activation=ShiftedSoftplus(),
            ),
            Dense(nr_filters, nr_filters, activation=None),
        )

    def forward(
        self,
        x: torch.Tensor,
        pairlist: torch.Tensor,  # shape [n_pairs, 2]
        f_ij: torch.Tensor,  # shape [n_pairs, 1, number_of_radial_basis_functions]
        f_ij_cutoff: torch.Tensor,  # shape [n_pairs, 1]
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Input feature tensor for atoms.
        pairlist : torch.Tensor, shape [n_pairs, 2]
        f_ij : torch.Tensor, shape [n_pairs, 1, number_of_radial_basis_functions]
            Radial basis functions for pairs of atoms.
        f_ij_cutoff : torch.Tensor, shape [n_pairs, 1]

        Returns
        -------
        torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Updated feature tensor after interaction block.
        """

        # Map input features to the filter space
        x = self.intput_to_feature(x)

        # Generate interaction filters based on radial basis functions
        Wij = self.filter_network(f_ij.squeeze(1))

        idx_i, idx_j = pairlist[0], pairlist[1]
        x_j = x[idx_j]

        # Perform continuous-filter convolution
        x_ij = x_j * Wij * f_ij_cutoff

        # Initialize a tensor to gather the results
        x_ = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # Sum contributions to update atom features
        # x shape: torch.Size([nr_of_atoms_in_batch, 64])
        # x_ij shape: torch.Size([nr_of_pairs, 64])
        idx_i_expand = idx_i.unsqueeze(1).expand_as(x_ij)
        x_.scatter_add_(0, idx_i_expand, x_ij)
        # Map back to the original feature space and reshape
        x = self.feature_to_output(x_)
        return x


class SchNETRepresentation(nn.Module):
    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_radial_basis_functions: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the SchNet representation layer.

        Parameters
        ----------
        Radial Basis Function Module
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        self.device = device
        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(radial_cutoff, self.device)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: unit.Quantity, number_of_radial_basis_functions: int
    ):
        from .utils import SchnetRadialSymmetryFunction

        radial_symmetry_function = SchnetRadialSymmetryFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, d_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        d_ij : Pairwise distances between atoms; shape [n_pairs, 1]

        Returns
        -------
        Radial basis functions for pairs of atoms; shape [n_pairs, 1, number_of_radial_basis_functions]
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(
            d_ij
        )  # shape (n_pairs, 1, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(d_ij)  # shape (n_pairs, 1)

        return {"f_ij": f_ij, "f_cutoff": f_cutoff}

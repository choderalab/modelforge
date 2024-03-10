from typing import Dict

import torch
from loguru import logger as log
import torch.nn as nn

from .models import BaseNNP
from openff.units import unit


class SchNET(BaseNNP):
    def __init__(
        self,
        max_Z: int = 100,
        embedding_dimensions: int = 64,
        nr_interaction_blocks: int = 2,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        number_of_gaussians_basis_functions: int = 16,
        nr_filters: int = None,
        shared_interactions: bool = False,
    ) -> None:
        """
        Initialize the SchNet class.

        Parameters
        ----------
        embedding : nn.Module
        nr_interaction_blocks : int
            Number of interaction blocks in the architecture.
        radial_symmetry_function_module : nn.Module
        cutoff : nn.Module
        nr_filters : int, optional
            Number of filters; defines the dimensionality of the intermediate features.
        """

        log.debug("Initializing SchNet model.")

        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.nr_atom_basis = embedding_dimensions
        self.nr_filters = nr_filters or self.nr_atom_basis
        self.number_of_gaussians_basis_functions = number_of_gaussians_basis_functions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, embedding_dimensions)

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff, self.device)

        # initialize the energy readout
        from .utils import EnergyReadout

        self.readout_module = EnergyReadout(self.nr_atom_basis)

        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            cutoff, number_of_gaussians_basis_functions, self.device
        )
        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SchNETInteractionBlock(
                    self.nr_atom_basis,
                    self.nr_filters,
                    number_of_gaussians_basis_functions,
                )
                for _ in range(nr_interaction_blocks)
            ]
        )

    def _readout(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the energy for each system
        return self.readout_module(inputs)

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

        # Compute the representation for each atom
        representation = self.schnet_representation_module(inputs["d_ij"])
        x = inputs["atomic_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                x,
                inputs["pair_indices"],
                representation["f_ij"],
                representation["rcut_ij"],
            )
            x = x + v  # Update atomic features

        return {
            "scalar_representation": x,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


class SchNETInteractionBlock(nn.Module):
    def __init__(
        self, nr_atom_basis: int, nr_filters: int, number_of_gaussians: int
    ) -> None:
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        nr_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        nr_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        number_of_gaussians : int
            Number of radial basis functions.
        """
        super().__init__()
        from .utils import ShiftedSoftplus, Dense

        assert (
            number_of_gaussians > 4
        ), "Number of radial basis functions must be larger than 10."
        assert nr_filters > 1, "Number of filters must be larger than 1."
        assert nr_atom_basis > 10, "Number of atom basis must be larger than 10."

        self.nr_atom_basis = nr_atom_basis  # Initialize parameters
        self.intput_to_feature = Dense(
            nr_atom_basis, nr_filters, bias=False, activation=None
        )
        self.feature_to_output = nn.Sequential(
            Dense(nr_filters, nr_atom_basis, activation=ShiftedSoftplus()),
            Dense(nr_atom_basis, nr_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(number_of_gaussians, nr_filters, activation=ShiftedSoftplus()),
            Dense(nr_filters, nr_filters, activation=None),
        )

    def forward(
        self,
        x: torch.Tensor,
        pairlist: torch.Tensor,  # shape [n_pairs, 2]
        f_ij: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Input feature tensor for atoms.
        pairlist : torch.Tensor, shape [n_pairs, 2]
        f_ij : torch.Tensor, shape [n_pairs, number_of_gaussians]
            Radial basis functions for pairs of atoms.
        rcut_ij : torch.Tensor, shape [n_pairs]
            Cutoff values for each pair.

        Returns
        -------
        torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Updated feature tensor after interaction block.
        """

        # Map input features to the filter space
        x = self.intput_to_feature(x)  # (nr_of_atoms_in_systems, n_filters)

        # Generate interaction filters based on radial basis functions
        Wij = self.filter_network(f_ij)  # (n_pairs, n_filters)
        Wij = Wij * rcut_ij[:, None]  # Apply the cutoff
        # Wij = Wij.to(dtype=x.dtype)

        idx_i, idx_j = pairlist[0], pairlist[1]
        x_j = x[idx_j]

        # Perform continuous-filter convolution
        x_ij = x_j * Wij  # shape (n_pairs, nr_filters)

        # Initialize a tensor to gather the results
        x = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # Sum contributions to update atom features
        x.scatter_add_(0, idx_i.unsqueeze(1).expand_as(x_ij), x_ij)

        # Map back to the original feature space and reshape
        x = self.feature_to_output(x)
        return x


class SchNETRepresentation(nn.Module):
    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_gaussians: int,
        device: torch.device,
    ):
        """
        Initialize the SchNet representation layer.

        Parameters
        ----------
        Radial Basis Function Module
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_gaussians
        )
        self.device = device

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: unit.Quantity, number_of_gaussians: int
    ):
        from .utils import RadialSymmetryFunction

        radial_symmetry_function = RadialSymmetryFunction(
            number_of_gaussians=number_of_gaussians,
            radial_cutoff=radial_cutoff,
            ani_style=False,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, d_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
        d_ij : Dict[str, torch.Tensor], Pairwise distances between atoms; shape [n_pairs, 1]

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'f_ij': Radial basis functions for pairs of atoms; shape [n_pairs, number_of_gaussians]
            - 'rcut_ij': Cutoff values for each pair; shape [n_pairs]
        """
        from modelforge.potential.utils import CosineCutoff

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(d_ij).squeeze(1)
        cutoff_module = CosineCutoff(
            self.radial_symmetry_function_module.radial_cutoff, device=d_ij.device

        )

        rcut_ij = cutoff_module(d_ij).squeeze(1)
        return {"f_ij": f_ij, "rcut_ij": rcut_ij}

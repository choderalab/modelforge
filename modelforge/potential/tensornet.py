from dataclasses import dataclass

import torch
from openff.units import unit

from modelforge.potential.models import InputPreparation
from modelforge.potential.models import BaseNetwork
from modelforge.potential.utils import NeuralNetworkData


class TensorNet(BaseNetwork):
    def __init__(
        self,
        radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
        radial_min_distanc: unit.Quantity = 0.0 * unit.angstrom,
        number_of_radial_basis_functions: int = 16,
    ) -> None:

        super().__init__()

        self.core_module = TensorNetCore(
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
        )
        self.only_unique_pairs = True  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=radial_max_distance, only_unique_pairs=self.only_unique_pairs
        )


class TensorNetCore(torch.nn.Module):
    def __init__(
        self,
        radial_max_distance: unit.Quantity,
        radial_min_distanc: unit.Quantity,
        number_of_radial_basis_functions: int,
    ):
        super().__init__()

        # Initialize representation block
        self.tensornet_representation_module = TensorNetRepresentation(
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
        )

        self.interaction_modules = ANIInteraction()

    def _forward(self):
        pass

    def _model_specific_input_preparation(self):
        pass

    def forward(self):
        pass


class TensorNetRepresentation(torch.nn.Module):
    def __init__(
        self,
        radial_max_distance,
        radial_min_distanc,
        number_of_radial_basis_functions,
    ):
        pass


class ANIInteraction(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


@dataclass
class TensorNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs for ANI neural network potentials, designed to
    facilitate the efficient representation of atomic systems for energy computation and
    property prediction.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A 2D tensor indicating the indices of atom pairs. Shape: [2, num_pairs].
    d_ij : torch.Tensor
        A 1D tensor containing distances between each pair of atoms. Shape: [num_pairs, 1].
    r_ij : torch.Tensor
        A 2D tensor representing displacement vectors between atom pairs. Shape: [num_pairs, 3].
    number_of_atoms : int
        An integer indicating the number of atoms in the batch.
    positions : torch.Tensor
        A 2D tensor representing the XYZ coordinates of each atom. Shape: [num_atoms, 3].
    atom_index : torch.Tensor
        A 1D tensor containing atomic numbers for each atom in the system(s). Shape: [num_atoms].
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule. Shape: [num_atoms].
    total_charge : torch.Tensor
        An tensor with the total charge of each system or molecule. Shape: [num_systems].
    atomic_numbers : torch.Tensor
        A 1D tensor containing the atomic numbers for atoms, used for identifying the atom types within the model. Shape: [num_atoms].

    """

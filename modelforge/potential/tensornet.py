from dataclasses import dataclass

import torch
from openff.units import unit
from torch import nn, Tensor
from typing import Optional, Tuple

from modelforge.potential.models import InputPreparation
from modelforge.potential.models import BaseNetwork
from modelforge.potential.models import CoreNetwork
from modelforge.potential.utils import CosineCutoff
from modelforge.potential.utils import TensorNetRadialSymmetryFunction
from modelforge.potential.utils import NeuralNetworkData
from modelforge.potential.utils import NNPInput

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
    pass

class TensorNet(BaseNetwork):
    def __init__(
        self,
        hidden_channels: int = 8,
        radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
        radial_min_distanc: unit.Quantity = 0.0 * unit.angstrom,
        number_of_radial_basis_functions: int = 16,
        activation_function: nn.Module = nn.SiLU,
        trainable_rbf: bool = False,
        max_z: int = 128,
        dtype: torch.dtype = torch.float32,
    ) -> None:

        super().__init__()

        self.core_module = TensorNetCore(
            hidden_channels,
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
            activation_function,
            trainable_rbf,
            max_z,
            dtype,
        )
        self.only_unique_pairs = True  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=radial_max_distance, only_unique_pairs=self.only_unique_pairs
        )


class TensorNetCore(CoreNetwork):
    def __init__(
        self,
        hidden_channels: int,
        radial_max_distance: unit.Quantity,
        radial_min_distanc: unit.Quantity,
        number_of_radial_basis_functions: int,
        activation_function: nn.Module,
        trainable_rbf: bool,
        max_z: int,
        dtype: torch.dtype,
    ):
        super().__init__()

        # Initialize representation block
        self.tensornet_representation_module = TensorNetRepresentation(
            hidden_channels,
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
            activation_function,
            trainable_rbf,
            max_z,
            dtype,
        )

        self.interaction_modules = TensorNetInteraction()

    def _forward(self):
        pass

    def _model_specific_input_preparation(self):
        pass

    def forward(self):
        pass


class TensorNetRepresentation(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        radial_max_distance: unit.Quantity,
        radial_min_distance: unit.Quantity,
        number_of_radial_basis_functions: int,
        activation_function: nn.Module,
        trainable_rbf: bool,
        max_z: int,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.cutoff_module = CosineCutoff(radial_max_distance)
        self.radial_symmetry_function = self._setup_radial_symmetry_functions(
            radial_max_distance, 
            radial_min_distance, 
            number_of_radial_basis_functions
        )
        self.hidden_channels = hidden_channels
        self. distance_proj1 = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self. distance_proj2 = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self. distance_proj3 = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.emb2 = nn.Embedding(2 * hidden_channels, hidden_channels, dtype=dtype)
        self.act = activation_function()
        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _get_atomic_number_message(self, z: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(z)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
                -1, self.hidden_channels * 2
            )
        )[..., None, None]
        return Zij

    def _get_tensor_messages(
        self, Zij: Tensor, edge_weight: Tensor, edge_vec_norm: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        C = self.cutoff(edge_weight).reshape(-1, 1, 1, 1) * Zij
        eye = torch.eye(3, 3, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)[
            None, None, ...
        ]
        Iij = self.distance_proj1(edge_attr)[..., None, None] * C * eye
        Aij = (
            self.distance_proj2(edge_attr)[..., None, None]
            * C
            * vector_to_skewtensor(edge_vec_norm)[..., None, :, :]
        )
        Sij = (
            self.distance_proj3(edge_attr)[..., None, None]
            * C
            * vector_to_symtensor(edge_vec_norm)[..., None, :, :]
        )
        return Iij, Aij, Sij

    def _setup_radial_symmetry_functions(
        self,
        max_distance: unit.Quantity,
        min_distance: unit.Quantity,
        number_of_radial_basis_functions: int,
    ):
        radial_symmetry_function = TensorNetRadialSymmetryFunction(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, data: TensorNetNeuralNetworkData):
        radial_feature_vector = self.radial_symmetry_function(data.d_ij)
        # cutoff
        rcut_ij = self.cutoff_module(data.d_ij)
        radial_feature_vector = radial_feature_vector * rcut_ij


class TensorNetInteraction(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass



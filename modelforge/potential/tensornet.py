from dataclasses import dataclass

import torch
from openff.units import unit
from torch import nn, Tensor
from torchmdnet.models.tensornet import tensor_norm, vector_to_skewtensor, vector_to_symtensor
from torchmdnet.models.utils import OptimizedDistance
from typing import Optional, Tuple

from modelforge.potential.models import InputPreparation
from modelforge.potential.models import BaseNetwork
from modelforge.potential.models import CoreNetwork
from modelforge.potential.utils import CosineCutoff
from modelforge.potential.utils import TensorNetRadialSymmetryFunction
from modelforge.potential.utils import NeuralNetworkData
from modelforge.potential.utils import NNPInput
from .models import PairListOutputs

ATOMIC_NUMBER_TO_INDEX_MAP = {
    1: 0,  # H
    6: 1,  # C
    7: 2,  # N
    8: 3,  # O
    9: 4,  # F
    16: 5,  # S
    17: 6,  # Cl
}


@dataclass
class TensorNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs for TensorNet neural network potentials, designed to
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
    """
    pass


class TensorNet(BaseNetwork):
    def __init__(
            self,
            # hidden_channels: int = 8,
            radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
            radial_min_distanc: unit.Quantity = 0.0 * unit.angstrom,
            number_of_radial_basis_functions: int = 16,
            # activation_function: nn.Module = nn.SiLU,
            # trainable_rbf: bool = False,
            # max_z: int = 128,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.core_module = TensorNetCore(
            # hidden_channels,
            radial_max_distance,
            radial_min_distanc,
            number_of_radial_basis_functions,
            # activation_function,
            # trainable_rbf,
            # -64,
            # max_z,
            dtype,
        )
        self.only_unique_pairs = True  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=radial_max_distance, only_unique_pairs=self.only_unique_pairs
        )


class TensorNetCore(CoreNetwork):
    def __init__(
            self,
            # hidden_channels: int,
            radial_max_distance: unit.Quantity,
            radial_min_distance: unit.Quantity,
            number_of_radial_basis_functions: int,
            # number_of_layers: int,
            # activation_function: nn.Module,
            # trainable_rbf: bool,
            # max_number_of_neighbors: int,
            # max_z: int,
            # static_shapes: bool = True,
            # check_errors: bool = True,
            dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # self.static_shapes = static_shapes

        # # TensorNet uses angstrom
        # _max_distance_in_angstrom = radial_max_distance.to(unit.angstrom).m
        # _min_distance_in_angstrom = radial_min_distance.to(unit.angstrom).m

        # self.distance = OptimizedDistance(
        #     _min_distance_in_angstrom,
        #     _max_distance_in_angstrom,
        #     max_num_pairs=-max_number_of_neighbors,
        #     return_vecs=True,
        #     loop=True,
        #     check_errors=check_errors,
        #     resize_to_fit=not self.static_shapes,
        #     box=None,
        #     long_edge_index=True,
        # )

        # # Initialize representation block
        # self.tensornet_representation_module = TensorNetRepresentation(
        #     hidden_channels,
        #     radial_max_distance,
        #     radial_min_distance,
        #     number_of_radial_basis_functions,
        #     activation_function,
        #     trainable_rbf,
        #     max_z,
        #     dtype,
        # )

        # self.interaction_modules = TensorNetInteraction()

        # max_atomic_number = max(ATOMIC_NUMBER_TO_INDEX_MAP.keys())
        # lookup_tensor = torch.full((max_atomic_number + 1,), -1, dtype=torch.long)

        # # Populate the lookup tensor with indices from your map
        # for atomic_number, index in ATOMIC_NUMBER_TO_INDEX_MAP.items():
        #     lookup_tensor[atomic_number] = index

        # self.register_buffer("lookup_tensor", lookup_tensor)

    def _forward(self):
        pass

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> TensorNetNeuralNetworkData:
        number_of_atoms = data.atomic_numbers.shape[0]

        # Where should $box and $batch be input???
        # test this separately: pair_indices, d_ij, r_ij
        # edge_index, edge_weight, edge_vec = self.distance(
        #     data.positions,
        #     data.atomic_subsystem_indices,
        #     box,
        # )

        nnpdata = TensorNetNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
        )

        # return nnpdata
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
        self.distance_proj1 = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.distance_proj2 = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.distance_proj3 = nn.Linear(
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
        Zij = self._get_atomic_number_message(data.atomic_numbers, data.edge_index)
        # Normalizing edge vectors by their length can result in NaNs, breaking Autograd.
        # I avoid dividing by zero by setting the weight of self edges and self loops to 1
        mask = data.edge_index[0] == data.edge_index[1]
        edge_vec_norm = data.edge_vec / data.edge_weight.masked_fill(mask, 1).unsqueeze(1)

        radial_feature_vector = self.radial_symmetry_function(data.edge_weight)
        # cutoff
        rcut_ij = self.cutoff_module(data.edge_weight)
        radial_feature_vector = radial_feature_vector * rcut_ij

        Iij, Aij, Sij = self._get_tensor_messages(
            Zij, data.edge_weight, edge_vec_norm, radial_feature_vector
        )
        source = torch.zeros(
            data.atomic_numbers.shape[0], self.hidden_channels, 3, 3, device=data.atomic_numbers.device, dtype=Iij.dtype
        )
        I = source.index_add(dim=0, index=data.edge_index[0], source=Iij)
        A = source.index_add(dim=0, index=data.edge_index[0], source=Aij)
        S = source.index_add(dim=0, index=data.edge_index[0], source=Sij)
        norm = self.init_norm(tensor_norm(I + A + S))
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, self.hidden_channels, 3)
        I = (
                self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                * norm[..., 0, None, None]
        )
        A = (
                self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                * norm[..., 1, None, None]
        )
        S = (
                self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                * norm[..., 2, None, None]
        )
        X = I + A + S
        return X


class TensorNetInteraction(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

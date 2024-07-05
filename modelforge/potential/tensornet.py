from dataclasses import dataclass

import torch
from openff.units import unit
from torch import nn
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
            hidden_channels: int = 8,
            number_of_radial_basis_functions: int = 16,
            activation_function: nn.Module = nn.SiLU,
            radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
            radial_min_distance: unit.Quantity = 0.0 * unit.angstrom,
            trainable_rbf: bool = False,
            max_atomic_number: int = 128,
            dtype: torch.dtype = torch.float32,
            representation_unit: unit.Quantity = unit.angstrom,
    ) -> None:
        super().__init__()

        self.core_module = TensorNetCore(
            hidden_channels,
            number_of_radial_basis_functions,
            activation_function,
            radial_max_distance,
            radial_min_distance,
            trainable_rbf,
            max_atomic_number,
            dtype,
            representation_unit,
        )
        self.only_unique_pairs = False
        self.input_preparation = InputPreparation(
            cutoff=radial_max_distance, only_unique_pairs=self.only_unique_pairs
        )


class TensorNetCore(CoreNetwork):
    def __init__(
            self,
            hidden_channels: int,
            number_of_radial_basis_functions: int,
            activation_function: nn.Module,
            radial_max_distance: unit.Quantity,
            radial_min_distance: unit.Quantity,
            trainable_rbf: bool,
            max_atomic_number: int,
            dtype: torch.dtype,
            representation_unit: unit.Quantity,
    ):
        super().__init__()

        self.tensornet_representation_module = TensorNetRepresentation(
            hidden_channels,
            number_of_radial_basis_functions,
            activation_function,
            radial_max_distance,
            radial_min_distance,
            trainable_rbf,
            max_atomic_number,
            dtype,
            representation_unit,
        )
        self.interaction_modules = TensorNetInteraction()

    def _forward(self):
        pass

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> TensorNetNeuralNetworkData:
        number_of_atoms = data.atomic_numbers.shape[0]

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

        return nnpdata


class TensorNetRepresentation(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            number_of_radial_basis_functions: int,
            activation_function: nn.Module,
            radial_max_distance: unit.Quantity,
            radial_min_distance: unit.Quantity,
            trainable_rbf: bool,  # TODO
            max_atomic_number: int,
            dtype: torch.dtype,
            representation_unit: unit.Quantity,
    ):
        super().__init__()

        # TensorNet uses angstrom
        self.dtype = dtype
        self.representation_unit = representation_unit

        self.cutoff_module = CosineCutoff(radial_max_distance, representation_unit)
        self.radial_symmetry_function = self._setup_radial_symmetry_functions(
            radial_max_distance,
            radial_min_distance,
            number_of_radial_basis_functions,
        )
        self.hidden_channels = hidden_channels
        self.rsf_projection_I = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.rsf_projection_A = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.rsf_projection_S = nn.Linear(
            number_of_radial_basis_functions,
            hidden_channels,
            dtype=dtype,
        )
        self.atomic_number_i_embedding_layer = nn.Embedding(
            max_atomic_number,
            hidden_channels,
            dtype=dtype,
        )
        self.atomic_number_ij_embedding_layer = nn.Linear(
            2 * hidden_channels,
            hidden_channels,
            dtype=dtype,
        )
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
        self.rsf_projection_I.reset_parameters()
        self.rsf_projection_A.reset_parameters()
        self.rsf_projection_S.reset_parameters()
        self.atomic_number_i_embedding_layer.reset_parameters()
        self.atomic_number_ij_embedding_layer.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _get_atomic_number_message(
            self,
            atomic_number: torch.Tensor,
            pair_indices: torch.Tensor
    ) -> torch.Tensor:
        atomic_number_i_embedding = self.atomic_number_i_embedding_layer(atomic_number)
        atomic_number_ij_embedding = self.atomic_number_ij_embedding_layer(
            atomic_number_i_embedding.index_select(
                0,
                pair_indices.t().reshape(-1)
            ).view(
                -1, self.hidden_channels * 2
            )
        )[..., None, None]
        return atomic_number_ij_embedding

    def _get_tensor_messages(
            self,
            atomic_number_embedding: torch.Tensor,
            d_ij_in_representation_unit: torch.Tensor,
            r_ij_norm: torch.Tensor,
            radial_feature_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C = self.cutoff_module(
            d_ij_in_representation_unit
        ).reshape(-1, 1, 1, 1) * atomic_number_embedding
        eye = torch.eye(3, 3, device=r_ij_norm.device, dtype=r_ij_norm.dtype)[
            None, None, ...
        ]
        Iij = self.rsf_projection_I(radial_feature_vector).permute(0, 2, 1)[..., None] * C * eye
        Aij = (
                self.rsf_projection_A(radial_feature_vector).permute(0, 2, 1)[..., None]
                * C
                * vector_to_skewtensor(r_ij_norm)[..., None, :, :]
        )
        Sij = (
                self.rsf_projection_S(radial_feature_vector).permute(0, 2, 1)[..., None]
                * C
                * vector_to_symtensor(r_ij_norm)[..., None, :, :]
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
            dtype=self.dtype,
            representation_unit=self.representation_unit,
        )
        return radial_symmetry_function

    def forward(self, data: TensorNetNeuralNetworkData):
        atomic_number_embedding = self._get_atomic_number_message(
            data.atomic_numbers,
            data.pair_indices,
        )
        _r_ij_in_representation_unit = (data.r_ij * unit.nanometer).to(self.representation_unit).m
        _d_ij_in_representation_unit = (data.d_ij * unit.nanometer).to(self.representation_unit).m
        _r_ij_norm = _r_ij_in_representation_unit / _d_ij_in_representation_unit

        radial_feature_vector = self.radial_symmetry_function(_d_ij_in_representation_unit)
        # cutoff
        rcut_ij = self.cutoff_module(_d_ij_in_representation_unit)  # cutoff function applied twice
        radial_feature_vector = radial_feature_vector * rcut_ij.unsqueeze(-1)

        Iij, Aij, Sij = self._get_tensor_messages(
            atomic_number_embedding,
            _d_ij_in_representation_unit,
            _r_ij_norm,
            radial_feature_vector
        )
        source = torch.zeros(
            data.atomic_numbers.shape[0],
            self.hidden_channels,
            3,
            3,
            device=data.atomic_numbers.device,
            dtype=Iij.dtype
        )
        I = source.index_add(dim=0, index=data.pair_indices[0], source=Iij)
        A = source.index_add(dim=0, index=data.pair_indices[0], source=Aij)
        S = source.index_add(dim=0, index=data.pair_indices[0], source=Sij)
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

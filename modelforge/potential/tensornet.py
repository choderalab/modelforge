from dataclasses import dataclass

import torch
from openff.units import unit
from torch import nn
# from torchmdnet.models.tensornet import decompose_tensor
# from torchmdnet.models.tensornet import tensor_message_passing
# from torchmdnet.models.tensornet import tensor_norm
# from torchmdnet.models.tensornet import vector_to_skewtensor
# from torchmdnet.models.tensornet import vector_to_symtensor
from typing import Tuple

from modelforge.potential.models import InputPreparation
from modelforge.potential.models import BaseNetwork
from modelforge.potential.models import CoreNetwork
from modelforge.potential.utils import CosineCutoff
from modelforge.potential.utils import TensorNetRadialSymmetryFunction
from modelforge.potential.utils import NeuralNetworkData
from modelforge.potential.utils import NNPInput
from .models import PairListOutputs


def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)

def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S

def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S

def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor**2).sum((-2, -1))

def tensor_message_passing(
    edge_index: torch.Tensor, factor: torch.Tensor, tensor: torch.Tensor, natoms: int
) -> torch.Tensor:
    """Message passing for tensors."""
    msg = factor * tensor.index_select(0, edge_index[1])
    shape = (natoms, tensor.shape[1], tensor.shape[2], tensor.shape[3])
    tensor_m = torch.zeros(*shape, device=tensor.device, dtype=tensor.dtype)
    tensor_m = tensor_m.index_add(0, edge_index[0], msg)
    return tensor_m


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
            number_of_interaction_layers: int = 2,
            number_of_radial_basis_functions: int = 16,
            activation_function: nn.Module = nn.SiLU,
            radial_max_distance: unit.Quantity = 5.1 * unit.angstrom,
            radial_min_distance: unit.Quantity = 0.0 * unit.angstrom,
            trainable_rbf: bool = False,
            max_atomic_number: int = 128,
            equivariance_invariance_group: str = "O(3)",
            dtype: torch.dtype = torch.float32,
            representation_unit: unit.Quantity = unit.angstrom,
    ) -> None:
        super().__init__(
            processing_operation = [],
            readout_operation = [],
        )

        self.core_module = TensorNetCore(
            hidden_channels,
            number_of_interaction_layers,
            number_of_radial_basis_functions,
            activation_function,
            radial_max_distance,
            radial_min_distance,
            trainable_rbf,
            max_atomic_number,
            equivariance_invariance_group,
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
            number_of_interaction_layers: int,
            number_of_radial_basis_functions: int,
            activation_function: nn.Module,
            radial_max_distance: unit.Quantity,
            radial_min_distance: unit.Quantity,
            trainable_rbf: bool,
            max_atomic_number: int,
            equivariance_invariance_group: str,
            dtype: torch.dtype,
            representation_unit: unit.Quantity,
    ):
        super().__init__()

        torch.manual_seed(0)
        self.representation_module = TensorNetRepresentation(
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
        self.interaction_modules = nn.ModuleList()
        if number_of_interaction_layers > 0:
            for _ in range(number_of_interaction_layers):
                self.interaction_modules.append(
                    TensorNetInteraction(
                        hidden_channels,
                        number_of_radial_basis_functions,
                        activation_function,
                        radial_max_distance,
                        equivariance_invariance_group,
                        dtype,
                        representation_unit,
                    )
                )
    def compute_properties(self):
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
        self.hidden_channels = hidden_channels
        self.dtype = dtype
        self.representation_unit = representation_unit

        self.cutoff_module = CosineCutoff(radial_max_distance, representation_unit)
        self.radial_symmetry_function = self._setup_radial_symmetry_functions(
            radial_max_distance,
            radial_min_distance,
            number_of_radial_basis_functions,
        )
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
        self.activation_function = activation_function()
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
            norm = self.activation_function(linear_scalar(norm))
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
    def __init__(
            self,
            hidden_channels: int,
            number_of_radial_basis_functions: int,
            activation_function: nn.Module,
            radial_max_distance: unit.Quantity,
            equivariance_invariance_group,
            dtype: torch.dtype,
            representation_unit: unit.Quantity,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        self.activation_function = activation_function()
        self.cutoff_module = CosineCutoff(radial_max_distance, representation_unit)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(number_of_radial_basis_functions, hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.act = activation_function()
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()

    def forward(
            self,
            X: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor,
            edge_attr: torch.Tensor,
            q: torch.Tensor,
    ) -> torch.Tensor:
        C = self.cutoff_module(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.activation_function(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.hidden_channels, 3
        )
        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = I + A + S
        Im = tensor_message_passing(
            edge_index, edge_attr[..., 0, None, None], I, X.shape[0]
        )
        Am = tensor_message_passing(
            edge_index, edge_attr[..., 1, None, None], A, X.shape[0]
        )
        Sm = tensor_message_passing(
            edge_index, edge_attr[..., 2, None, None], S, X.shape[0]
        )
        msg = Im + Am + Sm
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor((1 + 0.1 * q[..., None, None, None]) * (A + B))
        if self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(2 * B)
        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + (1 + 0.1 * q[..., None, None, None]) * torch.matrix_power(dX, 2)
        return X
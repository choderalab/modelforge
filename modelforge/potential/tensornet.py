"""
TensorNet network for molecular potential learning.
"""

from typing import Dict, Optional, Tuple, Type

import torch
from openff.units import unit
from torch import nn

from modelforge.potential.utils import (
    CosineAttenuationFunction,
    NNPInputTuple,
    TensorNetRadialBasisFunction,
)

from .models import PairlistData


def vector_to_skewtensor(r_ij_norm: torch.Tensor) -> torch.Tensor:
    """
    Creates a skew-symmetric tensor (A) from a vector
    (equation 3 in TensorNet paper).

    Parameters
    ----------
    r_ij_norm : torch.Tensor
        Normalized displacement vectors of given atom pairs.

    Returns
    -------
    torch.Tensor
        Matrix A from equation 3 in TensorNet paper.
    """

    zero = torch.zeros_like(r_ij_norm[:, 0])
    out = torch.stack(
        (
            zero,
            -r_ij_norm[:, 2],
            r_ij_norm[:, 1],
            r_ij_norm[:, 2],
            zero,
            -r_ij_norm[:, 0],
            -r_ij_norm[:, 1],
            r_ij_norm[:, 0],
            zero,
        ),
        dim=-1,
    ).view(-1, 3, 3)

    return out


def vector_to_symtensor(r_ij_norm: torch.Tensor) -> torch.Tensor:
    """
    Creates a symmetric traceless tensor (S) from the outer product of a vector
    with itself (equation 3 in TensorNet paper).

    Parameters
    ----------
    r_ij_norm : torch.Tensor
        Normalized displacement vectors of given atom pairs.

    Returns
    -------
    torch.Tensor
        Matrix S from equation 3 in TensorNet paper.
    """

    r_ij_norm = r_ij_norm.unsqueeze(-1) * r_ij_norm.unsqueeze(-2)
    I = torch.eye(3, device=r_ij_norm.device, dtype=r_ij_norm.dtype) * (
        r_ij_norm.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
    ).unsqueeze(-1)
    S = 0.5 * (r_ij_norm + r_ij_norm.transpose(-1, -2)) - I
    return S


def decompose_tensor(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full tensor decomposition into irreducible components
    (X=I+A+S, equation 2 and 3 in TensorNet paper).

    Parameters
    ----------
    tensor : torch.Tensor
        X tensor that specifies pair-wise features of the atomic system,
        initialized with I+A+S, and is updated along interaction layers.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Decomposed tensors, I, A, and S from tensor feature X.
    """

    diag_mean = tensor.diagonal(offset=0, dim1=-1, dim2=-2).mean(-1)
    I = diag_mean[..., None, None] * torch.eye(
        3, 3, device=tensor.device, dtype=tensor.dtype
    )
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I

    return I, A, S


def tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Frobenius norm
    (mentioned at the end of section 3.1 in TensorNet paper).

    Parameters
    ----------
    tensor : torch.Tensor
        X tensor that specifies pair-wise features of the atomic system,
        initialized with I+A+S, and is updated along interaction layers.

    Returns
    -------
    torch.Tensor
        Normalized tensor X.
    """
    # Note: the Frobenius norm is actually the square root of the sum of squares, so assert torch.allclose(torch.norm(tensor, p="fro", dim=(-2, -1))**2, (tensor**2).sum((-2, -1)) == True
    return (tensor**2).sum((-2, -1))


def tensor_message_passing(
    pair_indices: torch.Tensor,
    radial_feature_vector: torch.Tensor,
    tensor: torch.Tensor,
    tensor_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """
    Helper function to calculate message passing tensor M
    ("Interaction and node update", section 3.2 in TensorNet paper).
    Tensor I, A, and S are parsed separately into this helper function.

    Parameters
    ----------
    pair_indices : torch.Tensor
        A pair-wise index tensor specifying the corresponding atomic pairs.
    radial_feature_vector : torch.Tensor
        Radial feature vector calculated through TensorNetRadialBasisFunction.
    tensor : torch.Tensor
        A pair-wise feature tensor decomposed term (I, A, or S).
    number_of_atoms : int
        Number of atoms in the system.

    Returns
    -------
    torch.Tensor
        A Message tensor calculated from I, A, or S.
    """

    # Compute the message for each pair
    msg = radial_feature_vector * tensor.index_select(0, pair_indices[1])
    # Pre-allocate tensor for the aggregated messages
    tensor_m = torch.zeros(tensor_shape, device=tensor.device, dtype=tensor.dtype)
    # Aggregate the messages, using in-place addition to avoid unnecessary copies
    tensor_m.index_add_(0, pair_indices[0], msg)
    return tensor_m




class TensorNetCore(torch.nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_interaction_layers: int,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: float,
        minimum_interaction_radius: float,
        highest_atomic_number: int,
        equivariance_invariance_group: str,
        activation_function_parameter: Dict,
        potential_seed: int = -1,
        trainable_centers_and_scale_factors: bool = False,
    ) -> None:
        super().__init__()
        activation_function = activation_function_parameter["activation_function"]

        if potential_seed != -1:
            torch.manual_seed(potential_seed)

        self.representation_module = TensorNetRepresentation(
            number_of_per_atom_features=number_of_per_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            activation_function=activation_function,
            maximum_interaction_radius=maximum_interaction_radius,
            minimum_interaction_radius=minimum_interaction_radius,
            trainable_centers_and_scale_factors=trainable_centers_and_scale_factors,
            highest_atomic_number=highest_atomic_number,
        )
        self.interaction_modules = nn.ModuleList(
            [
                TensorNetInteraction(
                    number_of_per_atom_features=number_of_per_atom_features,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    activation_function=activation_function,
                    maximum_interaction_radius=maximum_interaction_radius,
                    equivariance_invariance_group=equivariance_invariance_group,
                )
                for _ in range(number_of_interaction_layers)
            ]
        )

        from modelforge.potential.utils import Dense

        self.readout = Dense(
            3 * number_of_per_atom_features,
            number_of_per_atom_features,
            activation_function=activation_function,
        )
        self.out_norm = nn.LayerNorm(3 * number_of_per_atom_features)

    def compute_properties(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Compute properties for the TensorNet model.

        Parameters
        ----------
        data : TensorNetNeuralNetworkData
            The input data for the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated properties including per atom energy, per atom features, and atomic subsystem indices.
        """

        # generate initial embedding
        X, radial_feature_vector = self.representation_module(data, pairlist_output)

        # using interlevae and bincount to generate a total charge per molecule
        expanded_total_charge = torch.repeat_interleave(
            data.total_charge, data.atomic_subsystem_indices.bincount()
        )

        for layer in self.interaction_modules:
            X = layer(
                X,
                pairlist_output.pair_indices,
                pairlist_output.d_ij.squeeze(-1),
                radial_feature_vector.squeeze(1),
                expanded_total_charge,
            )

        I, A, S = decompose_tensor(X)
        per_atom_scalar_representation = torch.cat(
            (tensor_norm(I), tensor_norm(A), tensor_norm(S)),
            dim=-1,
        )
        per_atom_scalar_representation = self.out_norm(per_atom_scalar_representation)
        per_atom_scalar_representation = self.readout(per_atom_scalar_representation)

        return {
            "per_atom_energy": per_atom_scalar_representation.sum(dim=1),
            "per_atom_scalar_representation": per_atom_scalar_representation,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

    def forward(
        self, data: NNPInputTuple, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the forward pass through the network.

        Parameters
        ----------
        data : NNPInput
            Contains input data for the batch obtained directly from the
            dataset, including atomic numbers, positions, and other relevant
            fields.
        pairlist_output : PairListOutputs
            Contains the indices for the selected pairs and their associated
            distances and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and other properties from the
            forward pass.
        """
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(data, pairlist_output)
        # add atomic numbers to the output
        outputs["atomic_numbers"] = data.atomic_numbers

        return outputs


class TensorNetRepresentation(torch.nn.Module):
    """
    TensorNet representation module for molecular systems.

    Parameters
    ----------
    number_of_per_atom_features : int
        Number of features per atom.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    activation_function : Type[nn.Module]
        Activation function class.
    maximum_interaction_radius : unit.Quantity
        Maximum interaction radius.
    minimum_interaction_radius : unit.Quantity
        Minimum interaction radius.
    trainable_centers_and_scale_factors : bool
        If True, centers and scale factors are trainable.
    highest_atomic_number : int
        Highest atomic number in the dataset.
    """

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        activation_function: Type[nn.Module],
        maximum_interaction_radius: float,
        minimum_interaction_radius: float,
        trainable_centers_and_scale_factors: bool,
        highest_atomic_number: int,
    ):
        super().__init__()
        from modelforge.potential.utils import Dense

        self.number_of_per_atom_features = number_of_per_atom_features

        self.cutoff_module = CosineAttenuationFunction(maximum_interaction_radius)

        self.radial_symmetry_function = TensorNetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            min_distance=minimum_interaction_radius,
            alpha=(
                (maximum_interaction_radius - minimum_interaction_radius) / 5.0
            ),  # TensorNet uses angstrom
            trainable_centers_and_scale_factors=trainable_centers_and_scale_factors,
        )
        self.rsf_projections = nn.ModuleDict(
            {
                "I": nn.Linear(
                    number_of_radial_basis_functions, number_of_per_atom_features
                ),
                "A": nn.Linear(
                    number_of_radial_basis_functions, number_of_per_atom_features
                ),
                "S": nn.Linear(
                    number_of_radial_basis_functions, number_of_per_atom_features
                ),
            }
        )
        self.atomic_number_i_embedding_layer = nn.Embedding(
            highest_atomic_number,
            number_of_per_atom_features,
        )
        self.atomic_number_ij_embedding_layer = nn.Linear(
            2 * number_of_per_atom_features,
            number_of_per_atom_features,
        )
        self.activation_function = activation_function
        # initialize linear layer for I, A and S
        self.linears_tensor = nn.ModuleList(
            [
                nn.Linear(
                    number_of_per_atom_features, number_of_per_atom_features, bias=False
                )
                for _ in range(3)
            ]
        )
        self.linears_scalar = nn.Sequential(
            *[
                Dense(
                    number_of_per_atom_features,
                    2 * number_of_per_atom_features,
                    bias=True,
                    activation_function=self.activation_function,
                ),
                Dense(
                    2 * number_of_per_atom_features,
                    3 * number_of_per_atom_features,
                    bias=True,
                    activation_function=self.activation_function,
                ),
            ]
        )
        self.batch_layer_normalization = nn.LayerNorm(number_of_per_atom_features)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize neural network parameters of the representation layer.
        """
        self.rsf_projections["I"].reset_parameters()
        self.rsf_projections["A"].reset_parameters()
        self.rsf_projections["S"].reset_parameters()
        self.atomic_number_i_embedding_layer.reset_parameters()
        self.atomic_number_ij_embedding_layer.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.batch_layer_normalization.reset_parameters()

    def _get_atomic_number_message(
        self, atomic_number: torch.Tensor, pair_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Embedding layers for atom types of atom pairs
        (mentioned in equation 8 in TensorNet paper, not explicitly defined).
        This embedding consists of two steps:
        1. embed atom type of each atom into a vector
        2. the embedding of an atom pair is the linear combination of the
            embedding vector of these two atoms in the atom pair

        Parameters
        ----------
        atomic_number : torch.Tensor
            A tensor includes atomic numbers for every atom in the system.
        pair_indices : torch.Tensor
            A pair-wise index tensor specifying the corresponding atomic pairs.

        Returns
        -------
        torch.Tensor
            The embedding tensor for atomic numbers of atom pairs.
        """
        atomic_number_i_embedding = self.atomic_number_i_embedding_layer(atomic_number)
        pair_indices_flat = pair_indices.t().reshape(-1)

        atomic_number_ij_embedding = self.atomic_number_ij_embedding_layer(
            atomic_number_i_embedding[pair_indices_flat].view(
                -1, self.number_of_per_atom_features * 2
            )
        )[..., None, None]
        return atomic_number_ij_embedding

    def _get_tensor_messages(
        self,
        atomic_number_embedding: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij_norm: torch.Tensor,
        radial_feature_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize pair-wise tensor representations of atoms
        (equation 8 in TensorNet paper).

        Parameters
        ----------
        atomic_number_embedding : torch.Tensor
            The embedding tensor for atomic numbers of atom pairs.
        d_ij : torch.Tensor
            Atomic pair-wise distances.
        r_ij_norm : torch.Tensor
            normalized displacement vectors, by dividing r_ij by d_ij
        radial_feature_vector : torch.Tensor
            Radial feature vector calculated through
            TensorNetRadialBasisFunction.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The Iij, Aij, Sij terms in equation 8, before adding up since these
            three terms are treated separately.
        """

        C = self.cutoff_module(d_ij).reshape(-1, 1, 1, 1) * atomic_number_embedding
        eye = torch.eye(3, 3, device=r_ij_norm.device, dtype=r_ij_norm.dtype)[
            None, None, ...
        ]
        Iij = (
            self.rsf_projections["I"](radial_feature_vector).permute(0, 2, 1)[..., None]
            * C
            * eye
        )
        Aij = (
            self.rsf_projections["A"](radial_feature_vector).permute(0, 2, 1)[..., None]
            * C
            * vector_to_skewtensor(r_ij_norm)[..., None, :, :]
        )
        Sij = (
            self.rsf_projections["S"](radial_feature_vector).permute(0, 2, 1)[..., None]
            * C
            * vector_to_symtensor(r_ij_norm)[..., None, :, :]
        )
        return Iij, Aij, Sij

    def forward(self, data: NNPInputTuple, pairlist_output: PairlistData):
        """
        Compute the output of the representation layer
        (equation 10 in TensorNet paper).

        Parameters
        ----------
        data : TensorNetNeuralNetworkData
            The input data about the system to build the model.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The first value is the X tensor as a representation of the system.
            The second value is the radial feature vector that is required
            by compute_properties of TensorNetCore.
        """
        atomic_number_embedding = self._get_atomic_number_message(
            data.atomic_numbers,
            pairlist_output.pair_indices,
        )
        r_ij_norm = pairlist_output.r_ij / pairlist_output.d_ij

        radial_feature_vector = self.radial_symmetry_function(
            pairlist_output.d_ij
        )  # in nanometer
        rcut_ij = self.cutoff_module(
            pairlist_output.d_ij
        )  # cutoff function applied twice
        radial_feature_vector = torch.mul(radial_feature_vector, rcut_ij).unsqueeze(1)

        Iij, Aij, Sij = self._get_tensor_messages(
            atomic_number_embedding,
            pairlist_output.d_ij,
            r_ij_norm,
            radial_feature_vector,
        )
        source = torch.zeros(
            data.atomic_numbers.shape[0],
            self.number_of_per_atom_features,
            3,
            3,
            device=data.atomic_numbers.device,
            dtype=Iij.dtype,
        )
        I = source.index_add(dim=0, index=pairlist_output.pair_indices[0], source=Iij)
        A = source.index_add(dim=0, index=pairlist_output.pair_indices[0], source=Aij)
        S = source.index_add(dim=0, index=pairlist_output.pair_indices[0], source=Sij)

        # equation 9 in TensorNet paper
        # batch normalization
        # NOTE: call init_norm differently
        nomalized_tensor_I_A_S = self.batch_layer_normalization(tensor_norm(I + A + S))

        nomalized_tensor_I_A_S = self.linears_scalar(nomalized_tensor_I_A_S).reshape(
            -1, self.number_of_per_atom_features, 3
        )

        # now equation 10
        # apply linear layers to I, A, S and return
        I = (
            self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * nomalized_tensor_I_A_S[..., 0, None, None]
        )
        A = (
            self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * nomalized_tensor_I_A_S[..., 1, None, None]
        )
        S = (
            self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * nomalized_tensor_I_A_S[..., 2, None, None]
        )
        X = I + A + S
        return X, radial_feature_vector


class TensorNetInteraction(torch.nn.Module):
    """
    TensorNet interaction module for message passing and updating atomic features.

    Parameters
    ----------
    number_of_per_atom_features : int
        Number of features per atom.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    activation_function : Type[nn.Module]
        Activation function class.
    maximum_interaction_radius : unit.Quantity
        Maximum interaction radius.
    equivariance_invariance_group : str
        Equivariance invariance group, either "O(3)" or "SO(3)".
    """

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_radial_basis_functions: int,
        activation_function: Type[nn.Module],
        maximum_interaction_radius: unit.Quantity,
        equivariance_invariance_group: str,
    ):
        super().__init__()
        from modelforge.potential.utils import Dense

        self.number_of_per_atom_features = number_of_per_atom_features
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        self.activation_function = activation_function
        self.cutoff_module = CosineAttenuationFunction(maximum_interaction_radius)
        self.mlp_scalar = nn.Sequential(
            Dense(
                number_of_radial_basis_functions,
                number_of_per_atom_features,
                bias=True,
                activation_function=self.activation_function,
            ),
            Dense(
                number_of_per_atom_features,
                2 * number_of_per_atom_features,
                bias=True,
                activation_function=self.activation_function,
            ),
            Dense(
                2 * number_of_per_atom_features,
                3 * number_of_per_atom_features,
                bias=True,
                activation_function=self.activation_function,
            ),
        )

        self.linear_layer = nn.Sequential(
            *[
                Dense(
                    number_of_per_atom_features, number_of_per_atom_features, bias=False
                )
                for _ in range(6)
            ]
        )
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize neural network parameters of the interaction layer.
        """
        for linear in self.mlp_scalar:
            try:
                linear.reset_parameters()
            except AttributeError:
                pass
        for linear in self.linear_layer:
            try:
                linear.reset_parameters()
            except AttributeError:
                pass

    def forward(
        self,
        X: torch.Tensor,
        pair_indices: torch.Tensor,
        d_ij: torch.Tensor,
        radial_feature_vector: torch.Tensor,
        atomic_charges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the output of the interaction layer and update X tensor.
        ("Interaction and node update" from section 3.2 in TensorNet paper).
        X^(i) <- X^(i) + Delta X^(i)

        Parameters
        ----------
        X : torch.Tensor
            X tensor specifies pair-wise features of the atomic system.
        pair_indices : torch.Tensor
            A pair-wise index tensor specifying the corresponding atomic pairs.
        d_ij : torch.Tensor
            Atomic pair-wise distances.
        radial_feature_vector : torch.Tensor
            Radial feature vector calculated through
            TensorNetRadialBasisFunction.
        atomic_charges: torch.Tensor
            Total charge q is a molecule-wise property.
            We transform it into an atom-wise property,
            with all atoms belonging to the same molecule being assigned
            the same charge q (https://github.com/torchmd/torchmd-net/blob/6dea4b61e24de3e18921397866b7d9c5fd6b8bf1/torchmdnet/models/tensornet.py#L237)

        Returns
        -------
        torch.Tensor
            The updated X tensor.
        """

        # see equation 11
        C = self.cutoff_module(d_ij).view(-1, 1)

        # apply scalar MLP to radial feature vector and combine with cutoff
        radial_feature_vector = self.mlp_scalar(radial_feature_vector) * C

        radial_feature_vector = radial_feature_vector.view(
            radial_feature_vector.shape[0], self.number_of_per_atom_features, 3
        )
        X_shape = X.shape
        feature_shape = (X_shape[0], X_shape[1], X_shape[2], X_shape[3])

        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        I = self.linear_layer[0](I.transpose(1, 3)).transpose(1, 3)
        A = self.linear_layer[1](A.transpose(1, 3)).transpose(1, 3)
        S = self.linear_layer[2](S.transpose(1, 3)).transpose(1, 3)

        Y = I + A + S

        Im = tensor_message_passing(
            pair_indices, radial_feature_vector[..., 0, None, None], I, feature_shape
        )
        Am = tensor_message_passing(
            pair_indices, radial_feature_vector[..., 1, None, None], A, feature_shape
        )
        Sm = tensor_message_passing(
            pair_indices, radial_feature_vector[..., 2, None, None], S, feature_shape
        )
        msg = Im + Am + Sm

        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(
                (1 + 0.1 * atomic_charges[..., None, None, None]) * (A + B)
            )

        if self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            I, A, S = decompose_tensor(2 * B)

        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linear_layer[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linear_layer[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linear_layer[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = (
            X
            + dX
            + (1 + 0.1 * atomic_charges[..., None, None, None])
            * torch.matrix_power(dX, 2)
        )
        return X

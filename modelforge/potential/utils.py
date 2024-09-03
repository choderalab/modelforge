"""
Utility functions for neural network potentials.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from openff.units import unit

from modelforge.dataset.dataset import NNPInputTuple, NNPInput


@dataclass
class NeuralNetworkData:
    """
    A dataclass to structure the inputs specifically for SchNet-based neural network potentials, including the necessary geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff (`f_cutoff`) to accurately represent atomistic systems for energy predictions.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A 2D tensor of shape [2, num_pairs], indicating the indices of atom pairs within a molecule or system.
    d_ij : torch.Tensor
        A 1D tensor containing the distances between each pair of atoms identified in `pair_indices`. Shape: [num_pairs, 1].
    r_ij : torch.Tensor
        A 2D tensor of shape [num_pairs, 3], representing the displacement vectors between each pair of atoms.
    number_of_atoms : int
        A integer indicating the number of atoms in the batch.
    positions : torch.Tensor
        A 2D tensor of shape [num_atoms, 3], representing the XYZ coordinates of each atom within the system.
    atomic_numbers : torch.Tensor
        A 1D tensor containing atomic numbers for each atom, used to identify the type of each atom in the system(s).
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule, useful for systems involving multiple
        molecules or distinct subsystems.
    total_charge : torch.Tensor
        A tensor with the total charge of each system or molecule. Shape: [num_systems], where each entry corresponds
        to a distinct system or molecule.
    """

    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor
    atomic_numbers: torch.Tensor
    number_of_atoms: int
    positions: torch.Tensor
    atomic_subsystem_indices: torch.Tensor
    total_charge: torch.Tensor


import torch


@dataclass(frozen=False)
class Metadata:
    """
    A NamedTuple to structure the inputs for neural network potentials.

    Parameters
    ----------
    """

    E: torch.Tensor
    atomic_subsystem_counts: torch.Tensor
    atomic_subsystem_indices_referencing_dataset: torch.Tensor
    number_of_atoms: int
    F: torch.Tensor = torch.tensor([], dtype=torch.float32)

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """Move all tensors in this instance to the specified device."""
        if device:
            self.E = self.E.to(device)
            self.F = self.F.to(device)
            self.atomic_subsystem_counts = self.atomic_subsystem_counts.to(device)
            self.atomic_subsystem_indices_referencing_dataset = (
                self.atomic_subsystem_indices_referencing_dataset.to(device)
            )
        if dtype:
            self.E = self.E.to(dtype)
            self.F = self.F.to(dtype)
        return self


@dataclass
class BatchData:
    nnp_input: NNPInput
    metadata: Metadata

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.nnp_input = self.nnp_input.to(device=device, dtype=dtype)
        self.metadata = self.metadata.to(device=device, dtype=dtype)
        return self


def shared_config_prior():

    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
    }


from typing import Dict, List


class AddPerMoleculeValue(nn.Module):
    """
    Module that adds a per-molecule value to a per-atom property tensor.
    The per-molecule value is expanded to match th elength of the per-atom property tensor.

    Parameters
    ----------
    key : str
        The key to access the per-molecule value from the input data.

    Attributes
    ----------
    key : str
        The key to access the per-molecule value from the input data.
    """

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(
        self, per_atom_property_tensor: torch.Tensor, data: NNPInputTuple
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        per_atom_property_tensor : torch.Tensor
            The per-atom property tensor.
        data : NNPInput
            The input data containing the per-molecule value.

        Returns
        -------
        torch.Tensor
            The updated per-atom property tensor with the per-molecule value appended.
        """
        values_to_append = getattr(data, self.key)
        _, counts = torch.unique(data.atomic_subsystem_indices, return_counts=True)
        expanded_values = torch.repeat_interleave(values_to_append, counts).unsqueeze(1)
        return torch.cat((per_atom_property_tensor, expanded_values), dim=1)


class AddPerAtomValue(nn.Module):
    """
    Module that adds a per-atom value to a tensor.

    Parameters
    ----------
    key : str
        The key to access the per-atom value from the input data.

    Attributes
    ----------
    key : str
        The key to access the per-atom value from the input data.
    """

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(
        self, per_atom_property_tensor: torch.Tensor, data: NNPInputTuple
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        per_atom_property_tensor : torch.Tensor
            The input tensor representing per-atom properties.
        data : NNPInput
            The input data object containing additional information.

        Returns
        -------
        torch.Tensor
            The tensor with the per-atom value appended.
        """
        values_to_append = getattr(data, self.key)
        return torch.cat((per_atom_property_tensor, values_to_append), dim=1)


class FeaturizeInput(nn.Module):
    """
    Module that featurizes the input data.

    Parameters
    ----------
    featurization_config : Dict[str, Union[List[str], int]]
        The configuration for featurization, including the properties to featurize and the maximum atomic number.

    Attributes
    ----------
    _SUPPORTED_FEATURIZATION_TYPES : List[str]
        The list of supported featurization types.
    nuclear_charge_embedding : Embedding
        The embedding layer for nuclear charges.
    append_to_embedding_tensor : nn.ModuleList
        The list of modules to append to the embedding tensor.
    registered_appended_properties : List[str]
        The list of registered appended properties.
    embeddings : nn.ModuleList
        The list of embedding layers for additional categorical properties.
    registered_embedding_operations : List[str]
        The list of registered embedding operations.
    increase_dim_of_embedded_tensor : int
        The increase in dimension of the embedded tensor.
    mixing : nn.Identity or Dense
        The mixing layer for the final embedding.

    Methods
    -------
    forward(data: NNPInput) -> torch.Tensor:
        Featurize the input data.
    """

    _SUPPORTED_FEATURIZATION_TYPES = [
        "atomic_number",
        "per_molecule_total_charge",
        "spin_state",
    ]

    def __init__(self, featurization_config: Dict[str, Dict[str, int]]) -> None:
        """
        Initialize the FeaturizeInput class.

        For per-atom non-categorical properties and per-molecule properties (both categorical and non-categorical), we append the embedded nuclear charges and mix them using a linear layer.

        For per-atom categorical properties, we define an additional embedding and add the embedding to the nuclear charge embedding.

        Parameters
        ----------
        featurization_config : dict
            A dictionary containing the featurization configuration. It should have the following keys:
            - "properties_to_featurize" : list
                A list of properties to featurize.
            - "maximum_atomic_number" : int
                The maximum atomic number.
            - "number_of_per_atom_features" : int
                The number of per-atom features.

        Returns
        -------
        None
        """
        super().__init__()

        # expend embedding vector
        self.append_to_embedding_tensor = nn.ModuleList()
        self.registered_appended_properties: List[str] = []
        # what other categorial properties are embedded
        self.embeddings = nn.ModuleList()
        self.registered_embedding_operations: List[str] = []

        self.increase_dim_of_embedded_tensor: int = 0

        # iterate through the supported featurization types and check if one of these is requested
        for featurization in self._SUPPORTED_FEATURIZATION_TYPES:

            # embed nuclear charges
            if (
                featurization == "atomic_number"
                and featurization in featurization_config
            ):
                self.nuclear_charge_embedding = torch.nn.Embedding(
                    int(featurization_config[featurization]["maximum_atomic_number"]),
                    int(
                        featurization_config[featurization][
                            "number_of_per_atom_features"
                        ]
                    ),
                )
                self.registered_embedding_operations.append("nuclear_charge_embedding")

            # add total charge to embedding vector
            if (
                featurization == "per_molecule_total_charge"
                and featurization in featurization_config
            ):
                # transform output o f embedding with shape (nr_atoms, nr_features) to (nr_atoms, nr_features + 1). The added features is the total charge (which will be transformed to a per-atom property)
                self.append_to_embedding_tensor.append(
                    AddPerMoleculeValue("total_charge")
                )
                self.increase_dim_of_embedded_tensor += 1
                self.registered_appended_properties.append("total_charge")

            # add partial charge to embedding vector
            if (
                featurization == "per_atom_partial_charge"
                and featurization in featurization_config
            ):  # transform output o f embedding with shape (nr_atoms, nr_features) to (nr_atoms, nr_features + 1). The added features is the total charge (which will be transformed to a per-atom property)
                self.append_to_embedding_tensor.append(
                    AddPerAtomValue("partial_charge")
                )
                self.increase_dim_of_embedded_tensor += 1
                self.append_to_embedding_tensor("partial_charge")

        # if only nuclear charges are embedded no mixing is performed
        self.mixing: Union[nn.Identity, DenseWithCustomDist]
        if self.increase_dim_of_embedded_tensor == 0:
            self.mixing = nn.Identity()
        else:
            self.mixing = DenseWithCustomDist(
                int(featurization_config["number_of_per_atom_features"])
                + self.increase_dim_of_embedded_tensor,
                int(featurization_config["number_of_per_atom_features"]),
            )

    def forward(self, data: NNPInputTuple) -> torch.Tensor:
        """
        Featurize the input data.

        Parameters
        ----------
        data : NNPInput
            The input data.

        Returns
        -------
        torch.Tensor
            The featurized input data.
        """

        atomic_numbers = data.atomic_numbers
        embedded_nuclear_charges = self.nuclear_charge_embedding(atomic_numbers)

        for additional_embedding in self.embeddings:
            embedded_nuclear_charges = additional_embedding(
                embedded_nuclear_charges, data
            )

        for append_embedding_vector in self.append_to_embedding_tensor:
            embedded_nuclear_charges = append_embedding_vector(
                embedded_nuclear_charges, data
            )

        return self.mixing(embedded_nuclear_charges)


import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


class Dense(nn.Linear):
    """
    Fully connected linear layer with activation function.

    forward(input)
        Forward pass of the layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: nn.Module = nn.Identity(),
    ):
        """
        A linear or non-linear transformation

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. Default is True.
        activation_function : Type[torch.nn.Module] , optional
            Activation function to be applied. Default is nn.Identity(), which applies the identity function and makes this a linear transformation.
        """

        super().__init__(in_features, out_features, bias)

        self.activation_function = activation_function

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and activation function.

        """
        y = F.linear(input, self.weight, self.bias)
        return self.activation_function(y)


class DenseWithCustomDist(nn.Linear):
    """
    Fully connected linear layer with activation function.

    Attributes
    ----------
    weight_init_distribution : Callable
        distribution used to initialize the weights.
    bias_init_distribution : Callable
        Distribution used to initialize the bias.

    Methods
    -------
    reset_parameters()
        Reset the weights and bias using the specified initialization distributions.
    forward(input)
        Forward pass of the layer.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: nn.Module = nn.Identity(),
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        A linear or non-linear transformation

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            If set to False, the layer will not learn an additive bias. Default is True.
        activation_function : nn.Module , optional
            Activation function to be applied. Default is nn.Identity(), which applies the identity function and makes this a linear ransformation.
        weight_init : Callable, optional
            Callable to initialize the weights. Default is xavier_uniform_.
        bias_init : Callable, optional
            Function to initialize the bias. Default is zeros_.
        """
        # NOTE: these two variables need to come before the init
        self.weight_init_distribution = weight_init
        self.bias_init_distribution = bias_init

        super().__init__(
            in_features, out_features, bias
        )  # NOTE: the `reset_paramters` method is called in the super class

        self.activation_function = activation_function

    def reset_parameters(self):
        """
        Reset the weights and bias using the specified initialization distributions.
        """
        self.weight_init_distribution(self.weight)
        if self.bias is not None:
            self.bias_init_distribution(self.bias)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation and activation function.

        """
        y = F.linear(input, self.weight, self.bias)
        return self.activation_function(y)


from openff.units import unit


class CosineAttenuationFunction(nn.Module):
    def __init__(self, cutoff: float):
        """
        Behler-style cosine cutoff module. This anneals the signal smoothly to zero at the cutoff distance.

        NOTE: The cutoff is converted to nanometer and the input MUST be in nanomter too.

        Parameters:
        -----------
        cutoff: unit.Quantity
            The cutoff distance.

        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):
        """
        Compute the cosine cutoff for a distance tensor.
        NOTE: the cutoff function doesn't care about units as long as they are consisten,

        Parameters
        -----------
        d_ij : Tensor
            Pairwise distance tensor in nanometer. Shape: [n_pairs, 1]

        Returns
        --------
        Tensor
            Cosine cutoff tensor. Shape: [n_pairs, 1]
        """
        # Compute values of cutoff function
        input_cut = 0.5 * (
            torch.cos(d_ij * np.pi / self.cutoff) + 1.0
        )  # NOTE: ANI adds 0.5 instead of 1.
        # Remove contributions beyond the cutoff radius
        input_cut *= (d_ij < self.cutoff).float()
        return input_cut


from typing import Dict


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_2 = math.log(2.0)

    def forward(self, x: torch.Tensor):
        """
        Compute shifted soft-plus activation function.

        The shifted soft-plus activation function is defined as:
        y = ln(1 + exp(-x)) - ln(2)

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -----------
        torch.Tensor
            Shifted soft-plus of the input.
        """

        return functional.softplus(x) - self.log_2


class AngularSymmetryFunction(nn.Module):
    """
    Initialize AngularSymmetryFunction module.

    """

    def __init__(
        self,
        maximum_interaction_radius: float,
        min_distance: float,
        number_of_gaussians_for_asf: int = 8,
        angle_sections: int = 4,
        trainable: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----
        number_of_gaussian: Number of gaussian functions to use for angular symmetry function.
        angular_cutoff: Cutoff distance for angular symmetry function.
        angular_start: Starting distance for angular symmetry function.
        ani_style: Whether to use ANI symmetry function style.
        """

        super().__init__()
        from loguru import logger as log

        self.number_of_gaussians_asf = number_of_gaussians_for_asf
        self.angular_cutoff = maximum_interaction_radius
        self.cosine_cutoff = CosineAttenuationFunction(self.angular_cutoff)
        _unitless_angular_cutoff = maximum_interaction_radius
        self.angular_start = min_distance
        _unitless_angular_start = min_distance

        # save constants
        EtaA = angular_eta = 12.5 * 100  # FIXME hardcoded eta
        Zeta = 14.1000  # FIXME hardcoded zeta

        if trainable:
            self.EtaA = torch.tensor([EtaA], dtype=dtype)
            self.Zeta = torch.tensor([Zeta], dtype=dtype)
            self.Rca = torch.tensor([_unitless_angular_cutoff], dtype=dtype)

        else:
            self.register_buffer("EtaA", torch.tensor([EtaA], dtype=dtype))
            self.register_buffer("Zeta", torch.tensor([Zeta], dtype=dtype))
            self.register_buffer(
                "Rca", torch.tensor([_unitless_angular_cutoff], dtype=dtype)
            )

        # ===============
        # # calculate shifts
        # ===============
        import math

        # ShfZ
        angle_start = math.pi / (2 * angle_sections)
        ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]
        # ShfA
        ShfA = torch.linspace(
            _unitless_angular_start,
            _unitless_angular_cutoff,
            number_of_gaussians_for_asf + 1,
        )[:-1]
        # register shifts
        if trainable:
            self.ShfZ = ShfZ
            self.ShfA = ShfA
        else:
            self.register_buffer("ShfZ", ShfZ)
            self.register_buffer("ShfA", ShfA)

        # The length of angular subaev of a single species
        self.angular_sublength = self.ShfA.numel() * self.ShfZ.numel()

    def forward(self, r_ij: torch.Tensor) -> torch.Tensor:
        # calculate the angular sub aev
        sub_aev = self.compute_angular_sub_aev(r_ij)
        return sub_aev

    def compute_angular_sub_aev(self, vectors12: torch.Tensor) -> torch.Tensor:
        """Compute the angular subAEV terms of the center atom given neighbor pairs.

        This correspond to equation (4) in the ANI paper. This function just
        compute the terms. The sum in the equation is not computed.
        The input tensor have shape (conformations, atoms, N), where N
        is the number of neighbor atom pairs within the cutoff radius and
        output tensor should have shape
        (conformations, atoms, ``self.angular_sublength()``)

        """
        vectors12 = vectors12.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        distances12 = vectors12.norm(2, dim=-5)

        # 0.95 is multiplied to the cos values to prevent acos from
        # returning NaN.
        cos_angles = 0.95 * torch.nn.functional.cosine_similarity(
            vectors12[0], vectors12[1], dim=-5
        )
        angles = torch.acos(cos_angles)
        fcj12 = self.cosine_cutoff(distances12)
        factor1 = ((1 + torch.cos(angles - self.ShfZ)) / 2) ** self.Zeta
        factor2 = torch.exp(
            -self.EtaA * (distances12.sum(0) / 2 - self.ShfA) ** 2
        ).unsqueeze(-1)
        factor2 = factor2.squeeze(4).squeeze(3)
        ret = 2 * factor1 * factor2 * fcj12.prod(0)
        # At this point, ret now have shape
        # (conformations, atoms, N, ?, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as one
        # dimension vector
        return ret.flatten(start_dim=-4)


import math
from abc import ABC, abstractmethod

from torch.nn import functional


class RadialBasisFunctionCore(nn.Module, ABC):

    def __init__(self, number_of_radial_basis_functions):
        super().__init__()
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

    @abstractmethod
    def forward(self, nondimensionalized_distances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---------
        nondimensionalized_distances: torch.Tensor, shape [number_of_pairs, number_of_radial_basis_functions]
            Nondimensional quantities that depend on pairwise distances.

        Returns
        ---------
        torch.Tensor, shape [number_of_pairs, number_of_radial_basis_functions]
        """
        pass


class GaussianRadialBasisFunctionCore(RadialBasisFunctionCore):

    def forward(self, nondimensionalized_distances: torch.Tensor) -> torch.Tensor:
        assert nondimensionalized_distances.ndim == 2
        assert (
            nondimensionalized_distances.shape[1]
            == self.number_of_radial_basis_functions
        )

        return torch.exp(-(nondimensionalized_distances**2))


class RadialBasisFunction(nn.Module, ABC):

    def __init__(
        self,
        radial_basis_function: RadialBasisFunctionCore,
        dtype: torch.dtype,
        prefactor: float = 1.0,
        trainable_prefactor: bool = False,
    ):
        super().__init__()
        if trainable_prefactor:
            self.prefactor = nn.Parameter(torch.tensor([prefactor], dtype=dtype))
        else:
            self.register_buffer("prefactor", torch.tensor([prefactor], dtype=dtype))
        self.radial_basis_function = radial_basis_function

    @abstractmethod
    def nondimensionalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---------
        distances: torch.Tensor, shape [number_of_pairs, 1]
            Distances between atoms in each pair in nanometers.

        Returns
        ---------
        torch.Tensor, shape [number_of_pairs, number_of_radial_basis_functions]
            Nondimensional quantities computed from the distances.
        """
        pass

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        The input distances have implicit units of nanometers by the convention of modelforge. This function applies
        nondimensionalization transformations on the distances and passes the dimensionless result to
        RadialBasisFunctionCore. There can be several nondimsionalization transformations, corresponding to each element
        along the number_of_radial_basis_functions axis in the output.

        Parameters
        ---------
        distances: torch.Tensor, shape [number_of_pairs, 1]
            Distances between atoms in each pair in nanometers.

        Returns
        ---------
        torch.Tensor, shape [number_of_pairs, number_of_radial_basis_functions]
            Output of radial basis functions.
        """
        nondimensionalized_distances = self.nondimensionalize_distances(distances)
        return self.prefactor * self.radial_basis_function(nondimensionalized_distances)


class GaussianRadialBasisFunctionWithScaling(RadialBasisFunction):
    """
    Shifts inputs by a set of centers and scales by a set of scale factors before passing into the standard Gaussian.
    """

    def __init__(
        self,
        number_of_radial_basis_functions: int,
        max_distance: float,
        min_distance: float = 0.0,
        dtype: torch.dtype = torch.float32,
        prefactor: float = 1.0,
        trainable_prefactor: bool = False,
        trainable_centers_and_scale_factors: bool = False,
    ):
        """
        Parameters
        ---------
        number_of_radial_basis_functions: int
            Number of radial basis functions to use.
        max_distance: unit.Quantity
            Maximum distance to consider for symmetry functions.
        min_distance: unit.Quantity
            Minimum distance to consider.
        dtype: torch.dtype, default None
            Data type for computations.
        prefactor: float
            Scalar factor by which to multiply output of radial basis functions.
        trainable_prefactor: bool, default False
            Whether prefactor is trainable
        trainable_centers_and_scale_factors: bool, default False
            Whether centers and scale factors are trainable.
        """

        super().__init__(
            GaussianRadialBasisFunctionCore(number_of_radial_basis_functions),
            dtype,
            prefactor,
            trainable_prefactor,
        )
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        self.dtype = dtype
        self.trainable_centers_and_scale_factors = trainable_centers_and_scale_factors
        # convert to nanometer
        _max_distance_in_nanometer = max_distance
        _min_distance_in_nanometer = min_distance

        # calculate radial basis centers
        radial_basis_centers = self.calculate_radial_basis_centers(
            self.number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            self.dtype,
        )
        # calculate scale factors
        radial_scale_factor = self.calculate_radial_scale_factor(
            self.number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            self.dtype,
        )

        # either add as parameters or register buffers
        if self.trainable_centers_and_scale_factors:
            self.radial_basis_centers = radial_basis_centers
            self.radial_scale_factor = radial_scale_factor
        else:
            self.register_buffer("radial_basis_centers", radial_basis_centers)
            self.register_buffer("radial_scale_factor", radial_scale_factor)

    @staticmethod
    @abstractmethod
    def calculate_radial_basis_centers(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        """
        NOTE: centers have units of nanometers
        """
        pass

    @staticmethod
    @abstractmethod
    def calculate_radial_scale_factor(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        """
        NOTE: radial scale factors have units of nanometers
        """
        pass

    def nondimensionalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        # Here, self.radial_scale_factor is interpreted as sqrt(2) times the standard deviation of the Gaussian.
        diff = distances - self.radial_basis_centers
        return diff / self.radial_scale_factor


class SchnetRadialBasisFunction(GaussianRadialBasisFunctionWithScaling):
    """
    Implementation of the radial basis function as used by the SchNet neural network
    """

    def __init__(
        self,
        number_of_radial_basis_functions: int,
        max_distance: float,
        min_distance: float = 0.0,
        dtype: torch.dtype = torch.float32,
        trainable_centers_and_scale_factors: bool = False,
    ):
        """
        Parameters
        ---------
        number_of_radial_basis_functions: int
            Number of radial basis functions to use.
        max_distance: unit.Quantity
            Maximum distance to consider for symmetry functions.
        min_distance: unit.Quantity
            Minimum distance to consider.
        dtype: torch.dtype, default None
            Data type for computations.
        trainable_centers_and_scale_factors: bool, default False
            Whether centers and scale factors are trainable.
        """
        super().__init__(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            dtype,
            trainable_prefactor=False,
            trainable_centers_and_scale_factors=trainable_centers_and_scale_factors,
        )

    @staticmethod
    def calculate_radial_basis_centers(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        return torch.linspace(
            _min_distance_in_nanometer,
            _max_distance_in_nanometer,
            number_of_radial_basis_functions,
            dtype=dtype,
        )

    @staticmethod
    def calculate_radial_scale_factor(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        scale_factors = torch.linspace(
            _min_distance_in_nanometer,
            _max_distance_in_nanometer,
            number_of_radial_basis_functions,
            dtype=dtype,
        )

        widths = torch.abs(scale_factors[1] - scale_factors[0]) * torch.ones_like(
            scale_factors
        )

        scale_factors = math.sqrt(2) * widths
        return scale_factors


class AniRadialBasisFunction(GaussianRadialBasisFunctionWithScaling):
    """
    Implementation of the radial basis function as used by the ANI neural network
    """

    def __init__(
        self,
        number_of_radial_basis_functions,
        max_distance: unit.Quantity,
        min_distance: unit.Quantity = 0.0 * unit.nanometer,
        dtype: torch.dtype = torch.float32,
        trainable_centers_and_scale_factors: bool = False,
    ):
        """
        Parameters
        ---------
        number_of_radial_basis_functions: int
            Number of radial basis functions to use.
        max_distance: unit.Quantity
            Maximum distance to consider for symmetry functions.
        min_distance: unit.Quantity
            Minimum distance to consider.
        dtype: torch.dtype, default torch.float32
            Data type for computations.
        trainable_centers_and_scale_factors: bool, default False
            Whether centers and scale factors are trainable.
        """
        super().__init__(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            dtype,
            prefactor=0.25,
            trainable_prefactor=False,
            trainable_centers_and_scale_factors=trainable_centers_and_scale_factors,
        )

    @staticmethod
    def calculate_radial_basis_centers(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        centers = torch.linspace(
            _min_distance_in_nanometer,
            _max_distance_in_nanometer,
            number_of_radial_basis_functions + 1,
            dtype=dtype,
        )[:-1]
        return centers

    @staticmethod
    def calculate_radial_scale_factor(
        number_of_radial_basis_functions,
        _max_distance_in_nanometer,
        _min_distance_in_nanometer,
        dtype,
    ):
        # ANI uses a predefined scaling factor
        scale_factors = torch.full(
            (number_of_radial_basis_functions,), (19.7 * 100) ** -0.5
        )
        return scale_factors


class PhysNetRadialBasisFunction(RadialBasisFunction):
    """
    Implementation of the radial basis function as used by the PysNet neural network
    """

    def __init__(
        self,
        number_of_radial_basis_functions: int,
        max_distance: float,
        min_distance: float = 0.0,
        alpha: float = 1.0,
        dtype: torch.dtype = torch.float32,
        trainable_centers_and_scale_factors: bool = False,
    ):
        """
        Parameters
        ----------
        number_of_radial_basis_functions : int
            Number of radial basis functions to use.
        max_distance : unit.Quantity
            Maximum distance to consider for symmetry functions.
        min_distance : unit.Quantity
            Minimum distance to consider, by default 0.0 * unit.nanometer.
        alpha: unit.Quantity
            Scale factor used to nondimensionalize the input to all exp calls. The PhysNet paper implicitly divides by 1
            Angstrom within exponentials. Note that this is distinct from the unitless scale factors used outside the
            exp but within the Gaussian.
        dtype : torch.dtype, optional
            Data type for computations, by default torch.float32.
        trainable_centers_and_scale_factors : bool, optional
            Whether centers and scale factors are trainable, by default False.
        """

        super().__init__(
            GaussianRadialBasisFunctionCore(number_of_radial_basis_functions),
            trainable_prefactor=False,
            dtype=dtype,
        )
        self._min_distance_in_nanometer = min_distance
        self._alpha_in_nanometer = alpha
        radial_basis_centers = self.calculate_radial_basis_centers(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            alpha,
            dtype,
        )
        # calculate scale factors
        radial_scale_factor = self.calculate_radial_scale_factor(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            alpha,
            dtype,
        )

        if trainable_centers_and_scale_factors:
            self.radial_basis_centers = radial_basis_centers
            self.radial_scale_factor = radial_scale_factor
        else:
            self.register_buffer("radial_basis_centers", radial_basis_centers)
            self.register_buffer("radial_scale_factor", radial_scale_factor)

    @staticmethod
    def calculate_radial_basis_centers(
        number_of_radial_basis_functions,
        max_distance,
        min_distance,
        alpha,
        dtype,
    ):
        # initialize centers according to the default values in PhysNet
        # (see mu_k in Figure 2 caption of https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181)
        # NOTE: Unlike GaussianRadialBasisFunctionWithScaling, the centers are unitless.

        start_value = torch.exp(
            torch.scalar_tensor(
                ((-max_distance + min_distance) / alpha),
                dtype=dtype,
            )
        )
        centers = torch.linspace(
            start_value, 1, number_of_radial_basis_functions, dtype=dtype
        )
        return centers

    @staticmethod
    def calculate_radial_scale_factor(
        number_of_radial_basis_functions,
        max_distance,
        min_distance,
        alpha,
        dtype,
    ):
        # initialize according to the default values in PhysNet (see beta_k in Figure 2 caption)
        # NOTES:
        # - Unlike GaussianRadialBasisFunctionWithScaling, the scale factors are unitless.
        # - Each element of radial_square_factor here is the reciprocal of the square root of beta_k in the
        # Eq. 7 of the PhysNet paper. This way, it is consistent with the sqrt(2) * standard deviation interpretation
        # of radial_scale_factor in GaussianRadialBasisFunctionWithScaling
        return torch.full(
            (number_of_radial_basis_functions,),
            (2 * (1 - math.exp(((-max_distance + min_distance) / alpha))))
            / number_of_radial_basis_functions,
            dtype=dtype,
        )

    def nondimensionalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        # Transformation within the outer exp of PhysNet Eq. 7
        # NOTE: the PhysNet paper implicitly multiplies by 1/Angstrom within the inner exp but distances are in
        # nanometers, so we multiply by 10/nanometer

        return (
            torch.exp(
                (-distances + self._min_distance_in_nanometer)
                / self._alpha_in_nanometer
            )
            - self.radial_basis_centers
        ) / self.radial_scale_factor


class TensorNetRadialBasisFunction(PhysNetRadialBasisFunction):
    """
    The only difference from PhysNetRadialBasisFunction is that alpha is set
    to 1 angstrom only for the purpose of unitless calculations.
    """

    @staticmethod
    def calculate_radial_basis_centers(
        number_of_radial_basis_functions,
        max_distance,
        min_distance,
        alpha,
        dtype,
    ):
        alpha = 1
        start_value = torch.exp(
            torch.scalar_tensor(
                ((-max_distance + min_distance) / alpha),
                dtype=dtype,
            )
        )
        centers = torch.linspace(
            start_value, 1, number_of_radial_basis_functions, dtype=dtype
        )
        return centers

    @staticmethod
    def calculate_radial_scale_factor(
        number_of_radial_basis_functions,
        max_distance,
        min_distance,
        alpha,
        dtype,
    ):
        alpha = 1
        start_value = torch.exp(
            torch.scalar_tensor(((-max_distance + min_distance) / alpha))
        )
        radial_scale_factor = torch.full(
            (number_of_radial_basis_functions,),
            2 / number_of_radial_basis_functions * (1 - start_value),
            dtype=dtype,
        )

        return radial_scale_factor

    def nondimensionalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        # Transformation within the outer exp of PhysNet Eq. 7
        # NOTE: the PhysNet paper implicitly multiplies by 1/Angstrom within the inner exp but distances are in
        # nanometers, so we multiply by 10/nanometer

        return (
            torch.exp(
                (-distances + self._min_distance_in_nanometer)
                / self._alpha_in_nanometer
            )
            - self.radial_basis_centers
        ) / self.radial_scale_factor


def pair_list(
    atomic_subsystem_indices: torch.Tensor,
    only_unique_pairs: bool = False,
) -> torch.Tensor:
    """Compute all pairs of atoms and their distances.

    Parameters
    ----------
    atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
        Atom indices to indicate which atoms belong to which molecule
    only_unique_pairs : bool, optional
        If True, only unique pairs are returned (default is False).
        Otherwise, all pairs are returned.
    """
    # generate index grid
    n = len(atomic_subsystem_indices)

    # get device that passed tensors lives on, initialize on the same device
    device = atomic_subsystem_indices.device

    if only_unique_pairs:
        i_indices, j_indices = torch.triu_indices(n, n, 1, device=device)
    else:
        # Repeat each number n-1 times for i_indices
        i_indices = torch.repeat_interleave(
            torch.arange(n, device=device), repeats=n - 1
        )

        # Correctly construct j_indices
        j_indices = torch.cat(
            [
                torch.cat(
                    (
                        torch.arange(i, device=device),
                        torch.arange(i + 1, n, device=device),
                    )
                )
                for i in range(n)
            ]
        )

    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = (
        atomic_subsystem_indices[i_indices] == atomic_subsystem_indices[j_indices]
    )

    # Apply mask to get final pair indices
    i_final_pairs = i_indices[same_molecule_mask]
    j_final_pairs = j_indices[same_molecule_mask]

    # concatenate to form final (2, n_pairs) tensor
    pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    return pair_indices.to(device)


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Softmax operation over all values in :attr:`src` tensor that share indices
    specified in the :attr:`index` tensor along a given axis :attr:`dim`.

    For one-dimensional tensors, the operation computes

    .. math::
        \mathrm{out}_i = {\textrm{softmax}(\mathrm{src})}_i =
        \frac{\exp(\mathrm{src}_i)}{\sum_j \exp(\mathrm{src}_j)}

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        dim_size: The number of classes, i.e. the number of unique indices in `index`.

    :rtype: :class:`Tensor`

    Adapted from: https://github.com/rusty1s/pytorch_scatter/blob/c31915e1c4ceb27b2e7248d21576f685dc45dd01/torch_scatter/composite/softmax.py
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            "`scatter_softmax` can only be computed over tensors "
            "with floating point data types."
        )

    assert dim >= 0, f"dim must be non-negative, got {dim}"
    assert (
        dim < src.dim()
    ), f"dim must be less than the number of dimensions of src {src.dim()}, got {dim}"

    out_shape = [
        other_dim_size if (other_dim != dim) else dim_size
        for (other_dim, other_dim_size) in enumerate(src.shape)
    ]
    index = index.to(torch.int64)
    zeros = torch.zeros(out_shape, dtype=src.dtype, device=device)
    max_value_per_index = zeros.scatter_reduce(
        dim, index, src, "amax", include_self=False
    )
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp()

    sum_per_index = torch.zeros(out_shape, dtype=src.dtype, device=device).scatter_add(
        dim, index, recentered_scores_exp
    )
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


from enum import Enum

ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU,
    "CeLU": nn.CELU,
    "Sigmoid": nn.Sigmoid,
    "Softmax": nn.Softmax,
    "ShiftedSoftplus": ShiftedSoftplus,
    "SiLU": nn.SiLU,
    "Tanh": nn.Tanh,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    # Add more activation functions as needed
}

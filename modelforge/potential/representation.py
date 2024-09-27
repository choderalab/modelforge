import torch
from typing import Optional
from torch import nn
import numpy as np


class PhysNetAttenuationFunction(nn.Module):
    def __init__(self, cutoff: float):
        """
        Initialize the PhysNet attenuation function.

        Parameters
        ----------
        cutoff : unit.Quantity
            The cutoff distance.
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff]))

    def forward(self, d_ij: torch.Tensor):

        return torch.clamp(
            (
                1
                - 6 * torch.pow((d_ij / self.cutoff), 5)
                + 15 * torch.pow((d_ij / self.cutoff), 4)
                - 10 * torch.pow((d_ij / self.cutoff), 3)
            ),
            min=0,
        )


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
        prefactor: float = 1,
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
        The input distances have implicit units of nanometers by the convention
        of modelforge. This function applies nondimensionalization
        transformations on the distances and passes the dimensionless result to
        RadialBasisFunctionCore. There can be several nondimsionalization
        transformations, corresponding to each element along the
        number_of_radial_basis_functions axis in the output.

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
        max_distance: float
            Maximum distance to consider for symmetry functions.
        min_distance: float
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
        alpha: float = 0.1,
        dtype: torch.dtype = torch.float32,
        trainable_centers_and_scale_factors: bool = False,
    ):
        """
        Parameters
        ----------
        number_of_radial_basis_functions : int
            Number of radial basis functions to use.
        max_distance : float
            Maximum distance to consider for symmetry functions.
        min_distance : float
            Minimum distance to consider, by default 0.0 * unit.nanometer.
        alpha: float
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
        # initialize centers according to the default values in PhysNet (see
        # mu_k in Figure 2 caption of
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181) NOTE: Unlike
        # GaussianRadialBasisFunctionWithScaling, the centers are unitless.

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
        # initialize according to the default values in PhysNet (see beta_k in
        # Figure 2 caption) NOTES:
        # - Unlike GaussianRadialBasisFunctionWithScaling, the scale factors are
        #   unitless.
        # - Each element of radial_square_factor here is the reciprocal of the
        # square root of beta_k in the Eq. 7 of the PhysNet paper. This way, it
        # is consistent with the sqrt(2) * standard deviation interpretation of
        # radial_scale_factor in GaussianRadialBasisFunctionWithScaling
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
        alpha = 0.1
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
        alpha = 0.1
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
        # Transformation within the outer exp of PhysNet Eq. 7 NOTE: the PhysNet
        # paper implicitly multiplies by 1/Angstrom within the inner exp but
        # distances are in nanometers, so we multiply by 10/nanometer

        return (
            torch.exp(
                (-distances + self._min_distance_in_nanometer)
                / self._alpha_in_nanometer
            )
            - self.radial_basis_centers
        ) / self.radial_scale_factor

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from openff.units import unit

from .models import CoreNetwork

if TYPE_CHECKING:
    from .models import PairListOutputs
    from modelforge.potential.utils import NNPInput

from modelforge.potential.utils import NeuralNetworkData


@dataclass
class SpookyNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for SpookyNet-based neural network potentials, including the necessary
    geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff
    (`f_cutoff`) to accurately represent atomistic systems for energy predictions.

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
    atomic_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived from atomic numbers.
        Shape: [num_atoms, embedding_dim], where `embedding_dim` is the dimensionality of the embedding vectors.
    f_ij : Optional[torch.Tensor]
        A tensor representing the radial symmetry function expansion of distances between atom pairs, capturing the
        local chemical environment. Shape: [num_pairs, number_of_atom_features], where `number_of_atom_features` is the dimensionality of
        the radial symmetry function expansion. This field will be populated after initialization.
    f_cutoff : Optional[torch.Tensor]
        A tensor representing the cosine cutoff function applied to the radial symmetry function expansion, ensuring
        that atom pair contributions diminish smoothly to zero at the cutoff radius. Shape: [num_pairs]. This field
        will be populated after initialization.

    Notes
    -----
    The `SpookyNetNeuralNetworkData` class is designed to encapsulate all necessary inputs for SpookyNet-based neural network
    potentials in a structured and type-safe manner, facilitating efficient and accurate processing of input data by
    the model. The inclusion of radial symmetry functions (`f_ij`) and cosine cutoff functions (`f_cutoff`) allows
    for a detailed and nuanced representation of the atomistic systems, crucial for the accurate prediction of system
    energies and properties.

    Examples
    --------
    >>> inputs = SpookyNetNeuralNetworkData(
    ...     pair_indices=torch.tensor([[0, 1], [0, 2], [1, 2]]),
    ...     d_ij=torch.tensor([1.0, 1.0, 1.0]),
    ...     r_ij=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ...     number_of_atoms=3,
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ...     atomic_numbers=torch.tensor([1, 6, 8]),
    ...     atomic_subsystem_indices=torch.tensor([0, 0, 0]),
    ...     total_charge=torch.tensor([0.0]),
    ...     atomic_embedding=torch.randn(3, 5),  # Example atomic embeddings
    ...     f_ij=torch.randn(3, 4),  # Example radial symmetry function expansion
    ...     f_cutoff=torch.tensor([0.5, 0.5, 0.5])  # Example cosine cutoff function
    ... )
    """

    atomic_embedding: torch.Tensor
    f_ij: Optional[torch.Tensor] = field(default=None)
    f_cutoff: Optional[torch.Tensor] = field(default=None)


class SpookyNetCore(CoreNetwork):
    def __init__(
            self,
            max_Z: int = 100,
            number_of_atom_features: int = 64,
            number_of_radial_basis_functions: int = 20,
            number_of_interaction_modules: int = 3,
            number_of_residual_blocks: int = 7,
            cutoff: unit.Quantity = 5.0 * unit.angstrom,
    ) -> None:
        """
        Initialize the SpookyNet class.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        number_of_interaction_modules : int, default=2
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """
        from .utils import Dense, ShiftedSoftplus

        log.debug("Initializing SpookyNet model.")
        super().__init__()
        self.number_of_atom_features = number_of_atom_features
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, number_of_atom_features)

        # initialize representation block
        self.spookynet_representation_module = SpookyNetRepresentation(cutoff, number_of_radial_basis_functions)

        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SpookyNetInteractionModule(
                    number_of_atom_features=number_of_atom_features,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    num_residual_pre=number_of_residual_blocks,
                    num_residual_local_x=number_of_residual_blocks,
                    num_residual_local_s=number_of_residual_blocks,
                    num_residual_local_p=number_of_residual_blocks,
                    num_residual_local_d=number_of_residual_blocks,
                    num_residual_local=number_of_residual_blocks,
                    num_residual_nonlocal_q=number_of_residual_blocks,
                    num_residual_nonlocal_k=number_of_residual_blocks,
                    num_residual_nonlocal_v=number_of_residual_blocks,
                    num_residual_post=number_of_residual_blocks,
                    num_residual_output=number_of_residual_blocks,
        )
                for _ in range(number_of_interaction_modules)
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

    def _model_specific_input_preparation(
            self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> SpookyNetNeuralNetworkData:
        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = SpookyNetNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
            atomic_embedding=self.embedding_module(
                data.atomic_numbers
            ),  # atom embedding
        )

        return nnp_input

    def compute_properties(self, data: SpookyNetNeuralNetworkData) -> Dict[str, torch.Tensor]:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        data : NamedTuple

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated energies; shape (nr_systems,).
        """

        # Compute the representation for each atom (transform to radial basis set, multiply by cutoff)
        representation = self.spookynet_representation_module(data.d_ij, data.r_ij)
        x = data.atomic_embedding

        f = x.new_zeros(x.size())  # initialize output features to zero
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            x, y = interaction(
                x=x,
                pairlist=data.pair_indices,
                filters=representation["filters"],
                dir_ij=representation["dir_ij"],
                d_orbital_ij=representation["d_orbital_ij"],
            )
            f += y  # accumulate module output to features

        E_i = self.energy_layer(x).squeeze(1)

        return {
            "E_i": E_i,
            "q": x,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


from .models import InputPreparation, NNPInput, BaseNetwork


class SpookyNet(BaseNetwork):
    def __init__(
            self,
            max_Z: int,
            number_of_atom_features: int,
            number_of_radial_basis_functions: int,
            number_of_interaction_modules: int,
            number_of_residual_blocks: int,
            cutoff: unit.Quantity,
            postprocessing_parameter: Dict[str, Dict[str, bool]],
            dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the SpookyNet network.
        
        Unke, O.T., Chmiela, S., Gastegger, M. et al. SpookyNet: Learning force fields with electronic degrees of
        freedom and nonlocal effects. Nat Commun 12, 7273 (2021).

        Parameters
        ----------
        max_Z : int
            Maximum atomic number to be embedded.
        number_of_atom_features : int
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions :int
        number_of_interaction_modules : int
        cutoff : openff.units.unit.Quantity
            The cutoff distance for interactions.
        """
        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
        )
        from modelforge.utils.units import _convert

        self.core_module = SpookyNetCore(
            max_Z=max_Z,
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_residual_blocks=number_of_residual_blocks,
        )
        self.only_unique_pairs = False  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=_convert(cutoff), only_unique_pairs=self.only_unique_pairs
        )

    def _config_prior(self):
        log.info("Configuring SpookyNet model hyperparameter prior distribution")
        from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior


class SpookyNetRepresentation(nn.Module):

    def __init__(
            self,
            cutoff: unit = 5 * unit.angstrom,
            number_of_radial_basis_functions: int = 16,
    ):
        """
        Representation module for the PhysNet potential, handling the generation of
        the radial basis functions (RBFs) with a cutoff.

        Parameters
        ----------
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        number_of_radial_basis_functions : int, default=16
            Number of radial basis functions
        """

        super().__init__()

        # cutoff
        # radial symmetry function
        from .utils import ExponentialBernsteinRadialBasisFunction, CosineCutoff

        self.radial_symmetry_function_module = ExponentialBernsteinRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            ini_alpha=1.0,  # TODO: put the right number
            dtype=torch.float32,
        )

        self.cutoff_module = CosineCutoff(cutoff=cutoff)

    def forward(self, d_ij: torch.Tensor, r_ij: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the representation module.

        Parameters
        ----------
        d_ij : torch.Tensor
            pairwise distances between atoms, shape [num_pairs, 1].
        r_ij : torch.Tensor
            pairwise displacements between atoms, shape [num_pairs, 3].

        Returns
        -------
        torch.Tensor
            The radial basis function expansion applied to the input distances,
            shape (n_pairs, n_gaussians), after applying the cutoff function.
        """

        sqrt3 = math.sqrt(3)
        sqrt3half = 0.5 * sqrt3
        # short-range distances
        dir_ij = r_ij / d_ij
        d_orbital_ij = torch.stack(
            [
                sqrt3 * dir_ij[:, 0] * dir_ij[:, 1],  # xy
                sqrt3 * dir_ij[:, 0] * dir_ij[:, 2],  # xz
                sqrt3 * dir_ij[:, 1] * dir_ij[:, 2],  # yz
                0.5 * (3 * dir_ij[:, 2] * dir_ij[:, 2] - 1.0),  # z2
                sqrt3half
                * (dir_ij[:, 0] * dir_ij[:, 0] - dir_ij[:, 1] * dir_ij[:, 1]),  # x2-y2
            ],
            dim=-1,
        )
        f_ij = self.radial_symmetry_function_module(d_ij)
        f_ij_cutoff = self.cutoff_module(d_ij)
        filters = f_ij * f_ij_cutoff  # TODO: replace with einsum

        return {"filters": filters, "dir_ij": dir_ij, "d_orbital_ij": d_orbital_ij}


class Swish(nn.Module):
    """
    Swish activation function with learnable feature-wise parameters:
    f(x) = alpha*x * sigmoid(beta*x)
    sigmoid(x) = 1/(1 + exp(-x))
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the "linear component".
        initial_beta (float):
            Initial "temperature" of the "sigmoid component". The default value
            of 1.702 has the effect of initializing swish to an approximation
            of the Gaussian Error Linear Unit (GELU) activation function from
            Hendrycks, Dan, and Gimpel, Kevin. "Gaussian error linear units
            (GELUs)."
    """

    def __init__(
            self, number_of_atom_features: int, initial_alpha: float = 1.0, initial_beta: float = 1.702
    ) -> None:
        """ Initializes the Swish class. """
        super(Swish, self).__init__()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(number_of_atom_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(number_of_atom_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters alpha and beta. """
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, number_of_atom_features]):
                Input features.

        Returns:
            y (FloatTensor [:, number_of_atom_features]):
                Activated features.
        """
        return self.alpha * F.silu(self.beta * x)


class SpookyNetResidual(nn.Module):
    """
    Pre-activation residual block inspired by He, Kaiming, et al. "Identity
    mappings in deep residual networks.".

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
    """

    def __init__(
            self,
            number_of_atom_features: int,
            bias: bool = True,
    ) -> None:
        """ Initializes the Residual class. """
        super(SpookyNetResidual, self).__init__()
        # initialize attributes
        self.activation1 = Swish(number_of_atom_features)
        self.linear1 = nn.Linear(number_of_atom_features, number_of_atom_features, bias=bias)
        self.activation2 = Swish(number_of_atom_features)
        self.linear2 = nn.Linear(number_of_atom_features, number_of_atom_features, bias=bias)
        self.reset_parameters(bias)

    def reset_parameters(self, bias: bool = True) -> None:
        """ Initialize parameters to compute an identity mapping. """
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input atomic features.
        N: Number of atoms.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Input feature representations of atoms.

        Returns:
            y (FloatTensor [N, number_of_atom_features]):
                Output feature representations of atoms.
        """
        y = self.activation1(x)
        y = self.linear1(y)
        y = self.activation2(y)
        y = self.linear2(y)
        return x + y


class SpookyNetResidualStack(nn.Module):
    """
    Stack of num_blocks pre-activation residual blocks evaluated in sequence.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_residual_blocks (int):
            Number of residual blocks to be stacked in sequence.
    """

    def __init__(
            self,
            number_of_atom_features: int,
            number_of_residual_blocks: int,
            bias: bool = True,
    ) -> None:
        """ Initializes the ResidualStack class. """
        super(SpookyNetResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [
                SpookyNetResidual(number_of_atom_features, bias)
                for _ in range(number_of_residual_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies all residual blocks to input features in sequence.
        N: Number of inputs.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Input feature representations.

        Returns:
            y (FloatTensor [N, number_of_atom_features]):
                Output feature representations.
        """
        for residual in self.stack:
            x = residual(x)
        return x


class SpookyNetResidualMLP(nn.Module):
    def __init__(
            self,
            number_of_atom_features: int,
            number_of_residual_blocks: int,
            bias: bool = True,
    ) -> None:
        super(SpookyNetResidualMLP, self).__init__()
        self.residual = SpookyNetResidualStack(
            number_of_atom_features, number_of_residual_blocks, bias=bias
        )
        self.activation = Swish(number_of_atom_features)
        self.linear = nn.Linear(number_of_atom_features, number_of_atom_features, bias=bias)
        self.reset_parameters(bias)

    def reset_parameters(self, bias: bool = True) -> None:
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.residual(x)))


class SpookyNetLocalInteraction(nn.Module):
    """
    Block for updating atomic features through local interactions with
    neighboring atoms (message-passing).

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_radial_basis_functions (int):
            Number of radial basis functions.
        num_residual_x (int):
            TODO
        num_residual_s (int):
            TODO
        num_residual_p (int):
            TODO
        num_residual_d (int):
            TODO
        num_residual (int):
            Number of residual blocks to be stacked in sequence.
    """

    def __init__(
            self,
            number_of_atom_features: int,
            number_of_radial_basis_functions: int,
            num_residual_x: int,
            num_residual_s: int,
            num_residual_p: int,
            num_residual_d: int,
            num_residual: int,
    ) -> None:
        """ Initializes the LocalInteraction class. """
        super(SpookyNetLocalInteraction, self).__init__()
        self.radial_s = nn.Linear(number_of_radial_basis_functions, number_of_atom_features, bias=False)
        self.radial_p = nn.Linear(number_of_radial_basis_functions, number_of_atom_features, bias=False)
        self.radial_d = nn.Linear(number_of_radial_basis_functions, number_of_atom_features, bias=False)
        self.resblock_x = SpookyNetResidualMLP(number_of_atom_features, num_residual_x)
        self.resblock_s = SpookyNetResidualMLP(number_of_atom_features, num_residual_s)
        self.resblock_p = SpookyNetResidualMLP(number_of_atom_features, num_residual_p)
        self.resblock_d = SpookyNetResidualMLP(number_of_atom_features, num_residual_d)
        self.projection_p = nn.Linear(number_of_atom_features, 2 * number_of_atom_features, bias=False)
        self.projection_d = nn.Linear(number_of_atom_features, 2 * number_of_atom_features, bias=False)
        self.resblock = SpookyNetResidualMLP(
            number_of_atom_features, num_residual
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        nn.init.orthogonal_(self.radial_s.weight)
        nn.init.orthogonal_(self.radial_p.weight)
        nn.init.orthogonal_(self.radial_d.weight)
        nn.init.orthogonal_(self.projection_p.weight)
        nn.init.orthogonal_(self.projection_d.weight)

    def forward(
            self,
            x_tilde: torch.Tensor,
            f_ij_after_cutoff: torch.Tensor,
            dir_ij: torch.Tensor,
            d_orbital_ij: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.
        P: Number of atom pairs.

        x (FloatTensor [N, number_of_atom_features]):
            Atomic feature vectors.
        rbf (FloatTensor [N, number_of_radial_basis_functions]):
            Values of the radial basis functions for the pairwise distances.
        dir_ij (TODO:):
            TODO:
        d_orbital_ij (TODO):
            TODO:
        idx_i (LongTensor [P]):
            Index of atom i for all atomic pairs ij. Each pair must be
            specified as both ij and ji.
        idx_j (LongTensor [P]):
            Same as idx_i, but for atom j.
        """
        # interaction functions
        gs = self.radial_s(f_ij_after_cutoff)
        gp = self.radial_p(f_ij_after_cutoff).unsqueeze(-2) * dir_ij.unsqueeze(-1)  # TODO: replace with einsum
        gd = self.radial_d(f_ij_after_cutoff).unsqueeze(-2) * d_orbital_ij.unsqueeze(-1)  # TODO: replace with einsum
        # atom featurizations
        xx = self.resblock_x(x_tilde)
        xs = self.resblock_s(x_tilde)
        xp = self.resblock_p(x_tilde)
        xd = self.resblock_d(x_tilde)
        # collect neighbors
        xs = xs[idx_j]  # L=0
        xp = xp[idx_j]  # L=1
        xd = xd[idx_j]  # L=2
        # sum over neighbors
        pp = x_tilde.new_zeros(x_tilde.shape[0], dir_ij.shape[-1], x_tilde.shape[-1])
        dd = x_tilde.new_zeros(x_tilde.shape[0], d_orbital_ij.shape[-1], x_tilde.shape[-1])
        s = xx.index_add(0, idx_i, gs * xs)  # L=0 # TODO: replace with einsum
        p = pp.index_add_(0, idx_i, gp * xp.unsqueeze(-2))  # L=1 # TODO: replace with einsum
        d = dd.index_add_(0, idx_i, gd * xd.unsqueeze(-2))  # L=2 # TODO: replace with einsum
        # project tensorial features to scalars
        pa, pb = torch.split(self.projection_p(p), p.shape[-1], dim=-1)
        da, db = torch.split(self.projection_d(d), d.shape[-1], dim=-1)
        return self.resblock(s + (pa * pb).sum(-2) + (da * db).sum(-2))  # TODO: replace with einsum


class SpookyNetAttention(nn.Module):
    """
    Efficient (linear scaling) approximation for attention described in
    Choromanski, K., et al. "Rethinking Attention with Performers.".

    Arguments:
        dim_qk (int):
            Dimension of query/key vectors.
        num_random_features (int):
            Number of random features for approximating attention matrix. If
            this is 0, the exact attention matrix is computed.
    """

    def __init__(
            self, dim_qk: int, num_random_features: int
    ) -> None:
        """ Initializes the Attention class. """
        super(SpookyNetAttention, self).__init__()
        self.num_random_features = num_random_features
        omega = self._omega(num_random_features, dim_qk)
        self.register_buffer("omega", torch.tensor(omega, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def _omega(self, nrows: int, ncols: int) -> np.ndarray:
        """ Return a (nrows x ncols) random feature matrix. """
        nblocks = int(nrows / ncols)
        blocks = []
        for i in range(nblocks):
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q))
        missing_rows = nrows - nblocks * ncols
        if missing_rows > 0:
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q)[:missing_rows])
        norm = np.linalg.norm(  # renormalize rows so they still follow N(0,1)
            np.random.normal(size=(nrows, ncols)), axis=1, keepdims=True
        )
        return (norm * np.vstack(blocks)).T

    def _phi(
            self,
            X: torch.Tensor,
            is_query: bool,
            eps: float = 1e-4,
    ) -> torch.Tensor:
        """ Normalize X and project into random feature space. """
        d = X.shape[-1]
        m = self.omega.shape[-1]
        U = torch.matmul(X / d ** 0.25, self.omega)
        h = torch.sum(X ** 2, dim=-1, keepdim=True) / (2 * d ** 0.5)  # OLD
        # determine maximum (is subtracted to prevent numerical overflow)
        if is_query:
            maximum, _ = torch.max(U, dim=-1, keepdim=True)
        else:
            maximum = torch.max(U)
        return (torch.exp(U - h - maximum) + eps) / math.sqrt(m)

    def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute attention for the given query, key and value vectors.
        N: Number of input values.
        dim_qk: Dimension of query/key vectors.
        dim_v: Dimension of value vectors.

        Arguments:
            Q (FloatTensor [N, dim_qk]):
                Matrix of N query vectors.
            K (FloatTensor [N, dim_qk]):
                Matrix of N key vectors.
            V (FloatTensor [N, dim_v]):
                Matrix of N value vectors.
            eps (float):
                Small constant to prevent numerical instability.
        Returns:
            y (FloatTensor [N, dim_v]):
                Attention-weighted sum of value vectors.
        """
        Q = self._phi(Q, True)  # random projection of Q
        K = self._phi(K, False)  # random projection of K
        norm = Q @ torch.sum(K, 0, keepdim=True).T + eps
        return (Q @ (K.T @ V)) / norm


class SpookyNetNonlocalInteraction(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with all
    atoms.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        num_residual_q (int):
            Number of residual blocks for queries.
        num_residual_k (int):
            Number of residual blocks for keys.
        num_residual_v (int):
            Number of residual blocks for values.
    """

    def __init__(
            self,
            number_of_atom_features: int,
            num_residual_q: int,
            num_residual_k: int,
            num_residual_v: int,
    ) -> None:
        """ Initializes the NonlocalInteraction class. """
        super(SpookyNetNonlocalInteraction, self).__init__()
        self.resblock_q = SpookyNetResidualMLP(
            number_of_atom_features, num_residual_q
        )
        self.resblock_k = SpookyNetResidualMLP(
            number_of_atom_features, num_residual_k
        )
        self.resblock_v = SpookyNetResidualMLP(
            number_of_atom_features, num_residual_v
        )
        self.attention = SpookyNetAttention(dim_qk=number_of_atom_features, num_random_features=number_of_atom_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(
            self,
            x_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, number_of_atom_features]):
            Atomic feature vectors.
        """
        q = self.resblock_q(x_tilde)  # queries
        k = self.resblock_k(x_tilde)  # keys
        v = self.resblock_v(x_tilde)  # values
        return self.attention(q, k, v)


class SpookyNetInteractionModule(nn.Module):
    """
    InteractionModule of SpookyNet, which computes a single iteration.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_radial_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features before
            interaction with neighbouring atoms.
        num_residual_local_x (int):
            TODO
        num_residual_local_s (int):
            TODO
        num_residual_local_p (int):
            TODO
        num_residual_local_d (int):
            TODO
        num_residual_local (int):
            TODO
        num_residual_nonlocal_q (int):
            Number of residual blocks for queries in nonlocal interactions.
        num_residual_nonlocal_k (int):
            Number of residual blocks for keys in nonlocal interactions.
        num_residual_nonlocal_v (int):
            Number of residual blocks for values in nonlocal interactions.
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms.
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch.
    """

    def __init__(
            self,
            number_of_atom_features: int,
            number_of_radial_basis_functions: int,
            num_residual_pre: int,
            num_residual_local_x: int,
            num_residual_local_s: int,
            num_residual_local_p: int,
            num_residual_local_d: int,
            num_residual_local: int,
            num_residual_nonlocal_q: int,
            num_residual_nonlocal_k: int,
            num_residual_nonlocal_v: int,
            num_residual_post: int,
            num_residual_output: int,
    ) -> None:
        """ Initializes the InteractionModule class. """
        super(SpookyNetInteractionModule, self).__init__()
        # initialize modules
        self.local_interaction = SpookyNetLocalInteraction(
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            num_residual_x=num_residual_local_x,
            num_residual_s=num_residual_local_s,
            num_residual_p=num_residual_local_p,
            num_residual_d=num_residual_local_d,
            num_residual=num_residual_local,
        )
        self.nonlocal_interaction = SpookyNetNonlocalInteraction(
            number_of_atom_features=number_of_atom_features,
            num_residual_q=num_residual_nonlocal_q,
            num_residual_k=num_residual_nonlocal_k,
            num_residual_v=num_residual_nonlocal_v,
        )

        self.residual_pre = SpookyNetResidualStack(number_of_atom_features, num_residual_pre)
        self.residual_post = SpookyNetResidualStack(number_of_atom_features, num_residual_post)
        self.resblock = SpookyNetResidualMLP(number_of_atom_features, num_residual_output)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(
            self,
            x: torch.Tensor,
            pairlist: torch.Tensor,  # shape [n_pairs, 2]
            filters: torch.Tensor,  # shape [n_pairs, 1, number_of_radial_basis_functions] TODO: why the 1?
            dir_ij: torch.Tensor,  # shape [n_pairs, 1]
            d_orbital_ij: torch.Tensor,  # shape [n_pairs, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate all modules in the block.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Latent atomic feature vectors.
            rbf (FloatTensor [P, number_of_radial_basis_functions]):
                Values of the radial basis functions for the pairwise distances.
            dir_ij (FloatTensor [P, 3]):
                Unit vectors pointing from atom i to atom j for all atomic pairs.
            d_orbital_ij (FloatTensor [P]):
                Distances between atom i and atom j for all atomic pairs.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.
        Returns:
            x (FloatTensor [N, number_of_atom_features]):
                Updated latent atomic feature vectors.
            y (FloatTensor [N, number_of_atom_features]):
                Contribution to output atomic features (environment
                descriptors).
        """
        idx_i, idx_j = pairlist[0], pairlist[1]
        x_tilde = self.residual_pre(x)
        del x
        l = self.local_interaction(
            x_tilde=x_tilde,
            f_ij_after_cutoff=filters,
            dir_ij=dir_ij,
            d_orbital_ij=d_orbital_ij,
            idx_i=idx_i,
            idx_j=idx_j,
        )
        n = self.nonlocal_interaction(x_tilde)
        x_updated = self.residual_post(x_tilde + l + n)
        del x_tilde
        return x_updated, self.resblock(x_updated)

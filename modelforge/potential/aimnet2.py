from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

if TYPE_CHECKING:
    from .models import PairListOutputs
    from modelforge.dataset.dataset import NNPInput

from modelforge.potential.utils import NeuralNetworkData
from .models import InputPreparation, NNPInput, BaseNetwork, CoreNetwork
from .utils import RadialSymmetryFunction


class AIMNet2RadialSymmetryFunction(RadialSymmetryFunction):

    def calculate_radial_scale_factor(
        self,
        _min_distance_in_nanometer,
        _max_distance_in_nanometer,
        number_of_radial_basis_functions,
    ):
        scale_factors = torch.linspace(
            _min_distance_in_nanometer,
            _max_distance_in_nanometer,
            number_of_radial_basis_functions,
        )

        widths = (
            torch.abs(scale_factors[1] - scale_factors[0])
            * torch.ones_like(scale_factors)
        ).to(self.dtype)

        scale_factors = 0.5 / torch.square_(widths)
        return scale_factors


@dataclass
class AIMNet2NeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for AIMNet2-based neural network potentials, including the necessary
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
        local chemical environment. Shape: [num_pairs, num_features], where `num_features` is the dimensionality of
        the radial symmetry function expansion. This field will be populated after initialization.
    f_cutoff : Optional[torch.Tensor]
        A tensor representing the cosine cutoff function applied to the radial symmetry function expansion, ensuring
        that atom pair contributions diminish smoothly to zero at the cutoff radius. Shape: [num_pairs]. This field
        will be populated after initialization.

    Notes
    -----
    The `SchnetNeuralNetworkInput` class is designed to encapsulate all necessary inputs for SchNet-based neural network
    potentials in a structured and type-safe manner, facilitating efficient and accurate processing of input data by
    the model. The inclusion of radial symmetry functions (`f_ij`) and cosine cutoff functions (`f_cutoff`) allows
    for a detailed and nuanced representation of the atomistic systems, crucial for the accurate prediction of system
    energies and properties.

    Examples
    --------
    >>> inputs = SchnetNeuralNetworkInput(
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


class AIMNet2Core(CoreNetwork):
    def __init__(
        self,
        max_Z: int = 100,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 20,
        number_of_interaction_modules: int = 3,
        number_of_filters: int = 64,
        shared_interactions: bool = False,
        cutoff: unit.Quantity = 5.0 * unit.angstrom,
    ) -> None:
        """
        Initialize the SchNet class.

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

        # classes that are modelspecific
        self.atomic_number_embedding = None
        self.representation_module = None
        self.interaction_module = None

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> AIMNet2NeuralNetworkData:

        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = AIMNet2NeuralNetworkData(
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

    def compute_properties(
        self, data: AIMNet2NeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the requested properties for a given input batch.

        Parameters
        ----------
        data : NamedTuple

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated energies; shape (nr_systems,).
        """
        pass


class AimNet2Representation(nn.Module):
    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_radial_basis_functions: int,
    ):
        """
        Initialize the SchNet representation layer.

        Parameters
        ----------
        Radial Basis Function Module
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: unit.Quantity, number_of_radial_basis_functions: int
    ):
        # NOTE: for now we are using SchNET radial symmetry function, this
        # should be replaced with the AIMNet2 radial symmetry function
        from .utils import SchnetRadialSymmetryFunction

        radial_symmetry_function = SchnetRadialSymmetryFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, d_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        d_ij : Pairwise distances between atoms; shape [n_pairs, 1]

        Returns
        -------
        Radial basis functions for pairs of atoms; shape [n_pairs, 1, number_of_radial_basis_functions]
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(
            d_ij
        )  # shape (n_pairs, 1, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(d_ij)  # shape (n_pairs, 1)

        return {"f_ij": f_ij, "f_cutoff": f_cutoff}


from typing import List


class AIMNet2(BaseNetwork):
    def __init__(
        self,
        max_Z: int,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        cutoff: unit.Quantity,
        number_of_filters: int,
        shared_interactions: bool,
        processing_operation: List[Dict[str, str]],
        readout_operation: List[Dict[str, str]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the AIMNet2 network.

        # NOTE: set correct reference

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
        super().__init__(
            dataset_statistic=dataset_statistic,
            processing_operation=processing_operation,
            readout_operation=readout_operation,
        )
        from modelforge.utils.units import _convert

        self.core_module = AIMNet2Core(
            max_Z=max_Z,
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_filters=number_of_filters,
            shared_interactions=shared_interactions,
        )
        self.only_unique_pairs = False  # NOTE: for pairlist
        self.input_preparation = InputPreparation(
            cutoff=_convert(cutoff), only_unique_pairs=self.only_unique_pairs
        )

    def _config_prior(self):
        pass

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData
from .models import NNPInput, BaseNetwork, CoreNetwork


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
    >>> inputs = AIMNet2NeuralNetworkInput(
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


from typing import Union, Type


class AIMNet2Core(CoreNetwork):
    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_filters: int,
        shared_interactions: bool,
        activation_function: Type[torch.nn.Module],
        maximum_interaction_radius: unit.Quantity,
    ) -> None:
        """
        Initialize the AIMNet2 class.

        Parameters
        ----------
        max_Z : int, default=100
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        number_of_interaction_modules : int, default=3
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """

        log.debug("Initializing the AimNet2 architecture.")

        super().__init__(activation_function)

        # Initialize representation block
        self.schnet_representation_module = AIMNet2Representation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )

        self.interaction_modules = nn.ModuleList(
            [
                # first pass
                AIMNet2InteractionModule(
                    number_of_atom_features,
                    number_of_radial_basis_functions,
                    first_pass=True,
                ),
            ]  # all other passes (num - 1)
            + [
                AIMNet2InteractionModule(
                    number_of_atom_features,
                    number_of_radial_basis_functions,
                    first_pass=False,
                ),
            ]
            * (number_of_interaction_modules - 1)
        )

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
            atomic_embedding=self.embedding_module(data.atomic_numbers),
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

        representation = self.representation_module(data.d_ij)

        data.f_ij = representation["f_ij"]
        data.f_cutoff = representation["f_cutoff"]

        # Atomic embedding "a" Eqn. (3)
        embedding = data.atomic_embedding

        # first pass
        delta_a, partial_point_charges = self.interaction_modules(
            embedding,
            data.pair_indices,
            data.f_ij,
            data.f_cutoff,
        )

        embedding += delta_a

        # subsequent passes
        for interaction in self.interaction_modules[1:]:
            delta_a, delta_q = interaction(
                embedding,
                data.pair_indices,
                data.f_ij,
                data.f_cutoff,
            )
            embedding += delta_a
            # TODO: implement nqe
            self.nqe(partial_point_charges, delta_q)

        raise NotImplementedError

    def nqe(self, partial_point_charges, delta_q):
        raise NotImplementedError


class AIMNet2InteractionModule(nn.Module):

    def __init__(
        self,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        first_pass: bool = False,
    ):
        super().__init__()

        # if this is the first pass, charge information is not needed
        self.first_pass = first_pass

        # TODO: include assertions like those found in schnet?
        self.number_of_atomic_features = number_of_atom_features
        self.input_to_feature = None
        self.feature_to_output = None

    def forward(
        self,
        # input_features: torch.Tensor,
        pairlist: torch.Tensor,
        f_ij: torch.Tensor,  # this is already from the representation, radial symmetry function module with distances preloaded
        f_cutoff: torch.Tensor,  # cutoff module with the distances preloaded
        atomic_embedding: torch.Tensor,  # outputs need to be the same shape of atomic_embedding, [N_atoms, atom_basis (100?)]
        partial_point_charges: Optional[torch.Tensor] = None,
    ):

        # NOTE: fixed "a" (atomic_embedding) and "q" (partial_point_charges)
        # if partial_point_charges is None, then we know we're in the first
        # pass

        # Eqn (2)
        g = f_ij * f_cutoff

        idx_i, idx_j = pairlist[0], pairlist[1]

        # required in all passes
        v_radial_atomic, v_vector_atomic = None, None

        # required in 1 + i passes
        if partial_point_charges:
            v_radial_charge, v_vector_charge = None, None

        raise NotImplementedError


class AIMNet2Representation(nn.Module):
    def __init__(
        self,
        radial_cutoff: unit.Quantity,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Union[List[str], int]],
    ):
        """
        Initialize the AIMNet2 representation layer.

        Parameters
        ----------
        Radial Basis Function Module
        """
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        # Initialize cutoff module
        from modelforge.potential import CosineAttenuationFunction, FeaturizeInput

        self.featurize_input = FeaturizeInput(featurization_config)
        self.cutoff_module = CosineAttenuationFunction(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: unit.Quantity, number_of_radial_basis_functions: int
    ):
        from .utils import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, data: Type[AIMNet2NeuralNetworkData]) -> Dict[str, torch.Tensor]:
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
            data.d_ij
        )  # shape (n_pairs, 1, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(data.d_ij)  # shape (n_pairs, 1)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(
                data
            ),  # add per-atom properties and embedding
        }


from typing import List


class AIMNet2(BaseNetwork):
    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        maximum_interaction_radius: Union[unit.Quantity, str],
        number_of_filters: int,
        activation_function_parameter: Dict,
        shared_interactions: bool,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        potential_seed: Optional[int] = None,
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
        self.only_unique_pairs = False  # NOTE: need to be set before super().__init__
        from modelforge.utils.units import _convert_str_to_unit

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            potential_seed=potential_seed,
        )

        activation_function = activation_function_parameter["activation_function"]

        self.core_module = AIMNet2Core(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_filters=number_of_filters,
            shared_interactions=shared_interactions,
            activation_function=activation_function,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
        )

    def _config_prior(self):
        pass

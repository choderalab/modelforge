from typing import Dict, Optional

import torch
from loguru import logger as log
import torch.nn as nn

from .models import BaseNeuralNetworkPotential
from openff.units import unit
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PairListOutputs
    from modelforge.potential.utils import NNPInput

from modelforge.potential.utils import NeuralNetworkData


@dataclass
class SchnetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for SchNet-based neural network potentials, including the necessary
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


class SchNet(BaseNeuralNetworkPotential):
    def __init__(
        self,
        max_Z: int = 100,
        number_of_atom_features: int = 64,
        number_of_radial_basis_functions: int = 20,
        number_of_interaction_modules: int = 3,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        number_of_filters: int = 64,
        shared_interactions: bool = False,
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
        from .utils import ShiftedSoftplus, Dense

        log.debug("Initializing SchNet model.")
        self.only_unique_pairs = False  # NOTE: for pairlist
        super().__init__(cutoff=cutoff)
        self.number_of_atom_features = number_of_atom_features
        self.number_of_filters = number_of_filters or self.number_of_atom_features
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, number_of_atom_features)

        # initialize the energy readout
        from .utils import FromAtomToMoleculeReduction

        self.readout_module = FromAtomToMoleculeReduction()

        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            cutoff, number_of_radial_basis_functions
        )
        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SchNETInteractionModule(
                    self.number_of_atom_features,
                    self.number_of_filters,
                    number_of_radial_basis_functions,
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

    def _config_prior(self):
        log.info("Configuring SchNet model hyperparameter prior distribution")
        from ray import tune
        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "number_of_filters": tune.randint(32, 128),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> SchnetNeuralNetworkData:
        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = SchnetNeuralNetworkData(
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

    def _forward(self, data: SchnetNeuralNetworkData) -> Dict[str, torch.Tensor]:
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
        representation = self.schnet_representation_module(data.d_ij)
        data.f_ij = representation["f_ij"]
        data.f_cutoff = representation["f_cutoff"]
        x = data.atomic_embedding
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                x,
                data.pair_indices,
                representation["f_ij"],
                representation["f_cutoff"],
            )
            x = x + v  # Update atomic features

        E_i = self.energy_layer(x).squeeze(1)

        return {
            "E_i": E_i,
            "q": x,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


class SchNETInteractionModule(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int,
        number_of_filters: int,
        number_of_radial_basis_functions: int,
    ) -> None:
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        number_of_atom_features : int
            Number of atom ffeatures, defines the dimensionality of the embedding.
        number_of_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        """
        super().__init__()
        from .utils import ShiftedSoftplus, Dense

        assert (
            number_of_radial_basis_functions > 4
        ), "Number of radial basis functions must be larger than 10."
        assert number_of_filters > 1, "Number of filters must be larger than 1."
        assert (
            number_of_atom_features > 10
        ), "Number of atom basis must be larger than 10."

        self.number_of_atom_features = number_of_atom_features  # Initialize parameters
        self.intput_to_feature = Dense(
            number_of_atom_features, number_of_filters, bias=False, activation=None
        )
        self.feature_to_output = nn.Sequential(
            Dense(
                number_of_filters, number_of_atom_features, activation=ShiftedSoftplus()
            ),
            Dense(number_of_atom_features, number_of_atom_features, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(
                number_of_radial_basis_functions,
                number_of_filters,
                activation=ShiftedSoftplus(),
            ),
            Dense(number_of_filters, number_of_filters, activation=None),
        )

    def forward(
        self,
        x: torch.Tensor,
        pairlist: torch.Tensor,  # shape [n_pairs, 2]
        f_ij: torch.Tensor,  # shape [n_pairs, 1, number_of_radial_basis_functions]
        f_ij_cutoff: torch.Tensor,  # shape [n_pairs, 1]
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Input feature tensor for atoms.
        pairlist : torch.Tensor, shape [n_pairs, 2]
        f_ij : torch.Tensor, shape [n_pairs, 1, number_of_radial_basis_functions]
            Radial basis functions for pairs of atoms.
        f_ij_cutoff : torch.Tensor, shape [n_pairs, 1]

        Returns
        -------
        torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Updated feature tensor after interaction block.
        """

        # Map input features to the filter space
        x = self.intput_to_feature(x)

        # Generate interaction filters based on radial basis functions
        Wij = self.filter_network(f_ij.squeeze(1))

        idx_i, idx_j = pairlist[0], pairlist[1]
        x_j = x[idx_j]

        # Perform continuous-filter convolution
        x_ij = x_j * Wij * f_ij_cutoff

        # Initialize a tensor to gather the results
        x_ = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # Sum contributions to update atom features
        # x shape: torch.Size([nr_of_atoms_in_batch, 64])
        # x_ij shape: torch.Size([nr_of_pairs, 64])
        idx_i_expand = idx_i.unsqueeze(1).expand_as(x_ij)
        x_.scatter_add_(0, idx_i_expand, x_ij)
        # Map back to the original feature space and reshape
        x = self.feature_to_output(x_)
        return x


class SchNETRepresentation(nn.Module):
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

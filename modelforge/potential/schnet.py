from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData
from .models import PairListOutputs, NNPInput, BaseNetwork, CoreNetwork


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


from typing import Union, List


class SchNetCore(CoreNetwork):
    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_filters: int,
        shared_interactions: bool,
        cutoff: unit.Quantity,
    ) -> None:
        """
        Initialize the SchNet class.

        Parameters
        ----------
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for featurization, including the number of per-atom features and the maximum atomic number to be embedded.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        number_of_interaction_modules : int
            Number of interaction modules.
        number_of_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        shared_interactions : bool
            Whether to share interaction parameters across all interaction modules.
        cutoff : openff.units.unit.Quantity
            The cutoff distance for interactions.
        """

        log.debug("Initializing the SchNet architecture.")
        from modelforge.potential.utils import FeaturizeInput

        super().__init__()
        self.number_of_filters = (
            number_of_filters or featurization_config["number_of_per_atom_features"]
        )
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

        # featurize the atomic input

        self.featurize_input = FeaturizeInput(featurization_config)
        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            cutoff, number_of_radial_basis_functions
        )
        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SchNETInteractionModule(
                    featurization_config["number_of_per_atom_features"],
                    self.number_of_filters,
                    number_of_radial_basis_functions,
                )
                for _ in range(number_of_interaction_modules)
            ]
        )

        # output layer to obtain per-atom energies
        self.energy_layer = nn.Sequential(
            Dense(
                featurization_config["number_of_per_atom_features"],
                featurization_config["number_of_per_atom_features"],
                activation=ShiftedSoftplus(),
            ),
            Dense(
                featurization_config["number_of_per_atom_features"],
                1,
            ),
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: PairListOutputs
    ) -> SchnetNeuralNetworkData:
        """
        Prepare the input data for the SchNet model.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist_output : PairListOutputs
            The pairlist output.

        Returns
        -------
        SchnetNeuralNetworkData
            The prepared input data for the SchNet model.
        """
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
            atomic_embedding=self.featurize_input(
                data
            ),  # add per-atom properties and embedding
        )

        return nnp_input

    def compute_properties(
        self, data: SchnetNeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the properties for a given input batch.

        Parameters
        ----------
        data : SchnetNeuralNetworkData
            The input data for the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated properties.
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
            "per_atom_energy": E_i,
            "scalar_representation": x,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


class SchNETInteractionModule(nn.Module):
    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_filters: int,
        number_of_radial_basis_functions: int,
    ) -> None:
        """
        Initialize the SchNet interaction block.

        Parameters
        ----------
        number_of_per_atom_features : int
            Number of atom ffeatures, defines the dimensionality of the embedding.
        number_of_filters : int
            Number of filters, defines the dimensionality of the intermediate features.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        """
        super().__init__()
        from .utils import Dense, ShiftedSoftplus

        assert (
            number_of_radial_basis_functions > 4
        ), "Number of radial basis functions must be larger than 10."
        assert number_of_filters > 1, "Number of filters must be larger than 1."
        assert (
            number_of_per_atom_features > 10
        ), "Number of atom basis must be larger than 10."

        self.number_of_per_atom_features = (
            number_of_per_atom_features  # Initialize parameters
        )
        self.intput_to_feature = Dense(
            number_of_per_atom_features, number_of_filters, bias=False, activation=None
        )
        self.feature_to_output = nn.Sequential(
            Dense(
                number_of_filters,
                number_of_per_atom_features,
                activation=ShiftedSoftplus(),
            ),
            Dense(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation=None,
            ),
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
        f_ij: torch.Tensor,  # shape [n_pairs, number_of_radial_basis_functions]
        f_ij_cutoff: torch.Tensor,  # shape [n_pairs, 1]
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Input feature tensor for atoms (output of embedding).
        pairlist : torch.Tensor, shape [n_pairs, 2]
        f_ij : torch.Tensor, shape [n_pairs, 1, number_of_radial_basis_functions]
            Radial basis functions for pairs of atoms.
        f_ij_cutoff : torch.Tensor, shape [n_pairs, 1]

        Returns
        -------
        torch.Tensor, shape [nr_of_atoms_in_systems, nr_atom_basis]
            Updated feature tensor after interaction block.
        """
        idx_i, idx_j = pairlist[0], pairlist[1]

        # Map input features to the filter space
        x = self.intput_to_feature(x)

        # Generate interaction filters based on radial basis functions
        W_ij = self.filter_network(f_ij.squeeze(1))  # FIXME
        W_ij = W_ij * f_ij_cutoff

        # Perform continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * W_ij  # (nr_of_atom_pairs, nr_atom_basis)
        out = torch.zeros_like(x)
        out.scatter_add_(
            0, idx_i.unsqueeze(-1).expand_as(x_ij), x_ij
        )  # from per_atom_pair to _per_atom

        return self.feature_to_output(out)  # shape: (nr_of_atoms, 1)


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
        from .utils import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
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
        )  # shape (n_pairs, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(d_ij)  # shape (n_pairs, 1)

        return {"f_ij": f_ij, "f_cutoff": f_cutoff}


from typing import List, Union
from modelforge.utils.units import _convert
from modelforge.utils.io import import_
from modelforge.potential.utils import shared_config_prior


class SchNet(BaseNetwork):
    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        cutoff: Union[unit.Quantity, str],
        number_of_filters: int,
        shared_interactions: bool,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the SchNet network.

        Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
        SchNet: A continuous-filter convolutional neural network for modeling quantum
        interactions.

        Parameters
        ----------
        featurization : Dict[str, Union[List[str], int]]
            Configuration for atom featurization.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        number_of_interaction_modules : int
            Number of interaction modules.
        cutoff : Union[unit.Quantity, str]
            The cutoff distance for interactions.
        number_of_filters : int
            Number of filters.
        shared_interactions : bool
            Whether to use shared interactions.
        postprocessing_parameter : Dict[str, Dict[str, bool]]
            Configuration for postprocessing parameters.
        dataset_statistic : Optional[Dict[str, float]], default=None
            Statistics of the dataset.

        Returns
        -------
        None
        """

        self.only_unique_pairs = False  # NOTE: need to be set before super().__init__

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            cutoff=_convert(cutoff),
        )

        self.core_module = SchNetCore(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_filters=number_of_filters,
            shared_interactions=shared_interactions,
            cutoff=_convert(cutoff),
        )

    def _config_prior(self):
        """
        Configure the SchNet model hyperparameter prior distribution.

        Returns
        -------
        dict
            The prior distribution of hyperparameters.
        """
        log.info("Configuring SchNet model hyperparameter prior distribution")

        tune = import_("ray").tune
        # from ray import tune

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "number_of_filters": tune.randint(32, 128),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior

    def combine_per_atom_properties(
        self, values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine per-atom properties.

        Parameters
        ----------
        values : Dict[str, torch.Tensor]
            Dictionary of per-atom properties.

        Returns
        -------
        torch.Tensor
            Combined per-atom properties.
        """
        return values

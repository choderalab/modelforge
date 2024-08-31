"""
SchNet neural network potential for modeling quantum interactions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union, List

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit

from modelforge.potential.utils import NeuralNetworkData
from .models import PairListOutputs, NNPInput, BaseNetwork, CoreNetwork


@torch.jit.script
@dataclass
class SchnetNeuralNetworkData(object):
    """
    A dataclass to structure the inputs specifically for SchNet-based neuraletwork potentials, including the necessary
    geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff
    (`f_cutoff`) to accurately represent atomistic systems for energy predictions.


    Note that only the arguments not present in the baseclass are described here.

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
    """

    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor
    atomic_numbers: torch.Tensor
    number_of_atoms: int
    positions: torch.Tensor
    atomic_subsystem_indices: torch.Tensor
    total_charge: torch.Tensor


class SchNetCore(nn.Module):

    def __init__(
        self,
        featurization_config: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_filters: int,
        shared_interactions: bool,
        activation_function: torch.nn.Module,
        maximum_interaction_radius: float,
    ) -> None:
        """
        Core network class for the SchNet neural network potential.

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
        maximum_interaction_radius : openff.units.unit.Quantity
            The cutoff distance for interactions.
        activation_function : Type[torch.nn.Module]
            Activation function to use.
        """
        super().__init__()
        log.debug("Initializing the SchNet architecture.")
        from modelforge.potential.utils import DenseWithCustomDist

        self.activation_function = activation_function
        atomic_number_feature = featurization_config["atomic_number"]
        self.number_of_filters = number_of_filters or int(
            atomic_number_feature["number_of_per_atom_features"]
        )
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        number_of_per_atom_features = int(
            atomic_number_feature["number_of_per_atom_features"]
        )

        # Initialize representation block
        self.schnet_representation_module = SchNETRepresentation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            featurization_config=featurization_config,
        )
        # Initialize interaction blocks
        if shared_interactions:
            self.interaction_modules = nn.ModuleList(
                [
                    SchNETInteractionModule(
                        number_of_per_atom_features,
                        self.number_of_filters,
                        number_of_radial_basis_functions,
                        activation_function=self.activation_function,
                    )
                ]
                * number_of_interaction_modules
            )

        else:
            self.interaction_modules = nn.ModuleList(
                [
                    SchNETInteractionModule(
                        number_of_per_atom_features,
                        self.number_of_filters,
                        number_of_radial_basis_functions,
                        activation_function=self.activation_function,
                    )
                    for _ in range(number_of_interaction_modules)
                ]
            )

        # output layer to obtain per-atom energies
        self.energy_layer = nn.Sequential(
            DenseWithCustomDist(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation_function=self.activation_function,
            ),
            DenseWithCustomDist(
                number_of_per_atom_features,
                1,
            ),
        )

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: Dict[str, PairListOutputs]
    ) -> SchnetNeuralNetworkData:
        """
        Prepare the input data for the SchNet model.

        Parameters
        ----------
        data : NNPInput
            The input data for the model.
        pairlist_output : Dict[str, PairListOutputs]
            The pairlist output(s).

        Returns
        -------
        SchnetNeuralNetworkData
            The prepared input data for the SchNet model.
        """
        number_of_atoms = data.atomic_numbers.shape[0]

        pairlist_output_section = pairlist_output["maximum_interaction_radius"]

        return SchnetNeuralNetworkData(
            pair_indices=pairlist_output_section.pair_indices,
            d_ij=pairlist_output_section.d_ij,
            r_ij=pairlist_output_section.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
        )

    #    @torch.jit.script
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
        # Compute the representation for each atom (transform to radial basis set, multiply by cutoff and add embedding)
        representation = self.schnet_representation_module(data)
        atomic_embedding = representation["atomic_embedding"]
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            v = interaction(
                atomic_embedding,
                data.pair_indices,
                representation["f_ij"],
                representation["f_cutoff"],
            )
            atomic_embedding = (
                atomic_embedding + v
            )  # Update per atom features given the environment

        E_i = self.energy_layer(atomic_embedding).squeeze(1)

        return {
            "per_atom_energy": E_i,
            "per_atom_scalar_representation": atomic_embedding,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

    def forward(self, nnp_input: NNPInput, pairlist_output: Dict[str, PairListOutputs]):

        # Perform model-specific modifications
        model_specific_input = self._model_specific_input_preparation(
            nnp_input, pairlist_output
        )

        # Perform the forward pass using the provided function
        outputs = self.compute_properties(model_specific_input)

        # Add atomic numbers to the output
        outputs["atomic_numbers"] = model_specific_input.atomic_numbers

        return outputs

    def load_pretrained_weights(self, path: str):
        """
        Loads pretrained weights into the model from the specified path.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()


class SchNETInteractionModule(nn.Module):
    """
    SchNet interaction module to compute interaction terms based on atomic distances and features.

    Parameters
    ----------
    number_of_per_atom_features : int
        Number of atom features, defines the dimensionality of the embedding.
    number_of_filters : int
        Number of filters, defines the dimensionality of the intermediate features.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    activation_function: Type[torch.nn.Module]
        The activation function to use in the interaction module.
    """

    def __init__(
        self,
        number_of_per_atom_features: int,
        number_of_filters: int,
        number_of_radial_basis_functions: int,
        activation_function: torch.nn.Module,
    ) -> None:

        super().__init__()
        from .utils import DenseWithCustomDist

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
        self.intput_to_feature = DenseWithCustomDist(
            number_of_per_atom_features,
            number_of_filters,
            bias=False,
        )
        self.feature_to_output = nn.Sequential(
            DenseWithCustomDist(
                number_of_filters,
                number_of_per_atom_features,
                activation_function=activation_function,
            ),
            DenseWithCustomDist(
                number_of_per_atom_features,
                number_of_per_atom_features,
            ),
        )
        self.filter_network = nn.Sequential(
            DenseWithCustomDist(
                number_of_radial_basis_functions,
                number_of_filters,
                activation_function=activation_function,
            ),
            DenseWithCustomDist(
                number_of_filters,
                number_of_filters,
            ),
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
            List of atom pairs.
        f_ij : torch.Tensor, shape [n_pairs, number_of_radial_basis_functions]
            Radial basis functions for pairs of atoms.
        f_ij_cutoff : torch.Tensor, shape [n_pairs, 1]
            Cutoff values for the pairs.

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
    """
    SchNet representation module to generate the radial symmetry representation of pairwise distances.

    Parameters
    ----------
    radial_cutoff : unit.Quantity
        The cutoff distance for interactions.
    number_of_radial_basis_functions : int
        Number of radial basis functions.
    featurization_config : Dict[str, Union[List[str], int]]
        Configuration for atom featurization.
    """

    def __init__(
        self,
        radial_cutoff: float,
        number_of_radial_basis_functions: int,
        featurization_config: Dict[str, Dict[str, int]],
    ):
        super().__init__()

        self.radial_symmetry_function_module = self._setup_radial_symmetry_functions(
            radial_cutoff, number_of_radial_basis_functions
        )
        # Initialize cutoff module
        from modelforge.potential import CosineAttenuationFunction

        self.featurize_input = torch.nn.Embedding(
            int(featurization_config["atomic_number"]["maximum_atomic_number"]),
            int(featurization_config["atomic_number"]["number_of_per_atom_features"]),
        )
        self.cutoff_module = CosineAttenuationFunction(radial_cutoff)

    def _setup_radial_symmetry_functions(
        self, radial_cutoff: float, number_of_radial_basis_functions: int
    ):
        from .utils import SchnetRadialBasisFunction

        radial_symmetry_function = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=radial_cutoff,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def forward(self, data: SchnetNeuralNetworkData) -> Dict[str, torch.Tensor]:
        """
        Generate the radial symmetry representation of the pairwise distances.

        Parameters
        ----------
        data : SchnetNeuralNetworkData

        Returns
        -------
        Dict[str, torch.Tensor]
            Radial basis functions, cutoff values for pairs of atoms and atomic embedding.
        """

        # Convert distances to radial basis functions
        f_ij = self.radial_symmetry_function_module(
            data.d_ij
        )  # shape (n_pairs, number_of_radial_basis_functions)

        f_cutoff = self.cutoff_module(data.d_ij)  # shape (n_pairs, 1)

        return {
            "f_ij": f_ij,
            "f_cutoff": f_cutoff,
            "atomic_embedding": self.featurize_input(
                data.atomic_numbers
            ),  # add per-atom properties and embedding
        }


from modelforge.potential.utils import shared_config_prior
from modelforge.potential.models import PostProcessing, ComputeInteractingAtomPairs


class SchNet(torch.nn.Module):

    def __init__(
        self,
        featurization: Dict[str, Dict[str, int]],
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        maximum_interaction_radius: float,
        number_of_filters: int,
        activation_function_parameter: Dict,
        shared_interactions: bool,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Dict[str, Dict[str, float]],
        potential_seed: int = -1,
    ) -> None:
        """
        SchNet network for modeling quantum interactions.

        Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller: SchNet: A continuous-filter
        convolutional neural network for modeling quantum interactions.
        """
        super().__init__()
        activation_function = activation_function_parameter["activation_function"]

        self.core_module = SchNetCore(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_filters=number_of_filters,
            shared_interactions=shared_interactions,
            activation_function=activation_function,
            maximum_interaction_radius=maximum_interaction_radius,
        )

        if potential_seed != -1:
            import torch, numpy, random

            torch.manual_seed(potential_seed)
            numpy.random.seed(potential_seed)
            random.seed(potential_seed)

        # self.postprocessing = PostProcessing(
        #    postprocessing_parameter, dataset_statistic
        # )

        cutoffs = {}
        cutoffs["maximum_interaction_radius"] = maximum_interaction_radius

        # create the ComputeInteractingAtomPairs object
        self.compute_interacting_pairs = ComputeInteractingAtomPairs(
            cutoffs=cutoffs,
            only_unique_pairs=False,
        )

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Load the state dictionary into the model, with optional prefix removal
        and key exclusions.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            The state dictionary to load.
        strict : bool, optional
            Whether to strictly enforce that the keys in `state_dict` match the
            keys returned by this module's `state_dict()` function (default is
            True).
        assign : bool, optional
            Whether to assign the state dictionary to the model directly
            (default is False).

        Notes
        -----
        This function can remove a specific prefix from the keys in the state
        dictionary. It can also exclude certain keys from being loaded into the
        model.
        """

        # Prefix to remove
        prefix = "potential."
        excluded_keys = ["loss.per_molecule_energy", "loss.per_atom_force"]

        # Create a new dictionary without the prefix in the keys if prefix exists
        if any(key.startswith(prefix) for key in state_dict.keys()):
            filtered_state_dict = {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
                if key not in excluded_keys
            }
            log.debug(f"Removed prefix: {prefix}")
        else:
            # Create a filtered dictionary without excluded keys if no prefix exists
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if k not in excluded_keys
            }
            log.debug("No prefix found. No modifications to keys in state loading.")

        super().load_state_dict(filtered_state_dict, strict=strict, assign=assign)

    @torch.jit.ignore
    def prepare_pairwise_properties(self, nnp_input):
        """
        Prepare the pairwise properties for the model.

        Parameters
        ----------
        nnp_input : Union[NNPInput, NamedTuple]
            The input data to prepare.

        Returns
        -------
        PairListOutputs
            The prepared pairwise properties.
        """

        self.compute_interacting_pairs._input_checks(nnp_input)
        return self.compute_interacting_pairs.prepare_inputs(nnp_input)

    def compute(
        self, data: NNPInput, neighborlist: Dict[str, PairListOutputs]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the core model's output.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            The input data.
        neighborlist : PairListOutputs
            The prepared pairwise properties.

        Returns
        -------
        Any
            The core model's output.
        """
        return self.core_module(data, neighborlist)

    def forward(self, input_data: NNPInput) -> Dict[str, torch.Tensor]:
        """
        Executes the forward pass of the model.

        This method performs input checks, prepares the inputs, and computes the
        outputs using the core network.

        Parameters
        ----------
        input_data : NNPInput
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        Any
            The outputs computed by the core network.
        """

        # compute all interacting pairs with distances
        pairwise_properties = self.prepare_pairwise_properties(input_data)
        # prepare the input for the forward pass
        output = self.compute(input_data, pairwise_properties)
        # perform postprocessing operations
        #        processed_output = self.postprocessing(output)
        return output

    def _config_prior(self):
        """
        Configure the SchNet model hyperparameter prior distribution.

        Returns
        -------
        dict
            The prior distribution of hyperparameters.
        """
        log.info("Configuring SchNet model hyperparameter prior distribution")

        from ray import tune

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "maximum_interaction_radius": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "number_of_filters": tune.randint(32, 128),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior

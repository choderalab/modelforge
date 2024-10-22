"""
This module contains the classes for the ANI2x neural network potential.

The ANI2x architecture is used for neural network potentials that compute atomic
energies based on Atomic Environment Vectors (AEVs). It supports multiple
species and interaction types, and allows prediction of properties like energy
using a neural network model.
"""

from typing import Dict, Tuple, List
import torch
from loguru import logger as log
from torch import nn

from modelforge.utils.prop import SpeciesAEV

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.neighbors import PairlistData


def triu_index(num_species: int) -> torch.Tensor:
    """
    Generate a tensor representing the upper triangular indices for species
    pairs. This is used for computing angular symmetry features, where pairwise
    combinations of species need to be considered.

    Parameters
    ----------
    num_species : int
        The number of species in the system.

    Returns
    -------
    torch.Tensor
        A tensor containing the pair indices.
    """
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index

    return ret


# A map from atomic number to an internal index used for species-specific
# computations.
ATOMIC_NUMBER_TO_INDEX_MAP = {
    1: 0,  # H
    6: 1,  # C
    7: 2,  # N
    8: 3,  # O
    9: 4,  # F
    16: 5,  # S
    17: 6,  # Cl
}


class ANIRepresentation(nn.Module):
    """
    Compute the Atomic Environment Vectors (AEVs) for the ANI architecture. AEVs
    are representations of the local atomic environment used as input to the
    neural network.

    Parameters
    ----------
    radial_max_distance : float
        The maximum distance for radial symmetry functions in nanometer.
    radial_min_distance : float
        The minimum distance for radial symmetry functions in nanometer.
    number_of_radial_basis_functions : int
        The number of radial basis functions.
    maximum_interaction_radius_for_angular_features : float
        The maximum interaction radius for angular features  in nanometer.
    minimum_interaction_radius_for_angular_features : float
        The minimum interaction radius for angular features  in nanometer.
    angular_dist_divisions : int
        The number of angular distance divisions.
    angle_sections : int
        The number of angle sections.
    nr_of_supported_elements : int, optional
        The number of supported elements, by default 7.
    """

    def __init__(
        self,
        radial_max_distance: float,
        radial_min_distance: float,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius_for_angular_features: float,
        minimum_interaction_radius_for_angular_features: float,
        angular_dist_divisions: int,
        angle_sections: int,
        nr_of_supported_elements: int = 7,
    ):
        super().__init__()
        from modelforge.potential import CosineAttenuationFunction

        self.maximum_interaction_radius_for_angular_features = (
            maximum_interaction_radius_for_angular_features
        )
        self.nr_of_supported_elements = nr_of_supported_elements

        self.cutoff_module = CosineAttenuationFunction(radial_max_distance)

        # Initialize radial and angular symmetry functions
        self.radial_symmetry_functions = self._setup_radial_symmetry_functions(
            radial_max_distance, radial_min_distance, number_of_radial_basis_functions
        )
        self.angular_symmetry_functions = self._setup_angular_symmetry_functions(
            maximum_interaction_radius_for_angular_features,
            minimum_interaction_radius_for_angular_features,
            angular_dist_divisions,
            angle_sections,
        )
        # Generate indices for species pairs
        self.register_buffer("triu_index", triu_index(self.nr_of_supported_elements))

    @staticmethod
    def _cumsum_from_zero(input_: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative sum from zero, used for sorting indices.
        """
        cumsum = torch.zeros_like(input_)
        torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
        return cumsum

    @staticmethod
    def triple_by_molecule(
        atom_pairs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
                Convert pairwise indices to central-others format for angular feature
                computation. This method rearranges pairwise atomic indices for angular
                symmetry functions.

                NOTE: this function is adopted from torchani library:
                https://github.com/aiqm/torchani/blob/17204c6dccf6210753bc8c0ca4c92278b60719c9/torchani/aev.py
                distributed under the MIT license.

        .
                Parameters
                ----------
                atom_pairs : torch.Tensor
                    A tensor of atom pair indices.

                Returns
                -------
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    Central atom indices, local pair indices, and sign of the pairs.
        """

        # convert representation from pair to central-others
        ai1 = atom_pairs.view(-1)

        # Note, torch.sort doesn't guarantee stable sort by default. This means
        # that the order of rev_indices is not guaranteed when there are "ties"
        # (i.e., identical values in the input tensor). Stable sort is more
        # expensive and ultimately unnecessary, so we will not use it here, but
        # it does mean that vector-wise comparison of the outputs of this
        # function may be inconsistent for the same input, and thus tests must
        # be designed accordingly.

        sorted_ai1, rev_indices = ai1.sort()

        # sort and compute unique key
        uniqued_central_atom_index, counts = torch.unique_consecutive(
            sorted_ai1, return_inverse=False, return_counts=True
        )

        # compute central_atom_index
        pair_sizes = torch.div(counts * (counts - 1), 2, rounding_mode="trunc")
        pair_indices = torch.repeat_interleave(pair_sizes)
        central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)

        # do local combinations within unique key, assuming sorted
        m = counts.max().item() if counts.numel() > 0 else 0
        n = pair_sizes.shape[0]
        intra_pair_indices = (
            torch.tril_indices(m, m, -1, device=ai1.device)
            .unsqueeze(1)
            .expand(-1, n, -1)
        )
        mask = (
            torch.arange(intra_pair_indices.shape[2], device=ai1.device)
            < pair_sizes.unsqueeze(1)
        ).flatten()
        sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
        sorted_local_index12 += ANIRepresentation._cumsum_from_zero(
            counts
        ).index_select(0, pair_indices)

        # unsort result from last part
        local_index12 = rev_indices[sorted_local_index12]

        # compute mapping between representation of central-other to pair
        n = atom_pairs.shape[1]
        sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
        return central_atom_index, local_index12 % n, sign12

    def _setup_radial_symmetry_functions(
        self,
        max_distance: float,
        min_distance: float,
        number_of_radial_basis_functions: int,
    ):
        """
        Initialize the radial symmetry function block.
        Parameters
        ----------
        max_distance : float
        min_distance: float
        number_of_radial_basis_functions : int
        """
        from .representation import AniRadialBasisFunction

        radial_symmetry_function = AniRadialBasisFunction(
            number_of_radial_basis_functions,
            max_distance,
            min_distance,
            dtype=torch.float32,
        )
        return radial_symmetry_function

    def _setup_angular_symmetry_functions(
        self,
        max_distance: float,
        min_distance: float,
        angular_dist_divisions: int,
        angle_sections: int,
    ):
        from .representation import AngularSymmetryFunction

        # set up modelforge angular features
        return AngularSymmetryFunction(
            max_distance,
            min_distance,
            angular_dist_divisions,
            angle_sections,
            dtype=torch.float32,
        )

    def forward(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
        atom_index: torch.Tensor,
    ) -> SpeciesAEV:
        """
        Forward pass to compute Atomic Environment Vectors (AEVs).

        Parameters
        ----------
        data : NNPInput
            The input data for the ANI model.
        pairlist_output : PairlistData
            Pairwise distances and displacement vectors.
        atom_index : torch.Tensor
            Indices of atomic species.

        Returns
        -------
        SpeciesAEV
            The computed atomic environment vectors (AEVs) for each species.
        """

        # ----------------- Radial symmetry vector ---------------- #
        # compute radial aev
        radial_feature_vector = self.radial_symmetry_functions(pairlist_output.d_ij)
        # Apply cutoff to radial features
        rcut_ij = self.cutoff_module(pairlist_output.d_ij)
        radial_feature_vector = radial_feature_vector * rcut_ij

        # Process output to prepare for angular symmetry vector
        postprocessed_radial_aev_and_additional_data = self._postprocess_radial_aev(
            radial_feature_vector,
            data=data,
            atom_index=atom_index,
            pairlist_output=pairlist_output,
        )
        processed_radial_feature_vector = postprocessed_radial_aev_and_additional_data[
            "radial_aev"
        ]

        # ----------------- Angular symmetry vector ---------------- #
        # Compute angular AEV
        angular_data = self._preprocess_angular_aev(
            postprocessed_radial_aev_and_additional_data
        )
        # calculate angular aev
        angular_feature_vector = self.angular_symmetry_functions(
            angular_data["angular_r_ij"]
        )
        # postprocess
        angular_data["angular_feature_vector"] = angular_feature_vector
        processed_angular_feature_vector = self._postprocess_angular_aev(
            data, angular_data
        )
        # Concatenate radial and angular features
        aevs = torch.cat(
            [processed_radial_feature_vector, processed_angular_feature_vector], dim=-1
        )

        return SpeciesAEV(atom_index, aevs)

    def _postprocess_angular_aev(
        self,
        data: NNPInput,
        angular_data: Dict[str, torch.Tensor],
    ):
        """
        Postprocess the angular AEVs.

        Parameters
        ----------
        data : NNPInput
            The input data.
        angular_data : Dict[str, torch.Tensor]
            The angular data including species and displacement vectors.

        Returns
        -------
        torch.Tensor
            The processed angular AEVs.
        """
        angular_sublength = self.angular_symmetry_functions.angular_sublength
        angular_length = (
            (self.nr_of_supported_elements * (self.nr_of_supported_elements + 1))
            // 2
            * angular_sublength
        )

        num_species_pairs = angular_length // angular_sublength

        number_of_atoms = data.atomic_numbers.shape[0]

        # compute angular aev
        central_atom_index = angular_data["central_atom_index"]
        angular_species12 = angular_data["angular_species12"]
        angular_r_ij = angular_data["angular_r_ij"]

        angular_terms_ = angular_data["angular_feature_vector"]
        # Initialize tensor to store angular AEVs

        angular_aev = angular_terms_.new_zeros(
            (number_of_atoms * num_species_pairs, angular_sublength)
        )

        index = (
            central_atom_index * num_species_pairs
            + self.triu_index[angular_species12[0], angular_species12[1]]
        )
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(number_of_atoms, angular_length)
        return angular_aev

    def _postprocess_radial_aev(
        self,
        radial_feature_vector: torch.Tensor,
        data: NNPInput,
        atom_index: torch.Tensor,
        pairlist_output: PairlistData,
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocess the radial AEVs.

        Parameters
        ----------
        radial_feature_vector : torch.Tensor
            The radial feature vectors.
        data : NNPInput
            The input data for the ANI model.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the radial AEVs and additional data.
        """
        radial_feature_vector = radial_feature_vector.squeeze(
            1
        )  # Shape [num_pairs, radial_sublength]
        number_of_atoms = data.atomic_numbers.shape[0]
        radial_sublength = (
            self.radial_symmetry_functions.number_of_radial_basis_functions
        )
        radial_length = radial_sublength * self.nr_of_supported_elements

        # Initialize tensor to store radial AEVs
        radial_aev = radial_feature_vector.new_zeros(
            (
                number_of_atoms * self.nr_of_supported_elements,
                radial_sublength,
            )
        )  # Shape [num_atoms * nr_of_supported_elements, radial_sublength]

        atom_index12 = (
            pairlist_output.pair_indices
        )  # Shape [2, num_pairs] # this is the pair list of the atoms (e.g. C=6)
        species = atom_index
        species12 = species[
            atom_index12
        ]  # Shape [2, num_pairs], this is the pair index but now with optimzied indexing

        # What are we doing here? we generate an atomic environment vector with
        # fixed dimensinos (nr_of_supported_elements, 16 (represents number of
        # radial symmetry functions)) for each **element** per atom (in a pair)

        # this is a magic indexing function that works
        index12 = atom_index12 * self.nr_of_supported_elements + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_feature_vector)
        radial_aev.index_add_(0, index12[1], radial_feature_vector)

        radial_aev = radial_aev.reshape(number_of_atoms, radial_length)

        # compute new neighbors with radial_cutoff
        distances = pairlist_output.d_ij.T.flatten()
        even_closer_indices = (
            (distances <= self.maximum_interaction_radius_for_angular_features)
            .nonzero()
            .flatten()
        )
        r_ij = pairlist_output.r_ij
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        r_ij_small = r_ij.index_select(0, even_closer_indices)

        return {
            "radial_aev": radial_aev,
            "atom_index12": atom_index12,
            "species12": species12,
            "r_ij": r_ij_small,
        }

    def _preprocess_angular_aev(self, data: Dict[str, torch.Tensor]):
        """
        Preprocess the angular AEVs.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The data dictionary containing radial AEVs and additional data.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the preprocessed angular AEV data.
        """
        atom_index12 = data["atom_index12"]
        species12 = data["species12"]
        r_ij = data["r_ij"]

        # compute angular aev
        central_atom_index, pair_index12, sign12 = self.triple_by_molecule(atom_index12)
        species12_small = species12[:, pair_index12]

        r_ij12 = r_ij.index_select(0, pair_index12.view(-1)).view(
            2, -1, 3
        ) * sign12.unsqueeze(-1)
        species12_ = torch.where(
            torch.eq(sign12, 1), species12_small[1], species12_small[0]
        )
        return {
            "angular_r_ij": r_ij12,
            "central_atom_index": central_atom_index,
            "angular_species12": species12_,
        }


class MultiOutputHeadNetwork(nn.Module):

    def __init__(self, shared_layers: nn.Sequential, output_dims: int):
        """
        A neural network module with multiple output heads for property prediction.

        This network shares a common set of layers and then splits into multiple
        heads, each of which predicts a different output property.

        Parameters
        ----------
        shared_layers : nn.Sequential
            The shared layers before branching into the output heads.
        output_dims : int
            The number of output properties (dimensions) to predict.
        """
        super().__init__()
        self.shared_layers = shared_layers
        # The input dimension is the output dimension of the last shared layer
        input_dim = shared_layers[
            -2
        ].out_features  # Get the output dim from the last shared layer

        # Create a list of output heads, one for each predicted property
        self.output_heads = nn.ModuleList(
            [nn.Linear(input_dim, 1) for _ in range(output_dims)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the multi-output head network.

        The input is processed by the shared layers, and each output head generates
        a prediction for one property.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            A concatenated tensor of predictions from all output heads.
        """
        # Pass the input through the shared layers
        x = self.shared_layers(x)
        # Get the output from each head and concatenate along the last dimension
        outputs = [head(x) for head in self.output_heads]
        return torch.cat(outputs, dim=1)


class ANIInteraction(nn.Module):

    def __init__(
        self,
        *,
        aev_dim: int,
        activation_function: torch.nn.Module,
        predicted_properties: List[str],
        predicted_dim: List[int],
    ):
        """
        Atomic neural network interaction module for ANI.

        This module applies a neural network to the Atomic Environment Vectors
        (AEVs) to compute atomic properties like energy.

        Parameters
        ----------
        aev_dim : int
            The dimensionality of the AEVs.
        activation_function : torch.nn.Module
            The activation function to use in the neural network layers.
        predicted_properties : List[str]
            The names of the properties that the network will predict.
        predicted_dim : List[int]
            The dimensions of each predicted property.
        """
        super().__init__()
        output_dim = int(sum(predicted_dim))
        self.predicted_properties = predicted_properties
        # define atomic neural network
        atomic_neural_networks = self.intialize_atomic_neural_network(
            aev_dim, activation_function, output_dim
        )
        # Initialize atomic neural networks for each element in the supported
        # species
        self.atomic_networks = nn.ModuleList(
            [
                atomic_neural_networks[element]
                for element in ["H", "C", "O", "N", "S", "F", "Cl"]
            ]
        )

    def intialize_atomic_neural_network(
        self,
        aev_dim: int,
        activation_function: torch.nn.Module,
        output_dim: int,
    ) -> Dict[str, nn.Module]:
        """
        Initialize the atomic neural networks for each chemical element.

        Each element gets a separate neural network to predict properties based
        on the AEVs.

        Parameters
        ----------
        aev_dim : int
            The dimensionality of the AEVs.
        activation_function : torch.nn.Module
            The activation function to use.
        output_dim : int
            The output dimensionality for each neural network (sum of all
            predicted properties).

        Returns
        -------
        Dict[str, nn.Module]
            A dictionary mapping element symbols to their corresponding neural
            networks.
        """

        def create_network(layers: List[int]) -> nn.Module:
            """
            Create a sequential neural network with the specified number of
            layers.

            Each layer consists of a linear transformation followed by an
            activation function.

            Parameters
            ----------
            layers : List[int]
                A list where each element is the number of units in the
                corresponding layer.

            Returns
            -------
            nn.Sequential
                A sequential neural network with the specified layers.
            """
            shared_network_layers = []
            input_dim = aev_dim
            for units in layers:
                shared_network_layers.append(nn.Linear(input_dim, units))
                shared_network_layers.append(activation_function)
                input_dim = units

            # Return a MultiOutputHeadNetwork with shared layers and specified
            # output dimensions
            shared_layers = nn.Sequential(*shared_network_layers)
            return MultiOutputHeadNetwork(shared_layers, output_dims=output_dim)

        # Define layer configurations for different elements
        return {
            element: create_network(layers)
            for element, layers in {
                "H": [256, 192, 160],
                "C": [224, 192, 160],
                "N": [192, 160, 128],
                "O": [192, 160, 128],
                "S": [160, 128, 96],
                "F": [160, 128, 96],
                "Cl": [160, 128, 96],
            }.items()
        }

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to compute atomic properties from AEVs.

        For each species, the corresponding atomic neural network is used to
        predict properties.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the species tensor and the AEV tensor.

        Returns
        -------
        torch.Tensor
            The computed atomic properties for each atom.
        """
        species, aev = input
        per_atom_property = torch.zeros(
            (species.shape[0], len(self.predicted_properties)),
            dtype=aev.dtype,
            device=aev.device,
        )

        for i, model in enumerate(self.atomic_networks):
            # create a mask to select the atoms of the current species (i)
            mask = torch.eq(species, i)
            per_element_index = mask.nonzero().flatten()
            # if the species is present in the batch, run it through the network
            if per_element_index.shape[0] > 0:
                input_ = aev.index_select(0, per_element_index)
                per_element_predction = model(input_)
                per_atom_property[per_element_index, :] = per_element_predction

        return per_atom_property


from typing import List


class ANI2xCore(torch.nn.Module):

    def __init__(
        self,
        *,
        maximum_interaction_radius: float,
        minimum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        maximum_interaction_radius_for_angular_features: float,
        minimum_interaction_radius_for_angular_features: float,
        activation_function_parameter: Dict[str, str],
        angular_dist_divisions: int,
        predicted_properties: List[str],
        predicted_dim: List[int],
        angle_sections: int,
        potential_seed: int = -1,
    ) -> None:
        """
        The main core module for the ANI2x architecture.

        ANI2x computes atomic properties (like energy) based on Atomic Environment Vectors (AEVs),
        with support for multiple atomic species.

        Parameters
        ----------
        maximum_interaction_radius : float
            The maximum interaction radius for radial symmetry functions.
        minimum_interaction_radius : float
            The minimum interaction radius for radial symmetry functions.
        number_of_radial_basis_functions : int
            The number of radial basis functions.
        maximum_interaction_radius_for_angular_features : float
            The maximum interaction radius for angular symmetry functions.
        minimum_interaction_radius_for_angular_features : float
            The minimum interaction radius for angular symmetry functions.
        activation_function_parameter : Dict[str, str]
            A dictionary specifying the activation function to use.
        angular_dist_divisions : int
            The number of angular distance divisions.
        predicted_properties : List[str]
            A list of property names that the model will predict.
        predicted_dim : List[int]
            A list of dimensions for each predicted property.
        angle_sections : int
            The number of angular sections for the angular symmetry functions.
        potential_seed : int, optional
            A seed for random number generation, by default -1.
        """

        from modelforge.utils.misc import seed_random_number

        if potential_seed != -1:
            seed_random_number(potential_seed)

        super().__init__()

        self.num_species = 7  # Number of elements supported by ANI2x
        self.predicted_dim = predicted_dim

        self.activation_function = activation_function_parameter["activation_function"]

        log.debug("Initializing the ANI2x architecture.")
        self.predicted_properties = predicted_properties

        # Initialize the representation block (AEVs)
        self.ani_representation_module = ANIRepresentation(
            maximum_interaction_radius,
            minimum_interaction_radius,
            number_of_radial_basis_functions,
            maximum_interaction_radius_for_angular_features,
            minimum_interaction_radius_for_angular_features,
            angular_dist_divisions,
            angle_sections,
        )
        # Calculate the dimensions of the radial and angular AEVs
        radial_length = self.num_species * number_of_radial_basis_functions
        angular_length = (
            (self.num_species * (self.num_species + 1))
            // 2
            * self.ani_representation_module.angular_symmetry_functions.angular_sublength
        )
        aev_length = radial_length + angular_length

        # Initialize interaction modules for predicting properties from AEVs
        self.interaction_modules = ANIInteraction(
            aev_dim=aev_length,
            activation_function=self.activation_function,
            predicted_properties=predicted_properties,
            predicted_dim=predicted_dim,
        )

        # ----- ATOMIC NUMBER LOOKUP --------
        # Create a tensor for direct lookup. The size of this tensor will be
        # # the max atomic number in map. Initialize with a default value (e.g., -1 for not found).

        maximum_atomic_number = max(ATOMIC_NUMBER_TO_INDEX_MAP.keys())
        lookup_tensor = torch.full((maximum_atomic_number + 1,), -1, dtype=torch.long)

        # Populate the lookup tensor with indices from your map
        for atomic_number, index in ATOMIC_NUMBER_TO_INDEX_MAP.items():
            lookup_tensor[atomic_number] = index

        self.register_buffer("lookup_tensor", lookup_tensor)

    def compute_properties(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
        atom_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute atomic properties (like energy) from AEVs.

        This is the main computation method, which processes the input data and
        pairlist to generate per-atom predictions.

        Parameters
        ----------
        data : NNPInput
            The input data for the ANI model, including atomic numbers and positions.
        pairlist_output : PairlistData
            The pairwise distances and displacement vectors between atoms.
        atom_index : torch.Tensor
            The indices of atomic species in the input data.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and the scalar representation of AEVs.
        """

        # Compute AEVs (atomic environment vectors)
        representation = self.ani_representation_module(
            data, pairlist_output, atom_index
        )
        # Use interaction modules to predict properties from AEVs
        predictions = self.interaction_modules(representation)

        # generate the output results
        return {
            "per_atom_prediction": predictions,
            "per_atom_scalar_representation": representation.aevs,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }

    def _aggregate_results(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate per-atom predictions into property-specific tensors.

        This method splits the concatenated per-atom predictions into individual properties.

        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            A dictionary containing per-atom predictions.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the split predictions for each property.
        """
        # retrieve the per-atom predictions (nr_atoms, nr_properties)
        per_atom_prediction = outputs.pop("per_atom_prediction")
        # split the predictions into individual properties
        split_tensors = torch.split(per_atom_prediction, self.predicted_dim, dim=1)
        # update the outputs with the split predictions
        outputs.update(
            {
                label: tensor.squeeze(1)
                for label, tensor in zip(self.predicted_properties, split_tensors)
            }
        )
        return outputs

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ANI2x model to compute atomic properties.

        This method combines the AEV computation with the property prediction
        step.

        Parameters
        ----------
        data : NNPInput
            The input data for the model, including atomic numbers and
            positions.
        pairlist_output : PairlistData
            The pairwise distance and displacement vectors between atoms.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of calculated properties, including per-atom
            predictions and AEVs.
        """
        atom_index = self.lookup_tensor[data.atomic_numbers.long()]
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(data, pairlist_output, atom_index)
        # add atomic numbers to the output
        outputs["atomic_numbers"] = data.atomic_numbers
        # extract predictions per property
        return self._aggregate_results(outputs)

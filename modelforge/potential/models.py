"""
This module contains the base classes for the neural network potentials.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

import lightning as pl
import torch
import wandb
from loguru import logger as log
from openff.units import unit
from torch.nn import Module

from modelforge.dataset.dataset import NNPInput, NNPInputTuple

from modelforge.dataset.dataset import DatasetParameters
from modelforge.potential.parameters import (
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
    AimNet2Parameters,
)
from modelforge.train.parameters import RuntimeParameters, TrainingParameters
from typing import TypeVar, Union

# Define a TypeVar that can be one of the parameter models
T_NNP_Parameters = TypeVar(
    "T_NNP_Parameters",
    ANI2xParameters,
    SAKEParameters,
    SchNetParameters,
    PhysNetParameters,
    PaiNNParameters,
    TensorNetParameters,
    AimNet2Parameters,
)


class PairlistData(NamedTuple):
    """
    A namedtuple to store the outputs of the Pairlist and Neighborlist forward methods.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A tensor of shape (2, n_pairs) containing the indices of the interacting atom pairs.
    d_ij : torch.Tensor
        A tensor of shape (n_pairs, 1) containing the Euclidean distances between the atoms in each pair.
    r_ij : torch.Tensor
        A tensor of shape (n_pairs, 3) containing the displacement vectors between the atoms in each pair.
    """

    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class Pairlist(Module):

    def __init__(self, only_unique_pairs: bool = False):
        """
         Handle pair list calculations for systems, returning indices, distances
         and distance vectors for atom pairs within a certain cutoff.

        Parameters
        ----------
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """
        super().__init__()
        self.only_unique_pairs = only_unique_pairs

    def enumerate_all_pairs(self, atomic_subsystem_indices: torch.Tensor):
        """
        Compute all pairs of atoms and their distances.

        Parameters
        ----------
        atomic_subsystem_indices : torch.Tensor
            Atom indices to indicate which atoms belong to which molecule.
            Note in all cases, the values in this tensor must be numbered from 0 to n_molecules - 1 sequentially, with no gaps in the numbering. E.g., [0,0,0,1,1,2,2,2 ...].
            This is the case for all internal data structures, and thus no validation is performed in this routine. If the data is not structured in this way, the results will be incorrect.

        Returns
        -------
        torch.Tensor
            Pair indices for all atom pairs.
        """

        # get device that passed tensors lives on, initialize on the same device
        device = atomic_subsystem_indices.device

        # if there is only one molecule, we do not need to use additional looping and offsets
        if torch.sum(atomic_subsystem_indices) == 0:
            n = len(atomic_subsystem_indices)
            if self.only_unique_pairs:
                i_final_pairs, j_final_pairs = torch.triu_indices(
                    n, n, 1, device=device
                )
            else:
                # Repeat each number n-1 times for i_indices
                i_final_pairs = torch.repeat_interleave(
                    torch.arange(n, device=device),
                    repeats=n - 1,
                )

                # Correctly construct j_indices
                j_final_pairs = torch.cat(
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

        else:
            # if we have more than one molecule, we will take into account molecule size and offsets when
            # calculating pairs, as using the approach above is not memory efficient for datasets with large molecules
            # and/or larger batch sizes; while not likely a problem on higher end GPUs with large amounts of memory
            # cheaper commodity and mobile GPUs may have issues

            # atomic_subsystem_indices are always numbered from 0 to n_molecules
            # - 1 e.g., a single molecule will be [0, 0, 0, 0 ... ] and a batch
            # of molecules will always start at 0 and increment [ 0, 0, 0, 1, 1,
            # 1, ...] As such, we can use bincount, as there are no gaps in the
            # numbering

            # Note if the indices are not numbered from 0 to n_molecules - 1, this will not work
            # E.g., bincount on [3,3,3, 4,4,4, 5,5,5] will return [0,0,0,3,3,3,3,3,3]
            # as we have no values for 0, 1, 2
            # using a combination of unique and argsort would make this work for any numbering ordering
            # but that is not how the data ends up being structured internally, and thus is not needed
            repeats = torch.bincount(atomic_subsystem_indices)
            offsets = torch.cat(
                (torch.tensor([0], device=device), torch.cumsum(repeats, dim=0)[:-1])
            )

            i_indices = torch.cat(
                [
                    torch.repeat_interleave(
                        torch.arange(o, o + r, device=device), repeats=r
                    )
                    for r, o in zip(repeats, offsets)
                ]
            )
            j_indices = torch.cat(
                [
                    torch.cat([torch.arange(o, o + r, device=device) for _ in range(r)])
                    for r, o in zip(repeats, offsets)
                ]
            )

            if self.only_unique_pairs:
                # filter out pairs that are not unique
                unique_pairs_mask = i_indices < j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]
            else:
                # filter out identical values
                unique_pairs_mask = i_indices != j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]

        # concatenate to form final (2, n_pairs) tensor
        pair_indices = torch.stack((i_final_pairs, j_final_pairs))

        return pair_indices.to(device)

    def construct_initial_pairlist_using_numpy(
        self, atomic_subsystem_indices: torch.Tensor
    ):
        """Compute all pairs of atoms and also return counts of the number of pairs for each molecule in batch.

        Parameters
        ----------
        atomic_subsystem_indices : torch.Tensor
            Atom indices to indicate which atoms belong to which molecule.

        Returns
        -------
        pair_indices : np.ndarray, shape (2, n_pairs)
            Pairs of atom indices, 0-indexed for each molecule
        number_of_pairs : np.ndarray, shape (n_molecules)
            The number to index into pair_indices for each molecule

        """

        # atomic_subsystem_indices are always numbered from 0 to n_molecules - 1
        # e.g., a single molecule will be [0, 0, 0, 0 ... ]
        # and a batch of molecules will always start at 0 and increment [ 0, 0, 0, 1, 1, 1, ...]
        # As such, we can use bincount, as there are no gaps in the numbering
        # Note if the indices are not numbered from 0 to n_molecules - 1, this will not work
        # E.g., bincount on [3,3,3, 4,4,4, 5,5,5] will return [0,0,0,3,3,3,3,3,3]
        # as we have no values for 0, 1, 2
        # using a combination of unique and argsort would make this work for any numbering ordering
        # but that is not how the data ends up being structured internally, and thus is not needed

        import numpy as np

        # get the number of atoms in each molecule
        repeats = np.bincount(atomic_subsystem_indices)

        # calculate the number of pairs for each molecule, using simple permutation
        npairs_by_molecule = np.array([r * (r - 1) for r in repeats], dtype=np.int16)

        i_indices = np.concatenate(
            [
                np.repeat(
                    np.arange(
                        0,
                        r,
                        dtype=np.int16,
                    ),
                    repeats=r,
                )
                for r in repeats
            ]
        )
        j_indices = np.concatenate(
            [
                np.concatenate([np.arange(0, 0 + r, dtype=np.int16) for _ in range(r)])
                for r in repeats
            ]
        )

        # filter out identical pairs where i==j
        unique_pairs_mask = i_indices != j_indices
        i_final_pairs = i_indices[unique_pairs_mask]
        j_final_pairs = j_indices[unique_pairs_mask]

        # concatenate to form final (2, n_pairs) vector
        pair_indices = np.stack((i_final_pairs, j_final_pairs))

        return pair_indices, npairs_by_molecule

    def calculate_r_ij(
        self, pair_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute displacement vectors between atom pairs.

        Parameters
        ----------
        pair_indices : torch.Tensor
            Atom indices for pairs of atoms. Shape: [2, n_pairs].
        positions : torch.Tensor
            Atom positions. Shape: [atoms, 3].

        Returns
        -------
        torch.Tensor
            Displacement vectors between atom pairs. Shape: [n_pairs, 3].
        """
        # Select the pairs of atom coordinates from the positions
        selected_positions = positions.index_select(0, pair_indices.view(-1)).view(
            2, -1, 3
        )
        return selected_positions[1] - selected_positions[0]

    def calculate_d_ij(self, r_ij: torch.Tensor) -> torch.Tensor:
        """
        ompute Euclidean distances between atoms in each pair.

        Parameters
        ----------
        r_ij : torch.Tensor
            Displacement vectors between atoms in a pair. Shape: [n_pairs, 3].

        Returns
        -------
        torch.Tensor
            Euclidean distances. Shape: [n_pairs, 1].
        """
        return r_ij.norm(dim=1).unsqueeze(1)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
    ) -> PairlistData:
        """
        Performs the forward pass of the Pairlist module.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_atoms, 3].
        atomic_subsystem_indices (torch.Tensor, shape (nr_atoms_per_systems)):
            Atom indices to indicate which atoms belong to which molecule.

        Returns
        -------
        PairListOutputs: A dataclass containing the following attributes:
            pair_indices (torch.Tensor): A tensor of shape (2, n_pairs) containing the indices of the interacting atom pairs.
            d_ij (torch.Tensor): A tensor of shape (n_pairs, 1) containing the Euclidean distances between the atoms in each pair.
            r_ij (torch.Tensor): A tensor of shape (n_pairs, 3) containing the displacement vectors between the atoms in each pair.
        """
        pair_indices = self.enumerate_all_pairs(
            atomic_subsystem_indices,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)
        return PairlistData(
            pair_indices=pair_indices,
            d_ij=self.calculate_d_ij(r_ij),
            r_ij=r_ij,
        )


class Neighborlist(Pairlist):
    def __init__(self, cutoff: float, only_unique_pairs: bool = False):
        """
        Manage neighbor list calculations with a specified cutoff distance.

        Extends the Pairlist class to compute neighbor lists based on a distance cutoff.

        Parameters
        ----------
        cutoff : float
            Cutoff distance for neighbor calculations.
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """
        super().__init__(only_unique_pairs=only_unique_pairs)

        self.register_buffer("cutoff", torch.tensor(cutoff))

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> PairlistData:
        """
        Compute the neighbor list considering a cutoff distance.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_systems, nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices identifying atoms in subsystems. Shape: [nr_atoms].
        pair_indices : Optional[torch.Tensor]
            Precomputed pair indices. If None, will compute pair indices.

        Returns
        -------
        PairListOutputs
            A dataclass containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """

        if pair_indices is None:
            pair_indices = self.enumerate_all_pairs(
                atomic_subsystem_indices,
            )

        r_ij = self.calculate_r_ij(pair_indices, positions)
        d_ij = self.calculate_d_ij(r_ij)

        in_cutoff = (d_ij <= self.cutoff).squeeze()
        # Get the atom indices within the cutoff
        pair_indices_within_cutoff = pair_indices[:, in_cutoff]

        return PairlistData(
            pair_indices=pair_indices_within_cutoff,
            d_ij=d_ij[in_cutoff],
            r_ij=r_ij[in_cutoff],
        )


from typing import Callable, Literal, Optional, Union

import numpy as np


class JAXModel:
    """
    A wrapper for calling a JAX function with predefined parameters and buffers.

    Attributes
    ----------
    jax_fn : Callable
        The JAX function to be called.
    parameter : np.ndarray
        Parameters required by the JAX function.
    buffer : Any
        Buffers required by the JAX function.
    name : str
        Name of the model.
    """

    def __init__(
        self, jax_fn: Callable, parameter: np.ndarray, buffer: np.ndarray, name: str
    ):
        self.jax_fn = jax_fn
        self.parameter = parameter
        self.buffer = buffer
        self.name = name

    def __call__(self, data: NamedTuple):
        """Calls the JAX function using the stored parameters and buffers along with additional data.

        Parameters
        ----------
        data : NamedTuple
            Data to be passed to the JAX function.

        Returns
        -------
        Any
            The result of the JAX function.
        """

        return self.jax_fn(self.parameter, self.buffer, data)

    def __repr__(self):
        return f"{self.__class__.__name__} wrapping {self.name}"


class ComputeInteractingAtomPairs(torch.nn.Module):

    def __init__(self, cutoff: float, only_unique_pairs: bool = False):
        """
        Initialize the ComputeInteractingAtomPairs module.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """

        super().__init__()
        from .models import Neighborlist

        self.only_unique_pairs = only_unique_pairs
        self.calculate_distances_and_pairlist = Neighborlist(cutoff, only_unique_pairs)

    def forward(self, data: Union[NNPInput, NamedTuple]) -> PairlistData:
        """
        Compute the pair list, distances, and displacement vectors for the given
        input data.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            Input data containing atomic numbers, positions, and subsystem
            indices.

        Returns
        -------
        PairlistData
            A namedtuple containing the pair indices, distances, and
            displacement vectors.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices
        # calculate pairlist if none is provided
        if data.pair_list is None:
            pairlist_output = self.calculate_distances_and_pairlist(
                positions=positions,
                atomic_subsystem_indices=atomic_subsystem_indices,
                pair_indices=None,
            )
            pair_list = data.pair_list
        else:
            # pairlist is provided, remove redundant pairs if requested
            if self.only_unique_pairs:
                i_indices = data.pair_list[0]
                j_indices = data.pair_list[1]
                unique_pairs_mask = i_indices < j_indices
                i_final_pairs = i_indices[unique_pairs_mask]
                j_final_pairs = j_indices[unique_pairs_mask]
                pair_list = torch.stack((i_final_pairs, j_final_pairs))
            else:
                pair_list = data.pair_list
            # only calculate d_ij and r_ij
            pairlist_output = self.calculate_distances_and_pairlist(
                positions=positions,
                atomic_subsystem_indices=atomic_subsystem_indices,
                pair_indices=pair_list.to(torch.int64),
            )

        return pairlist_output


from torch.nn import ModuleDict

from modelforge.potential.processing import PerAtomEnergy, CoulombPotential


class PostProcessing(torch.nn.Module):

    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_atom_charge",
        "general_postprocessing_operation",
    ]
    _SUPPORTED_OPERATIONS = [
        "normalize",
        "from_atom_to_molecule_reduction",
        "long_range_electrostatics" "conserve_integer_charge",
    ]

    def __init__(
        self,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Dict[str, Dict[str, float]],
    ):
        """
        Handle post-processing operations on model outputs, such as
        normalization and reduction.

        Parameters
        ----------
        postprocessing_parameter : Dict[str, Dict[str, bool]]
            A dictionary containing the postprocessing parameters for each
            property.
        dataset_statistic : Dict[str, Dict[str, float]]
            A dictionary containing the dataset statistics for normalization and
            other calculations.
        """
        super().__init__()

        self._registered_properties: List[str] = []
        self.registered_chained_operations = ModuleDict()
        self.dataset_statistic = dataset_statistic
        properties_to_process = postprocessing_parameter["properties_to_process"]

        if "per_atom_energy" in properties_to_process:
            self.registered_chained_operations["per_atom_energy"] = PerAtomEnergy(
                postprocessing_parameter["per_atom_energy"],
                dataset_statistic["training_dataset_statistics"],
            )
            self._registered_properties.append("per_atom_energy")

        if "coulomb_potential" in properties_to_process:
            self.registered_chained_operations["coulomb_potential"] = CoulombPotential(
                postprocessing_parameter["coulomb_potential"]["electrostatic_strategy"],
                postprocessing_parameter["coulomb_potential"][
                    "maximum_interaction_radius"
                ],
            )

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform post-processing for all registered properties.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The model output data to be post-processed.

        Returns
        -------
        Dict[str, torch.Tensor]
            The post-processed data.
        """
        processed_data: Dict[str, torch.Tensor] = {}
        # Iterate over items in ModuleDict
        for name, module in self.registered_chained_operations.items():

            module_output = module.forward(data)
            processed_data.update(module_output)

        return processed_data


class Potential(torch.nn.Module):
    def __init__(
        self,
        core_network,
        neighborlist,
        postprocessing,
        jit: bool = False,
        jit_neighborlist: bool = True,
    ):
        """
        Neural network potential model composed of a core network, neighborlist,
        and post-processing.

        Parameters
        ----------
        core_network : torch.nn.Module
            The core neural network used for potential energy calculation.
        neighborlist : torch.nn.Module
            Module for computing neighbor lists and pairwise distances.
        postprocessing : torch.nn.Module
            Module for handling post-processing operations.
        jit : bool, optional
            Whether to JIT compile the core network and post-processing
            (default: False).
        jit_neighborlist : bool, optional
            Whether to JIT compile the neighborlist (default: True).
        """

        super().__init__()
        self.core_network = torch.jit.script(core_network) if jit else core_network
        self.neighborlist = (
            torch.jit.script(neighborlist) if jit_neighborlist else neighborlist
        )
        self.postprocessing = (
            torch.jit.script(postprocessing) if jit else postprocessing
        )

    def forward(self, input_data: NNPInputTuple) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the potential model, computing energy and forces.

        Parameters
        ----------
        input_data : NNPInputTuple
            Input data containing atomic positions and other features.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the processed output data.
        """
        # Step 1: Compute pair list and distances using Neighborlist
        pairlist_output = self.neighborlist.forward(input_data)

        # Step 2: Compute the core network output
        core_output = self.core_network.forward(input_data, pairlist_output)

        # Step 3: Apply postprocessing using PostProcessing
        processed_output = self.postprocessing.forward(core_output)

        return processed_output

    def compute_core_network_output(
        self, input_data: NNPInputTuple
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the core network output, including energy predictions.

        Parameters
        ----------
        input_data : NNPInputTuple
            Input data containing atomic positions and other features.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tensor containing the predicted core network output.
        """
        # Step 1: Compute pair list and distances using Neighborlist
        pairlist_output = self.neighborlist.forward(input_data)

        # Step 2: Compute the core network output
        return self.core_network.forward(input_data, pairlist_output)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        use: Literal["training", "inference"] = "inference"
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

        # remove key neighborlist.calculate_distances_and_pairlist.cutoff
        # if present in the state_dict and replace it with 'neighborlist.cutoff'
        if (
            # only modify state_dict key in inference mode
            "neighborlist.calculate_distances_and_pairlist.cutoff"
            in filtered_state_dict and use == "inference"
        ):
            filtered_state_dict["neighborlist.cutoff"] = filtered_state_dict.pop(
                "neighborlist.calculate_distances_and_pairlist.cutoff"
            )
        elif use == "training":
            pass
        else:
            raise KeyError(
                "load_stat_dict() is only available for training and inference."
            )

        super().load_state_dict(filtered_state_dict, strict=strict, assign=assign)
        self.eval()  # Set the model to evaluation mode


def setup_potential(
    potential_parameter: T_NNP_Parameters,
    dataset_statistic: Dict[str, Dict[str, unit.Quantity]] = {
        "training_dataset_statistics": {
            "per_atom_energy_mean": unit.Quantity(0.0, unit.kilojoule_per_mole),
            "per_atom_energy_stddev": unit.Quantity(1.0, unit.kilojoule_per_mole),
        }
    },
    use_training_mode_neighborlist: bool = False,
    potential_seed: Optional[int] = None,
    jit: bool = True,
    only_unique_pairs: bool = False,
    neighborlist_strategy: Optional[str] = None,
    verlet_neighborlist_skin: Optional[float] = 0.08,
) -> Potential:
    from modelforge.potential import _Implemented_NNPs
    from modelforge.potential.utils import remove_units_from_dataset_statistics
    from modelforge.utils.misc import seed_random_number

    if potential_seed is not None:
        log.info(f"Setting random seed to: {potential_seed}")
        seed_random_number(potential_seed)

    model_type = potential_parameter.potential_name
    core_network = _Implemented_NNPs.get_neural_network_class(model_type)(
        **potential_parameter.core_parameter.model_dump()
    )

    postprocessing = PostProcessing(
        postprocessing_parameter=potential_parameter.postprocessing_parameter.model_dump(),
        dataset_statistic=remove_units_from_dataset_statistics(dataset_statistic),
    )
    if use_training_mode_neighborlist:
        neighborlist = ComputeInteractingAtomPairs(
            cutoff=potential_parameter.core_parameter.maximum_interaction_radius,
            only_unique_pairs=only_unique_pairs,
        )
    else:
        from modelforge.potential.neighbors import OrthogonalDisplacementFunction

        displacement_function = OrthogonalDisplacementFunction()

        if neighborlist_strategy == "verlet":
            from modelforge.potential.neighbors import NeighborlistVerletNsq

            neighborlist = NeighborlistVerletNsq(
                cutoff=potential_parameter.core_parameter.maximum_interaction_radius,
                displacement_function=displacement_function,
                only_unique_pairs=only_unique_pairs,
                skin=verlet_neighborlist_skin,
            )
        elif neighborlist_strategy == "brute":
            from modelforge.potential.neighbors import NeighborlistBruteForce

            neighborlist = NeighborlistBruteForce(
                cutoff=potential_parameter.core_parameter.maximum_interaction_radius,
                displacement_function=displacement_function,
                only_unique_pairs=only_unique_pairs,
            )
        else:
            raise ValueError(
                f"Unsupported neighborlist strategy: {neighborlist_strategy}"
            )

    model = Potential(
        core_network,
        neighborlist,
        postprocessing,
        jit=jit,
        jit_neighborlist=False if use_training_mode_neighborlist else True,
    )
    return model


from openff.units import unit


class NeuralNetworkPotentialFactory:

    @staticmethod
    def generate_potential(
        *,
        use: Literal["training", "inference"],
        potential_parameter: T_NNP_Parameters,
        runtime_parameter: Optional[RuntimeParameters] = None,
        training_parameter: Optional[TrainingParameters] = None,
        dataset_parameter: Optional[DatasetParameters] = None,
        dataset_statistic: Dict[str, Dict[str, float]] = {
            "training_dataset_statistics": {
                "per_atom_energy_mean": unit.Quantity(0.0, unit.kilojoule_per_mole),
                "per_atom_energy_stddev": unit.Quantity(1.0, unit.kilojoule_per_mole),
            }
        },
        potential_seed: Optional[int] = None,
        use_default_dataset_statistic: bool = False,
        use_training_mode_neighborlist: bool = False,
        simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
        only_unique_pairs: bool = False,
        jit: bool = True,
        inference_neighborlist_strategy: str = "verlet",
        verlet_neighborlist_skin: Optional[float] = 0.1,
    ) -> Union[Potential, JAXModel, pl.LightningModule]:
        """
        Create an instance of a neural network potential for training or
        inference.

        Parameters
        ----------
        use : Literal["training", "inference"]
            Whether the potential is for training or inference.
        potential_parameter : T_NNP_Parameters]
            Parameters specific to the neural network potential.
        runtime_parameter : Optional[RuntimeParameters], optional
            Parameters for configuring the runtime environment (default is
            None).
        training_parameter : Optional[TrainingParameters], optional
            Parameters for configuring training (default is None).
        dataset_parameter : Optional[DatasetParameters], optional
            Parameters for configuring the dataset (default is None).
        dataset_statistic : Dict[str, Dict[str, float]], optional
            Dataset statistics for normalization (default is provided).
        potential_seed : Optional[int], optional
            Seed for random number generation (default is None).
        use_default_dataset_statistic : bool, optional
            Whether to use default dataset statistics (default is False).
        use_training_mode_neighborlist : bool, optional
            Whether to use neighborlist during training mode (default is False).
        simulation_environment : Literal["PyTorch", "JAX"], optional
            Specify whether to use PyTorch or JAX as the simulation environment
            (default is "PyTorch").
        only_unique_pairs : bool, optional
            Whether to use only unique pairs of atoms (default is False).
        jit : bool, optional
            Whether to use JIT compilation (default is True).
        inference_neighborlist_strategy : Optional[str], optional
            Neighborlist strategy for inference (default is "verlet"). other option is "brute".
        verlet_neighborlist_skin : Optional[float], optional
            Skin for the Verlet neighborlist (default is 0.1, units nanometers).
        Returns
        -------
        Union[Potential, JAXModel, pl.LightningModule]
            An instantiated neural network potential for training or inference.
        """

        from modelforge.train.training import ModelTrainer

        log.debug(f"{training_parameter=}")
        log.debug(f"{potential_parameter=}")
        log.debug(f"{dataset_parameter=}")

        # obtain model for training
        if use == "training":
            model_trainer = ModelTrainer(
                potential_parameter=potential_parameter,
                training_parameter=training_parameter,
                dataset_parameter=dataset_parameter,
                runtime_parameter=runtime_parameter,
                potential_seed=potential_seed,
                dataset_statistic=dataset_statistic,
                use_default_dataset_statistic=use_default_dataset_statistic,
            )
            
            return model_trainer
        
        if use == "test_loading":
            model_trainer = ModelTrainer(
                potential_parameter=potential_parameter,
                training_parameter=training_parameter,
                dataset_parameter=dataset_parameter,
                runtime_parameter=runtime_parameter,
                potential_seed=potential_seed,
                dataset_statistic=dataset_statistic,
                use_default_dataset_statistic=use_default_dataset_statistic,
                restore_from_wandb={
                    "model_name": "model.pt",
                    "run_path": "modelforge_nnps/test_checkpoint/z9e9id8w",
                },
            )
            return model_trainer
        
        # obtain model for inference
        elif use == "inference":
            model = setup_potential(
                potential_parameter=potential_parameter,
                dataset_statistic=dataset_statistic,
                use_training_mode_neighborlist=use_training_mode_neighborlist,
                potential_seed=potential_seed,
                jit=jit,
                only_unique_pairs=only_unique_pairs,
                neighborlist_strategy=inference_neighborlist_strategy,
                verlet_neighborlist_skin=verlet_neighborlist_skin,
            )
            if simulation_environment == "JAX":
                return PyTorch2JAXConverter().convert_to_jax_model(model)
            else:
                return model
        else:
            raise NotImplementedError(f"Unsupported 'use' value: {use}")
    
    @staticmethod
    def load_potential(
        restore_from_wandb: Dict[str, str],
    ):
        restored_model = wandb.restore(
            restore_from_wandb["model_name"],
            run_path=restore_from_wandb["run_path"],
        )
        model = torch.load(restored_model.name)
        
        return model


class PyTorch2JAXConverter:
    """
    Wraps a PyTorch neural network potential instance in a Flax module using the
    `pytorch2jax` library (https://github.com/subho406/Pytorch2Jax).
    The converted model uses dlpack to convert between Pytorch and Jax tensors
    in-memory and executes Pytorch backend inside Jax wrapped functions.
    The wrapped modules are compatible with Jax backward-mode autodiff.
    """

    def convert_to_jax_model(
        self,
        nnp_instance: Potential,
    ) -> JAXModel:
        """
        Convert a PyTorch neural network instance to a JAX model.

        Parameters
        ----------
        nnp_instance :
            The PyTorch neural network instance to be converted.

        Returns
        -------
        JAXModel
            A JAX model containing the converted neural network function, parameters, and buffers.
        """

        jax_fn, params, buffers = self._convert_pytnn_to_jax(nnp_instance)
        return JAXModel(jax_fn, params, buffers, nnp_instance.__class__.__name__)

    @staticmethod
    def _convert_pytnn_to_jax(
        nnp_instance: Potential,
    ) -> Tuple[Callable, np.ndarray, np.ndarray]:
        """Internal method to convert PyTorch neural network parameters and buffers to JAX format.

        Parameters
        ----------
        nnp_instance : Any
            The PyTorch neural network instance.

        Returns
        -------
        Tuple[Callable, Any, Any]
            A tuple containing the JAX function, parameters, and buffers.
        """

        # make sure
        from modelforge.utils.io import import_

        jax = import_("jax")
        # use the wrapper to check if pytorch2jax is in the environment

        custom_vjp = import_("jax").custom_vjp

        # from jax import custom_vjp
        convert_to_jax = import_("pytorch2jax").pytorch2jax.convert_to_jax
        convert_to_pyt = import_("pytorch2jax").pytorch2jax.convert_to_pyt
        # from pytorch2jax.pytorch2jax import convert_to_jax, convert_to_pyt

        import functorch
        from functorch import make_functional_with_buffers

        # Convert the PyTorch model to a functional representation and extract the model function and parameters
        model_fn, model_params, model_buffer = make_functional_with_buffers(
            nnp_instance
        )

        # Convert the model parameters from PyTorch to JAX representations
        model_params = jax.tree_map(convert_to_jax, model_params)
        # Convert the model buffer from PyTorch to JAX representations
        model_buffer = jax.tree_map(convert_to_jax, model_buffer)

        # Define the apply function using a custom VJP
        @custom_vjp
        def apply(params, *args, **kwargs):
            # Convert the input data from JAX to PyTorch
            params, args, kwargs = map(
                lambda x: jax.tree_map(convert_to_pyt, x), (params, args, kwargs)
            )
            # Apply the model function to the input data
            out = model_fn(params, *args, **kwargs)
            # Convert the output data from PyTorch to JAX
            out = jax.tree_map(convert_to_jax, out)
            return out

        # Define the forward and backward passes for the VJP
        def apply_fwd(params, *args, **kwargs):
            return apply(params, *args, **kwargs), (params, args, kwargs)

        def apply_bwd(res, grads):
            params, args, kwargs = res
            params, args, kwargs = map(
                lambda x: jax.tree_map(convert_to_pyt, x), (params, args, kwargs)
            )
            grads = jax.tree_map(convert_to_pyt, grads)
            # Compute the gradients using the model function and convert them
            # from JAX to PyTorch representations
            grads = functorch.vjp(model_fn, params, *args, **kwargs)[1](grads)
            return jax.tree_map(convert_to_jax, grads)

        apply.defvjp(apply_fwd, apply_bwd)

        # Return the apply function and the converted model parameters
        return apply, model_params, model_buffer

"""
This module contains the base classes for the neural network potentials.
"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)

import lightning as pl
import torch
from loguru import logger as log
from openff.units import unit
from torch.nn import Module

from modelforge.dataset.dataset import NNPInput, NNPInputTuple

if TYPE_CHECKING:
    from modelforge.potential.ani import ANI2x, AniNeuralNetworkData
    from modelforge.potential.painn import PaiNN, PaiNNNeuralNetworkData
    from modelforge.potential.physnet import PhysNet, PhysNetNeuralNetworkData
    from modelforge.potential.sake import SAKE, SAKENeuralNetworkInput
    from modelforge.potential.schnet import SchNet, SchnetNeuralNetworkData


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
    mask: Dict[str, torch.Tensor]


class Pairlist(Module):
    """
    Handle pair list calculations for systems, returning indices, distances and distance vectors for atom pairs within a certain cutoff.

    Attributes
    ----------
    only_unique_pairs : bool
        If True, only unique pairs are returned (default is False).
    """

    def __init__(self, only_unique_pairs: bool = False):
        """
        Initialize the Pairlist object.

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

            # atomic_subsystem_indices are always numbered from 0 to n_molecules - 1
            # e.g., a single molecule will be [0, 0, 0, 0 ... ]
            # and a batch of molecules will always start at 0 and increment [ 0, 0, 0, 1, 1, 1, ...]
            # As such, we can use bincount, as there are no gaps in the numbering
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
        """Compute Euclidean distances between atoms in each pair.

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
    """
    Manage neighbor list calculations with a specified cutoff distance(s).

    This class extends Pairlist to consider a cutoff distance for neighbor calculations.
    """

    def __init__(self, cutoffs: Dict[str, float], only_unique_pairs: bool = False):
        """
        Initialize the Neighborlist with a specific cutoff distance.

        Parameters
        ----------
        cutoffs : Dict[str, unit.Quantity]
            Cutoff distances for neighbor calculations.
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
        """
        super().__init__(only_unique_pairs=only_unique_pairs)

        # self.register_buffer("cutoff", torch.tensor(cutoff.to(unit.nanometer).m))
        self.register_buffer("cutoffs", torch.tensor([c for c in cutoffs.values()]))
        self.labels = list(cutoffs.keys())

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> PairlistData:
        """
        Forward pass to compute neighbor list considering a cutoff distance.

        Overrides the `forward` method from Pairlist to include cutoff distance in calculations.

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

        interacting_outputs = {}
        for cutoff, label in zip(self.cutoffs, self.labels):
            # Find pairs within the cutoff
            in_cutoff = (d_ij <= cutoff).squeeze()
            # Get the atom indices within the cutoff
            pair_indices_within_cutoff = pair_indices[:, in_cutoff]

            interacting_outputs[label] = PairlistData(
                pair_indices=pair_indices_within_cutoff,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        return interacting_outputs


from typing import Callable, Literal, Optional, Union

import numpy as np


class JAXModel:
    """
    A model wrapper that facilitates calling a JAX function with predefined parameters and buffers.

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


class PyTorch2JAXConverter:
    """
    Wraps a PyTorch neural network potential instance in a Flax module using the
    `pytorch2jax` library (https://github.com/subho406/Pytorch2Jax).
    The converted model uses dlpack to convert between Pytorch and Jax tensors
    in-memory and executes Pytorch backend inside Jax wrapped functions.
    The wrapped modules are compatible with Jax backward-mode autodiff.

    Parameters
    ----------
    nnp_instance : Any
        The neural network potential instance to convert.

    Returns
    -------
    JAXModel
        The converted JAX model.
    """

    def convert_to_jax_model(
        self, nnp_instance: Union["ANI2x", "SchNet", "PaiNN", "PhysNet"]
    ) -> JAXModel:
        """
        Convert a PyTorch neural network instance to a JAX model.

        Parameters
        ----------
        nnp_instance : Union["ANI2x", "SchNet", "PaiNN", "PhysNet"]
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
        nnp_instance: Union["ANI2x", "SchNet", "PaiNN", "PhysNet"]
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
            # Compute the gradients using the model function and convert them from JAX to PyTorch representations
            grads = functorch.vjp(model_fn, params, *args, **kwargs)[1](grads)
            return jax.tree_map(convert_to_jax, grads)

        apply.defvjp(apply_fwd, apply_bwd)

        # Return the apply function and the converted model parameters
        return apply, model_params, model_buffer


from modelforge.dataset.dataset import DatasetParameters
from modelforge.potential.parameters import (
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
)
from modelforge.train.parameters import RuntimeParameters, TrainingParameters


class NeuralNetworkPotentialFactory:
    """
    Factory class for creating instances of neural network potentials for training/inference.
    """

    @staticmethod
    def generate_potential(
        *,
        use: Literal["training", "inference"],
        potential_parameter: Union[
            ANI2xParameters,
            SAKEParameters,
            SchNetParameters,
            PhysNetParameters,
            PaiNNParameters,
            TensorNetParameters,
        ],
        runtime_parameter: Optional[RuntimeParameters] = None,
        training_parameter: Optional[TrainingParameters] = None,
        dataset_parameter: Optional[DatasetParameters] = None,
        dataset_statistic: Dict[str, Dict[str, float]] = {
            "training_dataset_statistics": {
                "per_atom_energy_mean": 0.0,
                "per_atom_energy_stddev": 1.0,
            }
        },
        potential_seed: int = -1,
        simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
    ) -> Union[torch.nn.Module, JAXModel, pl.LightningModule]:
        """
        Creates an NNP instance of the specified type, configured either for training or inference.

        Parameters
        ----------
        use : Literal["training", "inference"]
            The use case for the model instance, either 'training' or 'inference'.
        potential_parameter : Union[ANI2xParameters, SAKEParameters, SchNetParameters, PhysNetParameters, PaiNNParameters, TensorNetParameters]
            Parameters specific to the potential.
        runtime_parameter : Optional[RuntimeParameters], optional
            Parameters for configuring the runtime environment.
        training_parameter : Optional[TrainingParameters], optional
            Parameters for configuring the training.
        dataset_parameter : Optional[DatasetParameters], optional
            Parameters for configuring the dataset.
        dataset_statistic : Optional[Dict[str, float]], optional
            Statistics of the dataset for normalization purposes.
        potential_seed : Optional[int], optional
            Seed for the random number generator.
        simulation_environment : Optional[Literal["PyTorch", "JAX"]], optional, None
            The simulation environment to use for training/inference. Will override the runtime parameter if provided.


        Returns
        -------
        Union[Type[torch.nn.Module], Type[JAXModel], Type[pl.LightningModule]]
            An instantiated model.

        Raises
        ------
        ValueError
            If an unknown use case is requested.
        NotImplementedError
            If the requested model type is not implemented.
        """

        from modelforge.potential import _Implemented_NNP_Cores
        from modelforge.potential.models import Potential
        from modelforge.train.training import ModelTrainer

        log.debug(f"{training_parameter=}")
        log.debug(f"{potential_parameter=}")
        log.debug(f"{dataset_parameter=}")

        # obtain model for training
        if use == "training":
            model = ModelTrainer(
                potential_parameter=potential_parameter,
                training_parameter=training_parameter,
                dataset_parameter=dataset_parameter,
                runtime_parameter=runtime_parameter,
                potential_seed=potential_seed,
            )
            return model
        # obtain model for inference
        elif use == "inference":

            postprocessing = PostProcessing(
                postprocessing_parameter=potential_parameter.postprocessing_parameter.model_dump(),
                dataset_statistic=dataset_statistic,
            )

            neighborlist = NeighborlistForInferenceNonUniquePairs(
                cutoffs={
                    "maximum_interaction_radius": potential_parameter.core_parameter.maximum_interaction_radius,
                    # Add more cutoffs if needed
                },
            )

            model_type = potential_parameter.potential_name
            core_network = _Implemented_NNP_Cores.get_neural_network_class(model_type)(
                **potential_parameter.core_parameter.model_dump()
            )

            model = Potential(core_network, neighborlist, postprocessing)
            if simulation_environment == "JAX":
                return PyTorch2JAXConverter().convert_to_jax_model(model)
            else:
                return model
        else:
            raise NotImplementedError(f"Unsupported 'use' value: {use}")


class NeighborlistForInferenceNonUniquePairs(torch.nn.Module):

    def __init__(self, cutoffs: Dict[str, float]):
        """
        Initialize the ComputeInteractingAtomPairs module.

        Parameters
        ----------
        cutoffs : Dict[str, unit.Quantity]
            The cutoff distance(s) for neighbor list calculations.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()
        keys_list: List[str] = []
        cutoffs_list: List[float] = []
        for key, value in cutoffs.items():
            keys_list.append(str(key))
            cutoffs_list.append(float(value))

        self.cutoff_keys = keys_list
        self.register_buffer("cutoff", torch.tensor(cutoffs_list))

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            A Dict for each cutoff type, where each entry is a namedtuple containing the pair indices, Euclidean distances
            (d_ij), and displacement vectors (r_ij).
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)
        # Generate a range of indices from 0 to n-1
        indices = torch.arange(n, device=atomic_subsystem_indices.device)

        # Create a meshgrid of indices
        i_final_pairs, j_final_pairs = torch.meshgrid(indices, indices, indexing="ij")

        # Filter out the diagonal elements (where i == j)
        mask = i_final_pairs != j_final_pairs
        i_final_pairs = i_final_pairs[mask]
        j_final_pairs = j_final_pairs[mask]
        # calculate r_ij and d_ij
        r_ij = positions[i_final_pairs] - positions[j_final_pairs]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True, p=2)
        pair_indices = torch.stack((i_final_pairs, j_final_pairs))

        # Initialize interaction mask in a TorchScript-friendly way
        interaction_mask: Dict[str, torch.Tensor] = {}
        for key in self.cutoff_keys:
            # cutoff_value = self.get_buffer(key)
            bool_mask = d_ij <= 0.4
            interaction_mask[key] = bool_mask

        return PairlistData(
            pair_indices=pair_indices, d_ij=d_ij, r_ij=r_ij, mask=interaction_mask
        )



class NeighborlistForInferenceNonUniquePairs(torch.nn.Module):

    def __init__(self, cutoffs: Dict[str, float]):
        """
        Initialize the ComputeInteractingAtomPairs module.

        Parameters
        ----------
        cutoffs : Dict[str, unit.Quantity]
            The cutoff distance(s) for neighbor list calculations.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()
        keys_list: List[str] = []
        cutoffs_list: List[float] = []
        for key, value in cutoffs.items():
            keys_list.append(str(key))
            cutoffs_list.append(float(value))

        self.cutoff_keys = keys_list
        self.register_buffer("cutoff", torch.tensor(cutoffs_list))

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            A Dict for each cutoff type, where each entry is a namedtuple containing the pair indices, Euclidean distances
            (d_ij), and displacement vectors (r_ij).
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)
        # Generate a range of indices from 0 to n-1
        indices = torch.arange(n, device=atomic_subsystem_indices.device)

        # Create a meshgrid of indices
        i_final_pairs, j_final_pairs = torch.meshgrid(indices, indices, indexing="ij")

        # Filter out the diagonal elements (where i == j)
        mask = i_final_pairs != j_final_pairs
        i_final_pairs = i_final_pairs[mask]
        j_final_pairs = j_final_pairs[mask]
        # calculate r_ij and d_ij
        r_ij = positions[i_final_pairs] - positions[j_final_pairs]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True, p=2)
        pair_indices = torch.stack((i_final_pairs, j_final_pairs))

        # Initialize interaction mask in a TorchScript-friendly way
        interaction_mask: Dict[str, torch.Tensor] = {}
        for key in self.cutoff_keys:
            # cutoff_value = self.get_buffer(key)
            bool_mask = d_ij <= 0.4
            interaction_mask[key] = bool_mask

        return PairlistData(
            pair_indices=pair_indices, d_ij=d_ij, r_ij=r_ij, mask=interaction_mask
        )



class ComputeInteractingAtomPairs(torch.nn.Module):
    """
    A module for preparing input data, including the calculation of pair lists,
    distances (d_ij), and displacement vectors (r_ij) for molecular simulations.
    """

    def __init__(self, cutoffs: Dict[str, float], only_unique_pairs: bool = True):
        """
        Initialize the ComputeInteractingAtomPairs module.

        Parameters
        ----------
        cutoffs : Dict[str, unit.Quantity]
            The cutoff distance(s) for neighbor list calculations.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()
        from .models import Neighborlist

        self.only_unique_pairs = only_unique_pairs
        self.calculate_distances_and_pairlist = Neighborlist(cutoffs, only_unique_pairs)

    def forward(self, data: Union[NNPInput, NamedTuple]):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            A Dict for each cutoff type, where each entry is a namedtuple containing the pair indices, Euclidean distances
            (d_ij), and displacement vectors (r_ij).
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

        # this will return a Dict of the PairListOutputs for each cutoff we specify
        return pairlist_output

    def _input_checks(self, data: Union[NNPInput, NamedTuple]):
        """
        Performs input validation checks.
        Ensures the input data conforms to expected shapes and types.

        Parameters
        ----------
        data : NNPInput
            The input data to be validated.

        Raises
        ------
        ValueError
            If the input data does not meet the expected criteria.
        """
        # check that the input is instance of NNPInput
        assert isinstance(data, NNPInput) or isinstance(data, Tuple)

        nr_of_atoms = data.atomic_numbers.shape[0]
        assert data.atomic_numbers.shape == torch.Size([nr_of_atoms])
        assert data.atomic_subsystem_indices.shape == torch.Size([nr_of_atoms])
        nr_of_molecules = torch.unique(data.atomic_subsystem_indices).numel()
        assert data.total_charge.shape == torch.Size([nr_of_molecules])
        assert data.positions.shape == torch.Size([nr_of_atoms, 3])


from torch.nn import ModuleDict

from modelforge.potential.processing import PerAtomEnergy


class PostProcessing(torch.nn.Module):
    """
    A module for handling post-processing operations on model outputs, including
    normalization, calculation of atomic self-energies, and reduction operations
    to compute per-molecule properties from per-atom properties.
    """

    _SUPPORTED_PROPERTIES = ["per_atom_energy", "general_postprocessing_operation"]

    def __init__(
        self,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Dict[str, Dict[str, float]],
    ):
        """
        Initialize the PostProcessing module.

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

        if "per_atom_energy" in postprocessing_parameter:
            self.registered_chained_operations["per_atom_energy"] = PerAtomEnergy(
                postprocessing_parameter["per_atom_energy"],
                dataset_statistic["training_dataset_statistics"],
            )
            self._registered_properties.append("per_atom_energy")

    def forward(self, data: Dict[str, torch.Tensor]):
        """
        Perform post-processing operations for all registered properties.

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
        if "per_atom_energy" in self._registered_properties:

            per_molecule_energy = self.registered_chained_operations[
                "per_atom_energy"
            ].forward(data["per_atom_energy"], data["atomic_subsystem_indices"])

            processed_data["per_molecule_energy"] = per_molecule_energy
            processed_data["per_atom_energy"] = data["per_atom_energy"].detach()

        return processed_data


class BaseNetwork(Module):
    """
    The BaseNetwork wraps the input preparation (including pairlist calculation,
    d_ij and r_ij calculation), the actual model as well as the output
    preparation in a wrapper class.

    Learned parameters are present only in the core model, the input preparation
    and output preparation are not learned.
    """

    def __init__(
        self,
        *,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]],
        maximum_interaction_radius: unit.Quantity,
        maximum_dispersion_interaction_radius: Optional[unit.Quantity] = None,
        maximum_coulomb_interaction_radius: Optional[unit.Quantity] = None,
        potential_seed: Optional[int] = None,
    ):
        """
        Initialize the BaseNetwork.

        Parameters
        ----------
        postprocessing_parameter : Dict[str, Dict[str, bool]]
            Parameters for postprocessing.
        dataset_statistic : Optional[Dict[str, float]]
            Dataset statistics for normalization.
        maximum_interaction_radius : unit.Quantity
            cutoff radius for local interactions
        maximum_dispersion_interaction_radius : unit.Quantity, optional
            cutoff radius for dispersion interactions.
        maximum_coulomb_interaction_radius : unit.Quantity, optional
            cutoff radius for Coulomb interactions.
        potential_seed : Optional[int], optional
            Value used for torch.manual_seed, by default None.
        """

        super().__init__()
        from modelforge.utils.units import _convert_str_to_unit

        if potential_seed:
            import torch

            torch.manual_seed(potential_seed)

            # according to https://docs.ray.io/en/latest/tune/faq.html#how-can-i-reproduce-experiments
            # we should also set the same seed for numpy.random and python random module
            # when doing hyperparameter optimization with ray tune.  E.g., the ASHA scheduler relies on numpy.random
            # and doesn't take a seed as an argument, so we need to set it here.
            import numpy as np

            np.random.seed(potential_seed)

            import random

            random.seed(potential_seed)

        self.postprocessing = PostProcessing(
            postprocessing_parameter, dataset_statistic
        )

        # check if self.only_unique_pairs is set in child class
        if not hasattr(self, "only_unique_pairs"):
            raise RuntimeError(
                "The only_unique_pairs attribute is not set in the child class. Please set it to True or False before calling super().__init__."
            )

        # to handle multiple cutoffs, we will create a dictionary with the cutoffs
        # the dictionary will make it more transparent which PairListOutputs belong to which cutoff

        cutoffs = {}
        cutoffs["maximum_interaction_radius"] = _convert_str_to_unit(
            maximum_interaction_radius
        )
        if maximum_dispersion_interaction_radius is not None:
            cutoffs["maximum_dispersion_interaction_radius"] = _convert_str_to_unit(
                maximum_dispersion_interaction_radius
            )
        if maximum_coulomb_interaction_radius is not None:
            cutoffs["maximum_coulomb_interaction_radius"] = _convert_str_to_unit(
                maximum_coulomb_interaction_radius
            )

        self.compute_interacting_pairs = ComputeInteractingAtomPairs(
            cutoffs=cutoffs,
            only_unique_pairs=self.only_unique_pairs,
        )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
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

    def prepare_pairwise_properties(self, data):
        """
        Prepare the pairwise properties for the model.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            The input data to prepare.

        Returns
        -------
        PairListOutputs
            The prepared pairwise properties.
        """

        self.compute_interacting_pairs._input_checks(data)
        return self.compute_interacting_pairs.forward(data)

    def compute(self, data, core_input):
        """
        Compute the core model's output.

        Parameters
        ----------
        data : Union[NNPInput, NamedTuple]
            The input data.
        core_input : PairListOutputs
            The prepared pairwise properties.

        Returns
        -------
        Any
            The core model's output.
        """
        return self.core_module(data, core_input)

    def forward(self, input_data: NNPInput):
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
        processed_output = self.postprocessing(output)
        return processed_output


class CoreNetwork(Module, ABC):
    """
    The CoreNetwork implements methods that are used by all neural network
    potentials. Every network inherits from CoreNetwork. Networks take in a
    NNPInput and pairlist and return a dictionary of atomic properties.

    Operations performed outside the network (e.g., pairlist calculation and
    operations that reduce atomic properties to molecule properties) are not
    part of the network and are implemented in the BaseNetwork, which is a
    wrapper around the CoreNetwork.
    """

    def __init__(self, activation_function: Type[torch.nn.Module]):
        """
        Initialize the CoreNetwork.

        Parameters
        ----------
        activation_function : Type[torch.nn.Module]
            The activation function to use.
        """

        super().__init__()

        self.activation_function = activation_function

    @abstractmethod
    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist: PairlistData
    ) -> Union[
        "PhysNetNeuralNetworkData",
        "PaiNNNeuralNetworkData",
        "SchnetNeuralNetworkData",
        "AniNeuralNetworkData",
        "SAKENeuralNetworkInput",
    ]:
        """
        Prepares model-specific inputs before the forward pass.

        This abstract method should be implemented by subclasses to accommodate
        any model-specific preprocessing of inputs.

        Parameters
        ----------
        data : NNPInput
            The initial inputs to the neural network model, including atomic
            numbers, positions, and other relevant data.
        pairlist : PairListOutputs
            The outputs of a pairlist calculation, including pair indices,
            distances, and displacement vectors.

        Returns
        -------
        NeuralNetworkData
            The processed inputs, ready for the model's forward pass.
        """
        pass

    @abstractmethod
    def compute_properties(
        self,
        data: Union[
            "PhysNetNeuralNetworkData",
            "PaiNNNeuralNetworkData",
            "SchnetNeuralNetworkData",
            "AniNeuralNetworkData",
            "SAKENeuralNetworkInput",
        ],
    ):
        """
        Defines the forward pass of the model.

        This abstract method should be implemented by subclasses to specify the
        model's computation from inputs (processed input data) to outputs (per
        atom properties).

        Parameters
        ----------
        data : Union[
            "PhysNetNeuralNetworkData", "PaiNNNeuralNetworkData",
            "SchnetNeuralNetworkData", "AniNeuralNetworkData",
            "SAKENeuralNetworkInput",
        ]
            The processed input data, specific to the model's requirements.

        Returns
        -------
        Dict[str, torch.Tensor]
            The model's output as computed from the inputs.
        """
        pass

    def load_pretrained_weights(self, path: str):
        """
        Loads pretrained weights into the model from the specified path.

        Parameters
        ----------
        path : str
            The path to the file containing the pretrained weights.

        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()  # Set the model to evaluation mode

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the forward pass through the network.

        Parameters
        ----------
        data : NNPInput
            Contains input data for the batch obtained directly from the
            dataset, including atomic numbers, positions, and other relevant
            fields.
        pairlist_output : PairListOutputs
            Contains the indices for the selected pairs and their associated
            distances and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            The calculated per-atom properties and other properties from the
            forward pass.
        """
        # perform model specific modifications
        nnp_input = self._model_specific_input_preparation(data, pairlist_output)
        # perform the forward pass implemented in the subclass
        outputs = self.compute_properties(nnp_input)
        # add atomic numbers to the output
        outputs["atomic_numbers"] = data.atomic_numbers

        return outputs


class Potential(torch.nn.Module):
    def __init__(self, core_network, neighborlist, postprocessing):
        super().__init__()
        self.core_network = torch.jit.script(core_network)
        self.neighborlist = torch.jit.script(neighborlist)
        self.postprocessing = torch.jit.script(postprocessing)

    def forward(self, input_data: NNPInputTuple):
        # Step 1: Compute pair list and distances using Neighborlist
        pairlist_output = self.neighborlist.forward(input_data)

        # Step 2: Compute the core network output using SchNetCore
        core_output = self.core_network.forward(input_data, pairlist_output)

        # Step 3: Apply postprocessing using PostProcessing
        processed_output = self.postprocessing.forward(core_output)

        return processed_output

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    NamedTuple,
    Tuple,
    Type,
    Union,
    Optional,
)

import lightning as pl
import torch
from loguru import logger as log
from openff.units import unit
from torch.nn import Module

from modelforge.potential.utils import NNPInput

if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics
    from modelforge.potential.ani import ANI2x, AniNeuralNetworkData
    from modelforge.potential.painn import PaiNN, PaiNNNeuralNetworkData
    from modelforge.potential.physnet import PhysNet, PhysNetNeuralNetworkData
    from modelforge.potential.sake import SAKE, SAKENeuralNetworkInput
    from modelforge.potential.schnet import SchNet, SchnetNeuralNetworkData


# Define NamedTuple for the outputs of Pairlist and Neighborlist forward method
class PairListOutputs(NamedTuple):
    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class EnergyOutput(NamedTuple):
    E: torch.Tensor
    raw_E: torch.Tensor
    E_i: torch.Tensor
    molecular_ase: torch.Tensor


class Pairlist(Module):
    """Handle pair list calculations for atoms, returning atom indices pairs and displacement vectors.

    Attributes
    ----------
    calculate_pairs : callable
        A function to calculate pairs given specific criteria.

    Methods
    -------
    calculate_r_ij(pair_indices, positions)
        Computes the displacement vector between atom pairs.
    calculate_d_ij(r_ij)
        Computes the Euclidean distance between atoms in a pair.
    forward(positions, atomic_subsystem_indices)
        Forward pass to compute pair indices, distances, and displacement vectors.
    """

    def __init__(self, only_unique_pairs: bool = False):
        """
        Parameters
        ----------
        only_unique_pairs : bool, optional
            If True, only unique pairs are returned (default is False).
            Otherwise, all pairs are returned.
        """
        super().__init__()
        self.only_unique_pairs = only_unique_pairs

    def enumerate_all_pairs(self, atomic_subsystem_indices: torch.Tensor):
        """Compute all pairs of atoms and their distances.

        Parameters
        ----------
        atomic_subsystem_indices : torch.Tensor, shape (nr_atoms_per_systems)
            Atom indices to indicate which atoms belong to which molecule
        """
        # generate index grid
        n = len(atomic_subsystem_indices)

        # get device that passed tensors lives on, initialize on the same device
        device = atomic_subsystem_indices.device

        i_final_pairs = torch.tensor([], dtype=torch.int64, device=device)
        j_final_pairs = torch.tensor([], dtype=torch.int64, device=device)

        # to avoid allocating very large tensors, we will wrap this in a loop where we allocate tensors of size n-1
        # and concatenate the results, rather than allocating tensors of size n*(n-1).

        for ii in range(0, n):
            # Repeat each number n-1 times for i_indices
            i_indices = torch.repeat_interleave(
                torch.tensor(ii, device=device),
                repeats=n - 1,
            )

            # Correctly construct j_indices
            j_indices = torch.cat(
                (
                    torch.arange(ii, device=device),
                    torch.arange(ii + 1, n, device=device),
                )
            )

            # filter pairs to only keep those belonging to the same molecule
            same_molecule_mask = (
                atomic_subsystem_indices[i_indices]
                == atomic_subsystem_indices[j_indices]
            )
            i_final_pairs_temp = i_indices[same_molecule_mask]
            j_final_pairs_temp = j_indices[same_molecule_mask]

            if self.only_unique_pairs:
                # filter out pairs that are not unique
                unique_pairs_mask = i_final_pairs_temp < j_final_pairs_temp
                i_final_pairs_temp = i_final_pairs_temp[unique_pairs_mask]
                j_final_pairs_temp = j_final_pairs_temp[unique_pairs_mask]

            i_final_pairs = torch.cat((i_final_pairs, i_final_pairs_temp), dim=0)
            j_final_pairs = torch.cat((j_final_pairs, j_final_pairs_temp), dim=0)

        # concatenate to form final (2, n_pairs) tensor
        pair_indices = torch.stack((i_final_pairs, j_final_pairs))

        return pair_indices.to(device)

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
    ) -> PairListOutputs:
        """
        Compute interacting pairs, distances, and displacement vectors.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices to identify atoms in subsystems. Shape: [nr_atoms].
        only_unique_pairs : bool, optional
            If True, considers only unique pairs of atoms. Default is False.

        Returns
        -------
        PairListOutputs
            A NamedTuple containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """
        pair_indices = self.enumerate_all_pairs(
            atomic_subsystem_indices,
        )
        r_ij = self.calculate_r_ij(pair_indices, positions)

        return PairListOutputs(
            pair_indices=pair_indices,
            d_ij=self.calculate_d_ij(r_ij),
            r_ij=r_ij,
        )


class Neighborlist(Pairlist):
    """Manage neighbor list calculations with a specified cutoff distance.

    This class extends Pairlist to consider a cutoff distance for neighbor calculations.

    Attributes
    ----------
    cutoff : unit.Quantity
        Cutoff distance for neighbor list calculations.
    """

    def __init__(self, cutoff: unit.Quantity, only_unique_pairs: bool = False):
        """
        Initialize the Neighborlist with a specific cutoff distance.

        Parameters
        ----------
        cutoff : unit.Quantity
            Cutoff distance for neighbor calculations.
        """
        super().__init__(only_unique_pairs=only_unique_pairs)
        self.register_buffer("cutoff", torch.tensor(cutoff.to(unit.nanometer).m))

    def forward(
        self,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> PairListOutputs:
        """
        Forward pass to compute neighbor list considering a cutoff distance.

        Overrides the `forward` method from Pairlist to include cutoff distance in calculations.

        Parameters
        ----------
        positions : torch.Tensor
            Atom positions. Shape: [nr_systems, nr_atoms, 3].
        atomic_subsystem_indices : torch.Tensor
            Indices identifying atoms in subsystems. Shape: [nr_atoms].
        only_unique_pairs : bool, optional
            If True, considers only unique pairs of atoms. Default is False.

        Returns
        -------
        PairListOutputs
            A NamedTuple containing 'pair_indices', 'd_ij' (distances), and 'r_ij' (displacement vectors).
        """

        if pair_indices is None:
            pair_indices = self.enumerate_all_pairs(
                atomic_subsystem_indices,
            )

        r_ij = self.calculate_r_ij(pair_indices, positions)
        d_ij = self.calculate_d_ij(r_ij)

        # Find pairs within the cutoff
        in_cutoff = (d_ij <= self.cutoff).squeeze()
        # Get the atom indices within the cutoff
        pair_indices_within_cutoff = pair_indices[:, in_cutoff]

        return PairListOutputs(
            pair_indices=pair_indices_within_cutoff,
            d_ij=d_ij[in_cutoff],
            r_ij=r_ij[in_cutoff],
        )


from typing import Callable, Literal, Optional, Union

import numpy as np


class JAXModel:
    """A model wrapper that facilitates calling a JAX function with predefined parameters and buffers.

    Attributes
    ----------
    jax_fn : Callable
        The JAX function to be called.
    parameter : jax.
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

        import functorch
        import jax
        from functorch import make_functional_with_buffers
        from jax import custom_vjp
        from pytorch2jax.pytorch2jax import convert_to_jax, convert_to_pyt

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


from modelforge.potential.processing import AtomicSelfEnergies


class NeuralNetworkPotentialFactory:
    """
    Factory class for creating instances of neural network potentials (NNP) that are traceable/scriptable and can be exported to torchscript.

    This factory allows for the creation of specific NNP instances configured for either
    training or inference purposes based on the given parameters.

    Methods
    -------
    create_nnp(use, nnp_type, nnp_parameters=None, training_parameters=None)
        Creates an instance of a specified NNP type configured for either training or inference.
    """

    @staticmethod
    def create_nnp(
        *,
        use: Literal["training", "inference"],
        model_type: Literal["ANI2x", "SchNet", "PaiNN", "SAKE", "PhysNet"],
        model_parameters: Dict[str, Union[int, float, str]],
        simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
        training_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[Type[torch.nn.Module], Type[JAXModel], Type[pl.LightningModule]]:
        """
        Creates an NNP instance of the specified type, configured either for training or inference.

        Parameters
        ----------
        use : str
            The use case for the NNP instance.
        nnp_name : str
            The type of NNP to instantiate.
        simulation_environment : str
            The environment to use, either 'PyTorch' or 'JAX'.
        nnp_parameters : dict, optional
            Parameters specific to the NNP model, by default {}.
        training_parameters : dict, optional
            Parameters for configuring the training, by default {}.

        Returns
        -------
        Union[Union[ANI2x, SchNet, PaiNN, PhysNet], TrainingAdapter]
            An instantiated NNP model.

        Raises
        ------
        ValueError
            If an unknown use case is requested.
        NotImplementedError
            If the requested NNP type is not implemented.
        """

        from modelforge.potential import _Implemented_NNPs
        from modelforge.train.training import TrainingAdapter

        model_parameters = model_parameters or {}
        training_parameters = training_parameters or {}

        log.debug(f"{training_parameters=}")
        # get NNP
        nnp_class: Type = _Implemented_NNPs.get_neural_network_class(model_type)
        if nnp_class is None:
            raise NotImplementedError(f"NNP type {model_type} is not implemented.")

        # add modifications to NNP if requested
        if use == "training":
            if simulation_environment == "JAX":
                log.warning(
                    "Training in JAX is not availalbe. Falling back to PyTorch."
                )
            model_parameters["nnp_name"] = model_type
            return TrainingAdapter(
                nnp_parameters=model_parameters, **training_parameters
            )
        elif use == "inference":
            # if this model_parameter dictionary ahs already been used
            # for training the `nnp_name` might have been set
            if "nnp_name" in model_parameters:
                del model_parameters["nnp_name"]
            nnp_instance = nnp_class(**model_parameters)
            if simulation_environment == "JAX":
                return PyTorch2JAXConverter().convert_to_jax_model(nnp_instance)
            else:
                return nnp_instance
        else:
            raise ValueError(f"Unsupported 'use' value: {use}")


class InputPreparation(torch.nn.Module):
    def __init__(self, cutoff: unit.Quantity, only_unique_pairs: bool = True):
        """
        Parameters
        ----------
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by default True.
        """

        super().__init__()
        from .models import Neighborlist

        self.calculate_distances_and_pairlist = Neighborlist(cutoff, only_unique_pairs)

    def prepare_inputs(self, data: Union[NNPInput, NamedTuple]):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating distances
        and generating the pair list. It also calls the model-specific input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers, positions,
            and other necessary information.

        Returns
        -------
        The processed input data, ready for the models forward pass.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        pairlist_output = self.calculate_distances_and_pairlist(
            positions=positions,
            atomic_subsystem_indices=atomic_subsystem_indices,
            pair_indices=data.pair_list.to(torch.int64),
        )

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


class BaseNetwork(Module):

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # Prefix to remove
        prefix = "model."

        # check if prefix is present
        if any(key.startswith(prefix) for key in state_dict.keys()):
            # Create a new dictionary without the prefix in the keys if prefix exists
            new_d = {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
            log.debug(f"Removed prefix: {prefix}")
        else:
            log.debug("No prefix found. No modifications to keys in state loading.")

        super().load_state_dict(new_d, strict=strict, assign=assign)

    def forward(self, data: NNPInput):
        # perform input checks
        self.input_preparation._input_checks(data)
        # prepare the input for the forward pass
        pairlist_output = self.input_preparation.prepare_inputs(data)
        return self.core_module(data, pairlist_output)


class CoreNetwork(Module, ABC):
    """Abstract base class for neural network potentials.

    Attributes
    ----------
    cutoff : unit.Quantity
        Cutoff distance for neighbor list calculations.
    calculate_distances_and_pairlist : Neighborlist
        Module for calculating distances and pairlist with a given cutoff.
    readout_module : FromAtomToMoleculeReduction
        Module for reading out per molecule properties from atomic properties.
    """

    def __init__(
        self,
        cutoff: unit.Quantity,
    ):
        """
        Initializes the neural network potential class with specified parameters.

        """

        super().__init__()
        self._dtype: Optional[bool] = None  # set at runtime
        self._log_message_dtype = False
        self._log_message_units = False
        # initialize the per molecule readout module
        from .processing import EnergyScaling, FromAtomToMoleculeReduction

        self.postprocessing = EnergyScaling()

        self.readout_module = FromAtomToMoleculeReduction()

    @abstractmethod
    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist: PairListOutputs
    ) -> Union[
        "PhysNetNeuralNetworkData",
        "PaiNNNeuralNetworkData",
        "SchnetNeuralNetworkData",
        "AniNeuralNetworkData",
        "SAKENeuralNetworkInput",
    ]:
        """
        Prepares model-specific inputs before the forward pass.

        This abstract method should be implemented by subclasses to accommodate any
        model-specific preprocessing of inputs.

        Parameters
        ----------
        data : NNPInput
            The initial inputs to the neural network model, including atomic numbers,
            positions, and other relevant data.
        pairlist : PairListOutputs
            The outputs of a pair list calculation, including pair indices, distances,
            and displacement vectors.

        Returns
        -------
        NeuralNetworkData: The processed inputs, ready for the model's forward pass.
        """
        pass

    @abstractmethod
    def _forward(
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

        This abstract method should be implemented by subclasses to specify the model's
        computation from inputs to outputs.

        Parameters
        ----------
        inputs : The processed input data, specific to the model's requirements.

        Returns
        -------
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

    def _readout(
        self, atom_specific_values: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the readout operation to generate per-molecule properties from atomic properties.

        Parameters
        ----------
        atom_specific_values : torch.Tensor
            The tensor containing atomic properties.
        index : torch.Tensor
            The tensor indicating the molecule to which each atom belongs.

        Returns
        -------
        torch.Tensor
            The tensor containing per-molecule properties.
        """
        return self.readout_module(atom_specific_values, index)

    def forward(self, data: NNPInput, pairlist_output) -> EnergyOutput:
        """
        Defines the forward pass of the neural network potential.

        Parameters
        ----------
        data : NNPInput
            The input data for the model, containing atomic numbers, positions, and other relevant fields.

        Returns
        -------
        EnergyOutput
            The calculated energies and other properties from the forward pass.
        """
        # ---------------------------
        # perform model specific modifications
        nnp_input = self._model_specific_input_preparation(data, pairlist_output)
        # perform the forward pass implemented in the subclass
        outputs = self._forward(nnp_input)
        # sum over atomic properties to generate per molecule properties
        E = self._readout(
            atom_specific_values=outputs["E_i"],
            index=outputs["atomic_subsystem_indices"],
        )
        # postprocess energies: add atomic self energies,
        # and other constant factors used to optionally normalize the data range of the training dataset
        processed_energy = self.postprocessing._energy_postprocessing(E, nnp_input)
        # from icecream import ic

        # ic(processed_energy["E"])
        # return energies
        return EnergyOutput(
            E=processed_energy["E"],
            raw_E=processed_energy["raw_E"],
            E_i=outputs["E_i"],
            molecular_ase=processed_energy["molecular_ase"],
        )

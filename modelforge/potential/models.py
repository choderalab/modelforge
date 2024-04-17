from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Tuple, Type, Mapping

import torch
from loguru import logger as log
from openff.units import unit
from torch.nn import Module

from modelforge.dataset.dataset import DatasetStatistics
from modelforge.potential.utils import AtomicSelfEnergies, NNPInput

if TYPE_CHECKING:
    from modelforge.dataset.dataset import DatasetStatistics
    from modelforge.potential.ani import ANI2x, AniNeuralNetworkData
    from modelforge.potential.painn import PaiNN, PaiNNNeuralNetworkData
    from modelforge.potential.physnet import PhysNet, PhysNetNeuralNetworkData
    from modelforge.potential.schnet import SchNet, SchnetNeuralNetworkData


# Define NamedTuple for the outputs of Pairlist and Neighborlist forward method
class PairListOutputs(NamedTuple):
    pair_indices: torch.Tensor
    d_ij: torch.Tensor
    r_ij: torch.Tensor


class EnergyOutput(NamedTuple):
    E: torch.Tensor
    raw_E: torch.Tensor
    rescaled_E: torch.Tensor
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
    forward(positions, atomic_subsystem_indices, only_unique_pairs=False)
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

        if self.only_unique_pairs:
            i_indices, j_indices = torch.triu_indices(n, n, 1, device=device)
        else:
            # Repeat each number n-1 times for i_indices
            i_indices = torch.repeat_interleave(
                torch.arange(n, device=device), repeats=n - 1
            )

            # Correctly construct j_indices
            j_indices = torch.cat(
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

        # filter pairs to only keep those belonging to the same molecule
        same_molecule_mask = (
            atomic_subsystem_indices[i_indices] == atomic_subsystem_indices[j_indices]
        )

        # Apply mask to get final pair indices
        i_final_pairs = i_indices[same_molecule_mask]
        j_final_pairs = j_indices[same_molecule_mask]

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


from typing import Literal, Optional, Union, Callable
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

    def convert_to_jax_model(
        self, nnp_instance: Union["ANI2x", "SchNet", "PaiNN", "PhysNet"]
    ) -> JAXModel:
        """Converts a PyTorch neural network potential instance to a JAXModel.

        Parameters
        ----------
        nnp_instance : Any
            The neural network potential instance to convert.

        Returns
        -------
        JAXModel
            The converted JAX model.
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

        import jax
        from jax import custom_vjp
        from pytorch2jax.pytorch2jax import convert_to_jax, convert_to_pyt
        import functorch
        from functorch import make_functional_with_buffers

        # skip input checks
        nnp_instance.mode = "fast"

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
            # Convert the input data from PyTorch to JAX representations
            params, args, kwargs = map(
                lambda x: jax.tree_map(convert_to_pyt, x), (params, args, kwargs)
            )
            # Apply the model function to the input data
            out = model_fn(params, *args, **kwargs)
            # Convert the output data from JAX to PyTorch representations
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
        use: Literal["training", "inference"],
        nnp_name: Literal["ANI2x", "SchNet", "PaiNN", "SAKE", "PhysNet"],
        simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
        nnp_parameters: Optional[Dict[str, Union[int, float, str]]] = None,
        training_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[Type[torch.nn.Module], Type[JAXModel]]:
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

        from modelforge.potential import _IMPLEMENTED_NNPS
        from modelforge.train.training import TrainingAdapter

        nnp_parameters = nnp_parameters or {}
        training_parameters = training_parameters or {}

        # get NNP
        nnp_class: Type = _IMPLEMENTED_NNPS.get(nnp_name)
        if nnp_class is None:
            raise NotImplementedError(f"NNP type {nnp_name} is not implemented.")

        # add modifications to NNP if requested
        if use == "training":
            if simulation_environment == "JAX":
                log.warning(
                    "Training in JAX is not availalbe. Falling back to PyTorch."
                )
            nnp_parameters["nnp_name"] = nnp_name
            return TrainingAdapter(nnp_parameters=nnp_parameters, **training_parameters)
        elif use == "inference":
            nnp_instance = nnp_class(**nnp_parameters)
            if simulation_environment == "JAX":
                return PyTorch2JAXConverter().convert_to_jax_model(nnp_instance)
            else:
                return nnp_instance
        else:
            raise ValueError(f"Unsupported 'use' value: {use}")


from modelforge.potential.utils import NeuralNetworkData


class Postprocessing:

    def __init__(self) -> None:
        from modelforge.dataset.dataset import DatasetStatistics

        self._dataset_statistics = DatasetStatistics(0.0, 1.0, AtomicSelfEnergies())

    def _calculate_molecular_self_energy(
        self, data: NeuralNetworkData, number_of_molecules: int
    ) -> torch.Tensor:
        """
        Calculates the molecular self energy.

        Parameters
        ----------
        data : NNPInput
            The input data for the model, including atomic numbers and subsystem indices.
        number_of_molecules : int
            The number of molecules in the batch.

        Returns
        -------
        torch.Tensor
            The tensor containing the molecular self energy for each molecule.
        """

        atomic_numbers = data.atomic_numbers
        atomic_subsystem_indices = data.atomic_subsystem_indices.to(
            dtype=torch.long, device=atomic_numbers.device
        )

        # atomic_number_to_energy
        atomic_self_energies = self.dataset_statistics.atomic_self_energies
        ase_tensor_for_indexing = atomic_self_energies.ase_tensor_for_indexing.to(
            device=atomic_numbers.device
        )

        # first, we need to use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        # then, we use the atomic_subsystem_indices to scatter add the atomic self
        # energies in the ase_tensor to generate the molecular self energies
        ase_tensor_zeros = torch.zeros((number_of_molecules,)).to(
            device=atomic_numbers.device
        )
        ase_tensor = ase_tensor_zeros.scatter_add(
            0, atomic_subsystem_indices, ase_tensor
        )

        return ase_tensor

    def _rescale_energy(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Rescales energies using the dataset statistics.

        Parameters
        ----------
        energies : torch.Tensor
            The tensor of energies to be rescaled.

        Returns
        -------
        torch.Tensor
            The rescaled energies.
        """

        return (
            energies * self.dataset_statistics.scaling_stddev
            + self.dataset_statistics.scaling_mean
        )

    def _energy_postprocessing(
        self, properties_per_molecule: torch.Tensor, inputs: NeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Postprocesses the energies by rescaling and adding molecular self energy.

        Parameters
        ----------
        properties_per_molecule : The properties computed per molecule.
        inputs : The original input data to the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The dictionary containing the postprocessed energy tensors.
        """

        # first, resale the energies
        processed_energy = {}
        processed_energy["raw_E"] = properties_per_molecule.clone().detach()
        properties_per_molecule = self._rescale_energy(properties_per_molecule)
        processed_energy["rescaled_E"] = properties_per_molecule.clone().detach()
        # then, calculate the molecular self energy
        molecular_ase = self._calculate_molecular_self_energy(
            inputs, properties_per_molecule.numel()
        )
        processed_energy["molecular_ase"] = molecular_ase.clone().detach()
        # add the molecular self energy to the rescaled energies
        processed_energy["E"] = properties_per_molecule + molecular_ase
        return processed_energy

    @property
    def dataset_statistics(self):
        """
        Property for accessing the model's dataset statistics.

        Returns
        -------
        DatasetStatistics
            The dataset statistics associated with the model.
        """

        return self._dataset_statistics

    @dataset_statistics.setter
    def dataset_statistics(self, value: "DatasetStatistics"):
        """
        Sets the dataset statistics for the model.

        Parameters
        ----------
        value : DatasetStatistics
            The new dataset statistics to be set for the model.
        """

        if not isinstance(value, DatasetStatistics):
            raise ValueError("Value must be an instance of DatasetStatistics.")

        self._dataset_statistics = value

    def update_dataset_statistics(self, **kwargs):
        """
        Updates specific fields of the model's dataset statistics.

        Parameters
        ----------
        **kwargs
            Fields and their new values to update in the dataset statistics.
        """

        for key, value in kwargs.items():
            if hasattr(self.dataset_statistics, key):
                setattr(self.dataset_statistics, key, value)
            else:
                log.warning(f"{key} is not a valid field of DatasetStatistics.")


class BaseNeuralNetworkPotential(Module, ABC):
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
        only_unique_pairs: bool = False,
        mode: Literal["safe", "fast"] = "safe",
    ):
        """
        Initializes the neural network potential class with specified parameters.

        Parameters
        ----------
        cutoff : openff.units.unit.Quantity
            Cutoff distance for the neighbor list calculations.
        """

        from .models import Neighborlist

        self.mode = mode
        super().__init__()
        self.calculate_distances_and_pairlist = Neighborlist(
            cutoff, only_unique_pairs=only_unique_pairs
        )
        self._log_message_dtype = False
        self._log_message_units = False
        # initialize the per molecule readout module
        from .utils import FromAtomToMoleculeReduction

        self.postprocessing = Postprocessing()

        self.readout_module = FromAtomToMoleculeReduction()

    @abstractmethod
    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist: PairListOutputs
    ) -> Union[
        "PhysNetNeuralNetworkData",
        "PaiNNNeuralNetworkData",
        "SchnetNeuralNetworkData",
        "AniNeuralNetworkData",
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

    def prepare_inputs(self, data: NNPInput):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating distances
        and generating the pair list. It also calls the model-specific input preparation.

        Parameters
        ----------
        data : NNPInput
            The input data provided by the dataset, containing atomic numbers, positions,
            and other necessary information.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by default True.

        Returns
        -------
        The processed input data, ready for the model's forward pass.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        pairlist_output = self.calculate_distances_and_pairlist(
            positions, atomic_subsystem_indices
        )

        # ---------------------------
        # perform model specific modifications
        nnp_input = self._model_specific_input_preparation(data, pairlist_output)

        return nnp_input

    def _input_checks(self, data: Union[NamedTuple, NNPInput]):
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

    def forward(self, data: NNPInput) -> EnergyOutput:
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
        # perform input checks
        if self.mode == "fast":
            self._input_checks(data)
        # prepare the input for the forward pass
        inputs = self.prepare_inputs(data)
        # perform the forward pass implemented in the subclass
        outputs = self._forward(inputs)
        # sum over atomic properties to generate per molecule properties
        E = self._readout(
            atom_specific_values=outputs["E_i"],
            index=outputs["atomic_subsystem_indices"],
        )
        # postprocess energies: add atomic self energies,
        # and other constant factors used to optionally normalize the data range of the training dataset
        processed_energy = self.postprocessing._energy_postprocessing(E, inputs)
        # return energies
        return EnergyOutput(
            E=processed_energy["E"],
            raw_E=processed_energy["raw_E"],
            rescaled_E=processed_energy["rescaled_E"],
            molecular_ase=processed_energy["molecular_ase"],
        )


class SingleTopologyAlchemicalBaseNNPModel(BaseNeuralNetworkPotential):
    """
    Subclass for handling alchemical energy calculations.

    Methods
    -------
    forward(data: NNPInput) -> torch.Tensor
        Calculate the alchemical energy for a given input batch.
    """

    def forward(self, data: NNPInput):
        """
        Calculate the alchemical energy for a given input batch.

        Parameters
        ----------
        data : NNPInput
            Inputs containing atomic numbers ('atomic_numbers'), coordinates ('positions') and pairlist ('pairlist').
            - 'atomic_numbers': shape (nr_of_atoms_in_batch, *, *), 0 indicates non-interacting atoms that will be masked
            - 'total_charge' : shape (nr_of_atoms_in_batch)
            - (only for alchemical transformation) 'alchemical_atomic_number': shape (nr_of_atoms_in_batch)
            - (only for alchemical transformation) 'lamb': float
            - 'positions': shape (nr_of_atoms_in_batch, 3)

        Returns
        -------
        torch.Tensor
            Calculated energies; shape (n_systems,).
        """

        # emb = nn.Embedding(1,200)
        # lamb_emb = (1 - lamb) * emb(input['Z1']) + lamb * emb(input['Z2'])	def __init__():
        self._input_checks(data)
        raise NotImplementedError

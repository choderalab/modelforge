"""
This module contains the base classes for the neural network potentials.
"""

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, TypeVar, Union

import lightning as pl
import torch
from loguru import logger as log
from openff.units import unit
from modelforge.potential.neighbors import PairlistData

from modelforge.dataset.dataset import DatasetParameters
from modelforge.utils.prop import NNPInput
from modelforge.potential.parameters import (
    AimNet2Parameters,
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
)
from modelforge.train.parameters import RuntimeParameters, TrainingParameters

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


from typing import Callable, Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from modelforge.train.training import PotentialTrainer

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


from torch.nn import ModuleDict

from modelforge.potential.processing import (
    CoulombPotential,
    PerAtomCharge,
    PerAtomEnergy,
)


class PostProcessing(torch.nn.Module):

    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_atom_charge",
        "electrostatic_potential",
        "general_postprocessing_operation",
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
            assert all(
                prop in PostProcessing._SUPPORTED_PROPERTIES
                for prop in self._registered_properties
            )

        if "per_atom_charge" in properties_to_process:
            self.registered_chained_operations["per_atom_charge"] = PerAtomCharge(
                postprocessing_parameter["per_atom_charge"]
            )
            self._registered_properties.append("per_atom_charge")
            assert all(
                prop in PostProcessing._SUPPORTED_PROPERTIES
                for prop in self._registered_properties
            )

        if "electrostatic_potential" in properties_to_process:
            if (
                postprocessing_parameter["electrostatic_potential"][
                    "electrostatic_strategy"
                ]
                == "coulomb"
            ):

                self.registered_chained_operations["electrostatic_potential"] = (
                    CoulombPotential(
                        postprocessing_parameter["electrostatic_potential"][
                            "maximum_interaction_radius"
                        ],
                    )
                )
                self._registered_properties.append("electrostatic_potential")
                assert all(
                    prop in PostProcessing._SUPPORTED_PROPERTIES
                    for prop in self._registered_properties
                )
            else:
                raise NotImplementedError(
                    "Only Coulomb potential is supported for electrostatics."
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

    def _add_total_charge(
        self, core_output: Dict[str, torch.Tensor], input_data: NNPInput
    ):
        """
        Add the total charge to the core output.

        Parameters
        ----------
        core_output : Dict[str, torch.Tensor]
            The core network output.
        input_data : NNPInput
            The input data containing the atomic numbers and charges.

        Returns
        -------
        Dict[str, torch.Tensor]
            The core network output with the total charge added.
        """
        # Add the total charge to the core output
        core_output["per_system_total_charge"] = input_data.per_system_total_charge
        return core_output

    def _add_pairlist(
        self, core_output: Dict[str, torch.Tensor], pairlist_output: PairlistData
    ):
        """
        Add the pairlist to the core output.

        Parameters
        ----------
        core_output : Dict[str, torch.Tensor]
            The core network output.
        pairlist_output : PairlistData
            The pairlist output from the neighborlist.

        Returns
        -------
        Dict[str, torch.Tensor]
            The core network output with the pairlist added.
        """
        # Add the pairlist to the core output
        core_output["pair_indices"] = pairlist_output.pair_indices
        core_output["d_ij"] = pairlist_output.d_ij
        core_output["r_ij"] = pairlist_output.r_ij
        return core_output

    def _remove_pairlist(self, processed_output: Dict[str, torch.Tensor]):
        """
        Remove the pairlist from the core output.

        Parameters
        ----------
        processed_output : Dict[str, torch.Tensor]
            The postprocessed output.

        Returns
        -------
        Dict[str, torch.Tensor]
            The postprocessed output with the pairlist removed.
        """
        # Remove the pairlist from the core output
        del processed_output["pair_indices"]
        del processed_output["d_ij"]
        del processed_output["r_ij"]
        return processed_output

    def set_neighborlist_strategy(self, strategy: str, skin: 0.1):
        """
        Set the neighborlist strategy and skin for the neighborlist module.
        Note this cannot be called from a JIT-compiled model.

        Parameters
        ----------
        strategy : str
            The neighborlist strategy to use.
        skin : Optional[float], optional
            The skin for the Verlet neighborlist (default is None).
        """
        self.neighborlist.strategy = strategy
        self.neighborlist.skin = skin
        self.neighborlist.skin = skin
        self.neighborlist.half_skin = skin * 0.5
        self.neighborlist.cutoff_plus_skin = self.neighborlist.cutoff + skin

    def forward(self, input_data: NNPInput) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the potential model, computing energy and forces.

        Parameters
        ----------
        input_data : NNPInput
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
        core_output = self._add_total_charge(core_output, input_data)
        core_output = self._add_pairlist(core_output, pairlist_output)

        processed_output = self.postprocessing.forward(core_output)
        processed_output = self._remove_pairlist(processed_output)
        return processed_output

    def compute_core_network_output(
        self, input_data: NNPInput
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the core network output, including energy predictions.

        Parameters
        ----------
        input_data : NNPInput
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
    ):
        """
        Load the state dictionary into the infenerence or training model. Note
        that the Trainer class encapsulates the Training adapter (the PyTorch
        Lightning module), which contains the model. When saving a state dict
        from the Trainer class, you need to use `trainer.model.state_dict()` to
        save the model state dict. To load this in inference mode, you can use
        the `load_state_dict()` function in the Potential class. This function
        can load a state dictionary into the model, and removes keys that are
        specific to the training mode.

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

        # Prefix to remove from the keys
        prefix = "potential."
        # Prefixes of keys to exclude entirely
        excluded_prefixes = ["loss."]

        filtered_state_dict = {}
        prefixes_removed = set()

        for key, value in state_dict.items():
            # Exclude keys starting with any of the excluded prefixes
            if any(key.startswith(ex_prefix) for ex_prefix in excluded_prefixes):
                continue  # Skip this key entirely

            original_key = key  # Keep track of the original key

            # Remove the specified prefix from the key if it exists
            if key.startswith(prefix):
                key = key[len(prefix) :]
                prefixes_removed.add(prefix)

            # change legacy key names
            # neighborlist.calculate_distances_and_pairlist.cutoff -> neighborlist.cutoffs
            if key == "neighborlist.calculate_distances_and_pairlist.cutoff":
                key = "neighborlist.cutoff"

            filtered_state_dict[key] = value

        if prefixes_removed:
            log.debug(f"Removed prefixes: {prefixes_removed}")
        else:
            log.debug("No prefixes found. No modifications to keys in state loading.")

        super().load_state_dict(
            filtered_state_dict,
            strict=strict,
            assign=assign,
        )


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
    neighborlist_strategy: Optional[str] = None,
    verlet_neighborlist_skin: Optional[float] = 0.08,
) -> Potential:
    from modelforge.potential import _Implemented_NNPs
    from modelforge.potential.utils import remove_units_from_dataset_statistics
    from modelforge.utils.misc import seed_random_number

    log.debug(f"potential_seed {potential_seed}")
    if potential_seed is not None:
        log.info(f"Setting random seed to: {potential_seed}")
        seed_random_number(potential_seed)

    model_type = potential_parameter.potential_name
    core_network = _Implemented_NNPs.get_neural_network_class(model_type)(
        **potential_parameter.core_parameter.model_dump()
    )

    # set unique_pairs based on potential name
    only_unique_pairs = potential_parameter.only_unique_pairs

    assert (
        only_unique_pairs is False
        if potential_parameter.potential_name.lower() != "ani2x"
        else True
    )

    log.debug(f"Only unique pairs: {only_unique_pairs}")

    postprocessing = PostProcessing(
        postprocessing_parameter=potential_parameter.postprocessing_parameter.model_dump(),
        dataset_statistic=remove_units_from_dataset_statistics(dataset_statistic),
    )
    if use_training_mode_neighborlist:
        from modelforge.potential.neighbors import NeighborListForTraining

        neighborlist = NeighborListForTraining(
            cutoff=potential_parameter.core_parameter.maximum_interaction_radius,
            only_unique_pairs=only_unique_pairs,
        )
    else:
        from modelforge.potential.neighbors import OrthogonalDisplacementFunction

        displacement_function = OrthogonalDisplacementFunction()

        from modelforge.potential.neighbors import NeighborlistForInference

        neighborlist = NeighborlistForInference(
            cutoff=potential_parameter.core_parameter.maximum_interaction_radius,
            displacement_function=displacement_function,
            only_unique_pairs=only_unique_pairs,
        )
        # we can set the strategy here before passing this to the Potential
        # this can still be modified later using Potential.set_neighborlist_strategy before it has bit JITTED
        # after that, we can access variables directly at init level in the TorchForce wrapper
        neighborlist._set_strategy(neighborlist_strategy, skin=verlet_neighborlist_skin)

    potential = Potential(
        core_network,
        neighborlist,
        postprocessing,
        jit=jit,
        jit_neighborlist=False if use_training_mode_neighborlist else True,
    )
    potential.eval()
    return potential


from openff.units import unit


class NeuralNetworkPotentialFactory:

    @staticmethod
    def generate_potential(
        *,
        potential_parameter: T_NNP_Parameters,
        training_parameter: Optional[TrainingParameters] = None,
        dataset_parameter: Optional[DatasetParameters] = None,
        dataset_statistic: Dict[str, Dict[str, float]] = {
            "training_dataset_statistics": {
                "per_atom_energy_mean": unit.Quantity(0.0, unit.kilojoule_per_mole),
                "per_atom_energy_stddev": unit.Quantity(1.0, unit.kilojoule_per_mole),
            }
        },
        potential_seed: Optional[int] = None,
        use_training_mode_neighborlist: bool = False,
        simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
        jit: bool = True,
        inference_neighborlist_strategy: str = "verlet_nsq",
        verlet_neighborlist_skin: Optional[float] = 0.1,
    ) -> Union[Potential, JAXModel, pl.LightningModule]:
        """
        Create an instance of a neural network potential for inference.

        Parameters
        ----------
        potential_parameter : T_NNP_Parameters]
            Parameters specific to the neural network potential.
        training_parameter : Optional[TrainingParameters], optional
            Parameters for configuring training (default is None).
        dataset_parameter : Optional[DatasetParameters], optional
            Parameters for configuring the dataset (default is None).
        dataset_statistic : Dict[str, Dict[str, float]], optional
            Dataset statistics for normalization (default is provided).
        potential_seed : Optional[int], optional
            Seed for random number generation (default is None).
        use_training_mode_neighborlist : bool, optional
            Whether to use neighborlist during training mode (default is False).
        simulation_environment : Literal["PyTorch", "JAX"], optional
            Specify whether to use PyTorch or JAX as the simulation environment
            (default is "PyTorch").
        jit : bool, optional
            Whether to use JIT compilation (default is True).
        inference_neighborlist_strategy : Optional[str], optional
            Neighborlist strategy for inference (default is "verlet_nsq"). other option is "brute_nsq".
        verlet_neighborlist_skin : Optional[float], optional
            Skin for the Verlet neighborlist (default is 0.1, units nanometers).
        Returns
        -------
        Union[Potential, JAXModel]
            An instantiated neural network potential for training or inference.
        """

        log.debug(f"{training_parameter=}")
        log.debug(f"{potential_parameter=}")
        log.debug(f"{dataset_parameter=}")

        # obtain model for inference
        potential = setup_potential(
            potential_parameter=potential_parameter,
            dataset_statistic=dataset_statistic,
            use_training_mode_neighborlist=use_training_mode_neighborlist,
            potential_seed=potential_seed,
            jit=jit,
            neighborlist_strategy=inference_neighborlist_strategy,
            verlet_neighborlist_skin=verlet_neighborlist_skin,
        )
        # Disable gradients for model parameters
        for param in potential.parameters():
            param.requires_grad = False
        # Set model to eval
        potential.eval()

        if simulation_environment == "JAX":
            # register nnp_input as pytree
            from modelforge.utils.io import import_

            jax = import_("jax")
            from modelforge.jax import nnpinput_flatten, nnpinput_unflatten

            # registering NNPInput multiple times will result in a
            # ValueError
            try:
                jax.tree_util.register_pytree_node(
                    NNPInput,
                    nnpinput_flatten,
                    nnpinput_unflatten,
                )
            except ValueError:
                log.debug("NNPInput already registered as pytree")
                pass
            return PyTorch2JAXConverter().convert_to_jax_model(potential)
        else:
            return potential

    @staticmethod
    def generate_trainer(
        *,
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
    ) -> "PotentialTrainer":
        """
        Create a lightning trainer object to train the neural network potential.

        Parameters
        ----------
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
        Returns
        -------
        PotentialTrainer
            An instantiated neural network potential for training.
        """
        from modelforge.utils.misc import seed_random_number
        from modelforge.train.training import PotentialTrainer

        if potential_seed is not None:
            log.info(f"Setting random seed to: {potential_seed}")
            seed_random_number(potential_seed)

        log.debug(f"{training_parameter=}")
        log.debug(f"{dataset_parameter=}")

        trainer = PotentialTrainer(
            potential_parameter=potential_parameter,
            training_parameter=training_parameter,
            dataset_parameter=dataset_parameter,
            runtime_parameter=runtime_parameter,
            potential_seed=potential_seed,
            dataset_statistic=dataset_statistic,
            use_default_dataset_statistic=use_default_dataset_statistic,
        )
        return trainer


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


def load_inference_model_from_checkpoint(
    checkpoint_path: str,
) -> Union[Potential, JAXModel]:
    """
    Creates an inference model from a checkpoint file.
    It loads the checkpoint file, extracts the hyperparameters, and creates the model in inference mode.

    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint file.
    """
    import torch

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Extract hyperparameters
    hyperparams = checkpoint["hyper_parameters"]
    potential_parameter = hyperparams["potential_parameter"]
    dataset_statistic = hyperparams.get("dataset_statistic", None)
    potential_seed = hyperparams.get("potential_seed", None)

    # Create the model in inference mode
    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=potential_parameter,
        dataset_statistic=dataset_statistic,
        potential_seed=potential_seed,
    )

    # Load the state dict into the model
    potential.load_state_dict(checkpoint["state_dict"])

    # Return the model
    return potential

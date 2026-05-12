"""
This module contains the base classes for the neural network potentials.
"""

from typing import (
    Any,
    Dict,
    Callable,
    Literal,
    TYPE_CHECKING,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import lightning as pl
import torch
from loguru import logger as log
from openff.units import unit
from modelforge.utils.units import GlobalUnitSystem, chem_context
from modelforge.potential.neighbors import PairlistData, PairlistOutputs

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


if TYPE_CHECKING:
    from modelforge.train.training import PotentialTrainer

import numpy as np


class JAXModel:
    """
    A wrapper for calling a JAX-converted NNP potential.

    Forces are computed on the JAX side via ``jax.grad``, taking the negative
    gradient of ``per_system_energy`` with respect to ``positions``.  This
    keeps force computation fully JAX-native and XLA-compilable — the
    underlying PyTorch model is responsible only for returning energies.

    Attributes
    ----------
    jax_fn : Callable
        The JAX function produced by ``PyTorch2JAXConverter``.
    parameter : tuple
        JAX-side model parameters.
    buffer : tuple
        JAX-side model buffers.
    name : str
        Name of the wrapped model class.
    compute_forces : bool
        Whether to compute forces via ``jax.grad`` (default: True).
    """

    def __init__(
        self,
        jax_fn: Callable,
        parameter: tuple,
        buffer: tuple,
        name: str,
        # compute_forces: bool = True,
    ):
        self.jax_fn = jax_fn
        self.parameter = parameter
        self.buffer = buffer
        self.name = name
        # self.compute_forces = compute_forces

    def __call__(self, data: NNPInput):
        """Run the JAX model, computing energy and (optionally) forces.

        Forces are computed as ``-d(sum(per_system_energy))/d(positions)``
        using ``jax.grad`` — pure JAX differentiation, no PyTorch autograd.

        Parameters
        ----------
        data : NNPInput
            Input data whose fields are ``jax.Array`` objects.

        Returns
        -------
        dict
            Dictionary of ``jax.Array`` outputs.  Always contains
            ``"per_system_energy"``.
        """
        import jax

        out = self.jax_fn(self.parameter, self.buffer, data)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__} wrapping {self.name}"


from torch.nn import ModuleDict

from modelforge.potential.processing import (
    CoulombPotential,
    PerAtomCharge,
    PerAtomEnergy,
    ZBLPotential,
    DispersionPotential,
    SumPerSystemEnergy,
)


class PostProcessing(torch.nn.Module):
    _SUPPORTED_PROPERTIES = [
        "per_atom_energy",
        "per_atom_charge",
        "per_system_electrostatic_energy",
        "per_system_zbl_energy",
        "per_system_vdw_energy",
        "general_postprocessing_operation",
        "sum_per_system_energy",
    ]

    def __init__(
        self,
        postprocessing_parameter: Dict[str, Dict[str, Any]],
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

        # note, we need to post process certain properties in a specific order
        # for example, if we want to calculate the electrostatic potential, this will depend upon
        # the per_atom_charge, and thus we should perform per_atom_charge operations first.

        # currently, per_atom_energy needs to go last, as it has a functionality to add
        # the electrostatic energy to the per_system_energy that results from that operations
        # however, it may be better to create a general energy summation that takes in a list of
        # the energies to sum (all would be per_system energies), allowing the other energy operations to
        # be performed in any order.

        if "per_atom_charge" in properties_to_process:
            self.registered_chained_operations["per_atom_charge"] = PerAtomCharge(
                postprocessing_parameter["per_atom_charge"]
            )
            self._registered_properties.append("per_atom_charge")
            assert all(
                prop in PostProcessing._SUPPORTED_PROPERTIES
                for prop in self._registered_properties
            )

        if "per_system_electrostatic_energy" in properties_to_process:
            if (
                postprocessing_parameter["per_system_electrostatic_energy"][
                    "electrostatic_strategy"
                ]
                == "coulomb"
            ):
                self.registered_chained_operations[
                    "per_system_electrostatic_energy"
                ] = CoulombPotential(
                    postprocessing_parameter["per_system_electrostatic_energy"][
                        "maximum_interaction_radius"
                    ],
                )

                self._registered_properties.append("per_system_electrostatic_energy")
                assert all(
                    prop in PostProcessing._SUPPORTED_PROPERTIES
                    for prop in self._registered_properties
                )
            else:
                raise NotImplementedError(
                    "Only Coulomb potential is supported for electrostatics."
                )
        if "per_system_zbl_energy" in properties_to_process:
            if (
                postprocessing_parameter["per_system_zbl_energy"]["calculate_zbl"]
                == True
            ):
                self.registered_chained_operations["per_system_zbl_energy"] = (
                    ZBLPotential()
                )
                self._registered_properties.append("per_system_zbl_energy")
                assert all(
                    prop in PostProcessing._SUPPORTED_PROPERTIES
                    for prop in self._registered_properties
                )
        if "per_system_vdw_energy" in properties_to_process:
            length_conversion_factor = (
                (1.0 * GlobalUnitSystem.get_units("length")).to("bohr").magnitude
            )
            energy_conversion_factor = (
                (1.0 * unit.hartree).to(GlobalUnitSystem.get_units("energy"), "chem").m
            )

            self.registered_chained_operations["per_system_vdw_energy"] = (
                DispersionPotential(
                    cutoff=postprocessing_parameter["per_system_vdw_energy"][
                        "maximum_interaction_radius"
                    ],
                    length_conversion_factor=length_conversion_factor,
                    energy_conversion_factor=energy_conversion_factor,
                    parameter_set=postprocessing_parameter["per_system_vdw_energy"][
                        "parameter_set"
                    ],
                    d3_engine=postprocessing_parameter["per_system_vdw_energy"][
                        "d3_engine"
                    ],
                    d3_parameters_path=postprocessing_parameter[
                        "per_system_vdw_energy"
                    ]["d3_parameters_path"],
                )
            )
            self._registered_properties.append("per_system_vdw_energy")
            assert all(
                prop in PostProcessing._SUPPORTED_PROPERTIES
                for prop in self._registered_properties
            )
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

        if "sum_per_system_energy" in properties_to_process:
            contributions = postprocessing_parameter["sum_per_system_energy"][
                "contributions"
            ]
            # if an empty list is provided, we do not register the operation
            if len(contributions) > 0:
                self.registered_chained_operations["sum_per_system_energy"] = (
                    SumPerSystemEnergy(
                        contributions=postprocessing_parameter["sum_per_system_energy"][
                            "contributions"
                        ]
                    )
                )
                self._registered_properties.append("sum_per_system_energy")
                assert all(
                    prop in PostProcessing._SUPPORTED_PROPERTIES
                    for prop in self._registered_properties
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
        Methods
        -------
        forward(input_data: NNPInput) -> Dict[str, torch.Tensor]
            Forward pass for the potential model, computing energy and forces.
        compute_core_network_output(input_data: NNPInput) -> Dict[str, torch.Tensor]
            Compute the core network output, including energy predictions.
        load_state_dict(state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False)
            Load the state dictionary into the infenerence or training model.
        set_neighborlist_strategy(strategy: str, skin: float = 0.1)
            Set the neighborlist strategy and skin for the neighborlist module.
        forward_for_jit_inference(atomic_numbers: torch.Tensor, positions: torch.Tensor, atomic_subsystem_indices: torch.Tensor, per_system_total_charge: torch.Tensor, pair_list: torch.Tensor, per_atom_partial_charge: torch.Tensor, box_vectors: torch.Tensor, is_periodic: torch.Tensor) -> Dict[str, torch.Tensor]
            Forward pass for the potential model, computing energy and forces that accepts individual tensors rather than NNPInput class, necessary for JIT compiled model.
        forward(input_data: NNPInput) -> Dict[str, torch.Tensor]
            Forward pass for the potential model, computing energy and forces.
        """

        super().__init__()

        self.core_network = torch.jit.script(core_network) if jit else core_network
        self.neighborlist = (
            torch.jit.script(neighborlist) if jit_neighborlist else neighborlist
        )
        # note cannot jit compile the dispersion interactions as tad-dftd3 is not compatible with torchscript
        if "per_system_vdw_energy" in postprocessing._registered_properties:
            # double check if nvalchemiops works with JIT
            log.warning(
                "JIT compiling the postprocessing module with vdw interactions will not work if using tad-dftd3."
            )
            log.warning("tad-dftd3 packaged is not compatible with torchscript.")
            log.warning(
                "Use nvalchemiops as the DFTD3 engine or disable JIT compilation by setting jit=False."
            )

        # self.postprocessing = (
        #     torch.jit.script(postprocessing) if jit else postprocessing
        # )
        self.postprocessing = postprocessing

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

    def _add_positions(
        self, core_output: Dict[str, torch.Tensor], positions: torch.Tensor
    ):
        """
        Add the positions to the core output.

        Parameters
        ----------
        core_output : Dict[str, torch.Tensor]
            The core network output.
        positions : torch.Tensor
            The tensor containing atomic positions.

        Returns
        -------
        Dict[str, torch.Tensor]
            The core network output with the positions added.
        """
        # Add the positions to the core output
        core_output["positions"] = positions
        return core_output

    def _remove_positions(self, core_output: Dict[str, torch.Tensor]):
        """
        Remove the positions from the core output.

        Parameters
        ----------
        core_output : Dict[str, torch.Tensor]
            The core network output.

        Returns
        -------
        Dict[str, torch.Tensor]
            The core network output with the positions removed.
        """
        # Remove the positions from the core output
        if "positions" in core_output:
            del core_output["positions"]
        return core_output

    def _add_pairlist(
        self,
        core_output: Dict[str, torch.Tensor],
        pairlist_output: PairlistOutputs,
    ):
        """
        Add the pairlist to the core output.

        Parameters
        ----------
        core_output : Dict[str, torch.Tensor]
            The core network output.
        pairlist_output : Dict[str, PairlistData]
            The pairlist output from the neighborlist.
            The keys in the dictionary correspond to different cutoffs: local_cutoff, vdw_cutoff, electrostatic_cutoff
            Only local_cutoff is guaranteed to be present, while vdw_cutoff and electrostatic_cutoff are optional
            depending on the postprocessing configuration.
        Returns
        -------
        Dict[str, torch.Tensor]
            The core network output with the pairlist added.
        """
        # Add the pairlist to the core output
        # looping over all the cutoffs that have been defined in the pairlist_output
        core_output["local_pair_indices"] = pairlist_output.local_cutoff.pair_indices
        core_output["local_d_ij"] = pairlist_output.local_cutoff.d_ij
        core_output["local_r_ij"] = pairlist_output.local_cutoff.r_ij

        if (
            "per_system_vdw_energy" in self.postprocessing._registered_properties
        ):  # FIXME: no need when running tad-dftd3
            core_output["vdw_pair_indices"] = pairlist_output.vdw_cutoff.pair_indices
            core_output["vdw_d_ij"] = pairlist_output.vdw_cutoff.d_ij
            core_output["vdw_r_ij"] = pairlist_output.vdw_cutoff.r_ij

        if (
            "per_system_electrostatic_energy"
            in self.postprocessing._registered_properties
        ):
            core_output["electrostatic_pair_indices"] = (
                pairlist_output.electrostatic_cutoff.pair_indices
            )
            core_output["electrostatic_d_ij"] = (
                pairlist_output.electrostatic_cutoff.d_ij
            )
            core_output["electrostatic_r_ij]"] = (
                pairlist_output.electrostatic_cutoff.r_ij
            )

        return core_output

    def _remove_pairlist(
        self,
        processed_output: Dict[str, torch.Tensor],
    ):
        """
        Remove the pairlist from the core output.

        Parameters
        ----------
        processed_output : Dict[str, torch.Tensor]
            The postprocessed output.
        pairlist_output : Dict[str, PairlistData]
            The pairlist output from the neighborlist.
            The keys in the dictionary correspond to different cutoffs: local_cutoff, vdw_cutoff, electrostatic_cutoff
            Only local_cutoff is guaranteed to be present, while vdw_cutoff and electrostatic_cutoff are optional
            depending on the postprocessing configuration.
        Returns
        -------
        Dict[str, torch.Tensor]
            The postprocessed output with the pairlist removed.
        """
        # Remove the pairlist from the core output
        prefixes = ["local", "vdw", "electrostatic"]
        suffixes = ["pair_indices", "d_ij", "r_ij"]
        for prefix in prefixes:
            for suffix in suffixes:
                key = f"{prefix}_{suffix}"
                if key in processed_output:
                    del processed_output[key]

        return processed_output

    @torch.jit.export
    def set_neighborlist_strategy(self, strategy: str, skin: float = 0.1):
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
        self.neighborlist._set_strategy(strategy, skin=skin)

    # This function accepts the NNPInput data as individual tensors
    # as opposed to a single NNPInput object.
    # This is necessary when using a JIT compiled version of the model
    @torch.jit.export
    def forward_for_jit_inference(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        per_system_total_charge: torch.Tensor,
        pair_list: torch.Tensor,
        per_atom_partial_charge: torch.Tensor,
        per_system_spin_state: torch.Tensor,
        box_vectors: torch.Tensor,
        is_periodic: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        input_data = NNPInput(
            atomic_numbers=atomic_numbers,
            positions=positions,
            atomic_subsystem_indices=atomic_subsystem_indices,
            per_system_total_charge=per_system_total_charge,
            pair_list=pair_list,
            per_atom_partial_charge=per_atom_partial_charge,
            per_system_spin_state=per_system_spin_state,
            box_vectors=box_vectors,
            is_periodic=is_periodic,
        )

        return self.forward(input_data)

    def forward(self, input_data: NNPInput) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the potential model, computing energy.

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
        core_output = self.core_network.forward(
            input_data, pairlist_output.local_cutoff
        )

        # Step 3: Apply postprocessing using PostProcessing
        core_output = self._add_total_charge(core_output, input_data)
        core_output = self._add_pairlist(core_output, pairlist_output)

        # we need positions for both vdw energy and force computation
        if "per_system_vdw_energy" in self.postprocessing._registered_properties:
            core_output = self._add_positions(core_output, input_data.positions)

        processed_output = self.postprocessing.forward(core_output)
        processed_output = self._remove_pairlist(processed_output)
        if "per_system_vdw_energy" in self.postprocessing._registered_properties:
            processed_output = self._remove_positions(processed_output)

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
        return self.core_network.forward(input_data, pairlist_output.local_cutoff)

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
        legacy : bool, optional
            Earlier version of the potential model did not include only_unique_pairs in the
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

            if key == "neighborlist.only_unique_pairs":
                key = "neighborlist.local_only_unique_pairs"
                filtered_state_dict["neighborlist.vdw_only_unique_pairs"] = (
                    torch.tensor([False])
                )
                filtered_state_dict["neighborlist.electrostatic_only_unique_pairs"] = (
                    torch.tensor([True])
                )

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
            "per_atom_energy_mean": unit.Quantity(
                0.0, GlobalUnitSystem.get_units("energy")
            ),
            "per_atom_energy_stddev": unit.Quantity(
                1.0, GlobalUnitSystem.get_units("energy")
            ),
        }
    },
    use_training_mode_neighborlist: bool = False,
    potential_seed: Optional[int] = None,
    jit: bool = False,
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
    post_processing_dict = potential_parameter.postprocessing_parameter.model_dump()

    postprocessing = PostProcessing(
        postprocessing_parameter=post_processing_dict,
        dataset_statistic=remove_units_from_dataset_statistics(dataset_statistic),
    )
    # we will define the possible cutoffs.
    # - local_cutoff is the maximum interaction radius for the NNP core network (i.e., the local interaction radius)
    # this is always required.
    # - vdw_cutoff is the cutoff for the van der Waals interactions, which is only required in the vdw interactions are
    # included as a post-processing step.
    # - electrostatic_cutoff is the cutoff for the electrostatic interactions, which is only required in the electrostatic
    # interactions are included as a post-processing step.
    #
    # note zbl potential does not require a unique cutoff definition; it will use local cutoff and then calculates
    # the zbl potential based on the radii of the two atoms in the pair.

    local_cutoff = potential_parameter.core_parameter.maximum_interaction_radius
    electrostatic_cutoff = -1
    vdw_cutoff = -1
    use_vdw_cutoff = False
    use_electrostatic_cutoff = False

    if "per_system_electrostatic_energy" in postprocessing._registered_properties:
        electrostatic_cutoff = (
            potential_parameter.postprocessing_parameter.per_system_electrostatic_energy.maximum_interaction_radius
        )
        use_electrostatic_cutoff = True

    if "per_system_vdw_energy" in postprocessing._registered_properties:
        vdw_cutoff = (
            potential_parameter.postprocessing_parameter.per_system_vdw_energy.maximum_interaction_radius
        )
        use_vdw_cutoff = True
    log.debug(
        f"Cutoffs: local_cutoff={local_cutoff}, vdw_cutoff={vdw_cutoff}, electrostatic_cutoff={electrostatic_cutoff}"
    )
    if use_training_mode_neighborlist:
        from modelforge.potential.neighbors import NeighborListForTraining

        neighborlist = NeighborListForTraining(
            local_cutoff=local_cutoff,
            vdw_cutoff=vdw_cutoff,
            electrostatic_cutoff=electrostatic_cutoff,
            local_only_unique_pairs=only_unique_pairs,
            use_vdw_cutoff=use_vdw_cutoff,
            use_electrostatic_cutoff=use_electrostatic_cutoff,
        )
    else:
        from modelforge.potential.neighbors import OrthogonalDisplacementFunction

        displacement_function = OrthogonalDisplacementFunction()

        from modelforge.potential.neighbors import NeighborlistForInference

        neighborlist = NeighborlistForInference(
            local_cutoff=local_cutoff,
            vdw_cutoff=vdw_cutoff,
            electrostatic_cutoff=electrostatic_cutoff,
            displacement_function=displacement_function,
            local_only_unique_pairs=only_unique_pairs,
            use_vdw_cutoff=use_vdw_cutoff,
            use_electrostatic_cutoff=use_electrostatic_cutoff,
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
        jit_neighborlist=False,  # if use_training_mode_neighborlist else True, # will be removing jit support
    )
    potential.eval()
    return potential


class NeuralNetworkPotentialFactory:
    @staticmethod
    def generate_potential(
        *,
        potential_parameter: T_NNP_Parameters,
        training_parameter: Optional[TrainingParameters] = None,
        dataset_parameter: Optional[DatasetParameters] = None,
        dataset_statistic: Dict[str, Dict[str, float]] = {
            "training_dataset_statistics": {
                "per_atom_energy_mean": unit.Quantity(
                    0.0, GlobalUnitSystem.get_units("energy")
                ),
                "per_atom_energy_stddev": unit.Quantity(
                    1.0, GlobalUnitSystem.get_units("energy")
                ),
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
            Skin for the Verlet neighborlist (default is 0.1, internal length units which are by default nanometers).
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
    def load_from_wandb(
        *,
        run_path: str,
        version: str,
        local_cache_dir: str = "./",
        only_unique_pairs: Optional[bool] = None,
        old_config_only_local_cutoff: Optional[bool] = False,
        jit: bool = False,
    ) -> Union[Potential, JAXModel]:
        """
        Load a neural network potential from a Weights & Biases run.

        Parameters
        ----------
        run_path : str
            The path to the Weights & Biases run.
        version : str
            The version of the run to load.
        local_cache_dir : str, optional
            The local cache directory for downloading the model (default is "./"),
        only_unique_pairs : Optional[bool], optional
            For models trained prior to PR #299 in modelforge, this parameter is required to be able to read the model.
            This value should be True for the ANI models, False for most other models.
        long_config_only_local_cutoff : Optional[bool], optional
            For older models, this parameter is required to be able to read the model. Replaces neighborlist.cutoff with neighborlist.local_cutoff
            and other associated parameters for vdw and electrostatic cutoffs.
        jit : bool, optional
            Whether to use JIT compilation (default is False).
        Returns
        -------
        Union[Potential, JAXModel]
            An instantiated neural network potential for training or inference.
        """
        import wandb

        run = wandb.init()
        artifact_path = f"{run_path}:{version}"
        artifact = run.use_artifact(artifact_path)
        artifact_dir = artifact.download(root=local_cache_dir)
        checkpoint_file = f"{artifact_dir}/model.ckpt"
        potential = load_inference_model_from_checkpoint(
            checkpoint_path=checkpoint_file,
            only_unique_pairs=only_unique_pairs,
            old_config_only_local_cutoff=old_config_only_local_cutoff,
            jit=jit,
        )

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
                "per_atom_energy_mean": unit.Quantity(
                    0.0, GlobalUnitSystem.get_units("energy")
                ),
                "per_atom_energy_stddev": unit.Quantity(
                    1.0, GlobalUnitSystem.get_units("energy")
                ),
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
        log.debug(f"{potential_parameter=}")
        log.debug(f"{runtime_parameter=}")
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
    Wraps a PyTorch neural-network potential in a JAX-callable function.

    Uses DLPack for zero-copy tensor sharing between PyTorch and JAX.
    Compatible with PyTorch >= 2.0 (uses ``torch.func`` instead of the
    deprecated standalone ``functorch`` package) and JAX >= 0.4.1 (uses the
    ``__dlpack__`` protocol instead of the old capsule-based DLPack API).

    This also correctly handles NNP potentials that call ``torch.autograd.grad``
    internally (e.g. to compute forces from energies): the conversion layer
    re-attaches a gradient tape to *positions* before every forward pass so
    that ``torch.autograd.grad`` succeeds inside the model.
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
            A JAX model containing the converted neural network function,
            parameters, and buffers.
        """
        jax_fn, params, buffers = self._convert_pytnn_to_jax(nnp_instance)
        return JAXModel(jax_fn, params, buffers, nnp_instance.__class__.__name__)

    @staticmethod
    def _convert_pytnn_to_jax(
        nnp_instance: Potential,
    ) -> Tuple[Callable, tuple, tuple]:
        """Internal method to convert PyTorch neural network parameters and
        buffers to JAX format.

        Parameters
        ----------
        nnp_instance : Any
            The PyTorch neural network instance.

        Returns
        -------
        Tuple[Callable, Any, Any]
            A tuple containing the JAX function, parameters, and buffers.
        """
        import torch
        import jax
        from modelforge.utils.prop import NNPInput

        from modelforge.jax import torch_to_jax, jax_to_torch

        # Build dtype-aware param/buffer converters.
        # We record which parameter/buffer names were originally bool so we
        # can cast them back correctly in jax_to_torch.
        _bool_param_names: set = set()
        _bool_buffer_names: set = set()
        for name, p in nnp_instance.named_parameters():
            if p.dtype == torch.bool:
                _bool_param_names.add(name)
        for name, b in nnp_instance.named_buffers():
            if b.dtype == torch.bool:
                _bool_buffer_names.add(name)

        def _params_to_torch(jax_params_tuple):
            """Convert jax param tuple back to torch, restoring bool dtypes."""
            param_names = list(dict(nnp_instance.named_parameters()).keys())
            result = []
            for name, arr in zip(param_names, jax_params_tuple):
                result.append(jax_to_torch(arr, as_bool=(name in _bool_param_names)))
            return tuple(result)

        def _buffers_to_torch(jax_buffers_tuple):
            """Convert jax buffer tuple back to torch, restoring bool dtypes."""
            buffer_names = list(dict(nnp_instance.named_buffers()).keys())
            result = []
            for name, arr in zip(buffer_names, jax_buffers_tuple):
                result.append(jax_to_torch(arr, as_bool=(name in _bool_buffer_names)))
            return tuple(result)

        def _jax_data_to_nnpinput(data) -> NNPInput:
            """Reconstruct a typed NNPInput from JAX-side data.

            NNPInput is registered as a JAX pytree with a children/aux_data
            split (defined in modelforge/jax.py) to avoid passing
            non-DLPack-compatible dtypes (bool, int indices) through JAX's
            tracing machinery.

            children  (JAX-traced jax.Arrays, converted via DLPack):
                0  positions
                1  per_system_total_charge
                2  box_vectors
                3  per_system_spin_state
                4  per_atom_partial_charge  (may be None)

            aux_data  (static torch.Tensors / None, never DLPack-converted):
                0  atomic_numbers
                1  atomic_subsystem_indices
                2  is_periodic              (int8-encoded bool)
                3  pair_list                (may be None)

            When JAX passes data back across the custom_vjp boundary it may
            arrive as an NNPInput (forward call) or a (children, aux_data)
            tuple-of-tuples (after pytree flattening).  Both cases are handled.
            """

            from modelforge.jax import nnpinput_unflatten, convert_NNPInput_jax_to_torch

            if isinstance(data, NNPInput):
                # Fields are a mix: float children are jax.Arrays,
                # aux_data fields are still torch.Tensors / None.

                return convert_NNPInput_jax_to_torch(data)

            else:
                #

                # JAX has flattened the pytree to (children_tuple, aux_tuple).
                # Unpack by the canonical index order above.
                children, aux_data = data
                children = list(children)
                aux_data = list(aux_data)

                return nnpinput_unflatten(aux_data=aux_data, children=children)

        # Compatibility shim: functorch (standalone) -> torch.func (>= 2.0)
        try:
            from torch.func import functional_call as _functional_call

            def _make_functional_with_buffers(model):
                """Mimic functorch.make_functional_with_buffers."""
                param_names = list(dict(model.named_parameters()).keys())
                buffer_names = list(dict(model.named_buffers()).keys())
                param_values = tuple(dict(model.named_parameters()).values())
                buffer_values = tuple(dict(model.named_buffers()).values())

                def model_fn(params_t, buffers_t, *args, **kwargs):
                    p = dict(zip(param_names, params_t))
                    b = dict(zip(buffer_names, buffers_t))
                    return _functional_call(model, {**p, **b}, args, kwargs)

                return model_fn, param_values, buffer_values

            def _torch_vjp(fn, *primals):
                from torch.func import vjp

                return vjp(fn, *primals)

        except ImportError:
            # Fallback for standalone functorch / PyTorch < 2.0
            import functorch as _functorch  # type: ignore[import]

            _make_functional_with_buffers = _functorch.make_functional_with_buffers

            def _torch_vjp(fn, *primals):
                return _functorch.vjp(fn, *primals)

        # Build a stateless functional version of the model

        model_fn, model_params, model_buffers = _make_functional_with_buffers(
            nnp_instance
        )

        # Convert params / buffers to JAX arrays once (stored on JAX side).
        # torch_to_jax handles bool to int8 cast and empty to None.
        jax_params = jax.tree_util.tree_map(torch_to_jax, model_params)
        jax_buffers = jax.tree_util.tree_map(torch_to_jax, model_buffers)

        # ------------------------------------------------------------------
        # Helper: run model with positions.requires_grad=True.
        #
        # NNP models call torch.autograd.grad(energy, positions) inside their
        # forward pass to compute forces.  DLPack-converted tensors arrive
        # detached, so we re-attach a gradient tape to positions and wrap the
        # call in torch.enable_grad().
        # ------------------------------------------------------------------

        def _run_model(torch_params, torch_buffers, nnp_input: NNPInput):
            # is needed for the forward pass. JAXModel.__call__ computes
            # forces via jax.grad on the JAX side.
            with torch.no_grad():
                return model_fn(torch_params, torch_buffers, nnp_input)

        # ------------------------------------------------------------------
        # JAX-visible apply function with custom VJP
        # ------------------------------------------------------------------

        @jax.custom_vjp
        def apply(params, buffers, data):
            torch_params = _params_to_torch(params)
            torch_buffers = _buffers_to_torch(buffers)
            # Reconstruct a typed NNPInput so neighbour-list forward() and
            # other sub-modules receive the correct object, not a raw tuple.
            nnp_input = _jax_data_to_nnpinput(data)

            out = _run_model(torch_params, torch_buffers, nnp_input)
            return jax.tree_util.tree_map(torch_to_jax, out)

        def apply_fwd(params, buffers, data):
            out = apply(params, buffers, data)
            return out, (params, buffers, data)

        def apply_bwd(res, g):
            params, buffers, data = res

            torch_params = _params_to_torch(params)
            torch_buffers = _buffers_to_torch(buffers)
            nnp_input = _jax_data_to_nnpinput(data)

            # Only float params can accumulate gradients.
            float_params = [
                t.detach().requires_grad_(True)
                for t in torch_params
                if t is not None and t.is_floating_point()
            ]
            float_param_indices = [
                i
                for i, t in enumerate(torch_params)
                if t is not None and t.is_floating_point()
            ]
            # Positions gradient is what JAXModel needs for forces.
            positions = nnp_input.positions.detach().requires_grad_(True)
            nnp_input.positions = positions

            # Rebuild the full param tuple with grad-enabled float params.
            full_params = list(torch_params)
            for slot, fp in zip(float_param_indices, float_params):
                full_params[slot] = fp

            with torch.enable_grad():
                out = model_fn(tuple(full_params), torch_buffers, nnp_input)

            # Collect the float outputs — only these can have grad_fn.dlpack
            float_out_tensors = [
                v
                for v in out.values()
                if isinstance(v, torch.Tensor)
                and v.is_floating_point()
                and v.requires_grad
            ]

            if not float_out_tensors:
                # No differentiable outputs — return zero grads.
                jax_grad_params = jax.tree_util.tree_map(
                    lambda t: (
                        torch_to_jax(torch.zeros_like(t)) if t is not None else None
                    ),
                    tuple(torch_params),
                )
                jax_grad_buffers = jax.tree_util.tree_map(
                    lambda t: (
                        torch_to_jax(torch.zeros_like(t)) if t is not None else None
                    ),
                    torch_buffers,
                )
                jax_grad_positions = torch_to_jax(torch.zeros_like(positions))
            else:
                # Build the upstream gradient vector from JAX's g dict.
                # Entries with void dtype (float0) or None have no gradient.
                def _safe_g(array):
                    if array is None:
                        return None
                    if hasattr(array, "dtype") and array.dtype.kind == "V":
                        return None
                    return jax_to_torch(array)

                # Match upstream grads to the float outputs in dict-order.
                float_out_keys = [
                    k
                    for k, v in out.items()
                    if isinstance(v, torch.Tensor)
                    and v.is_floating_point()
                    and v.requires_grad
                ]
                upstream = []
                for k in float_out_keys:
                    raw = _safe_g(g.get(k)) if isinstance(g, dict) else None
                    if raw is None:
                        raw = torch.zeros_like(out[k])
                    upstream.append(raw)

                # Differentiate w.r.t. float params and positions.
                wrt = float_params + [positions]
                grads = torch.autograd.grad(
                    outputs=float_out_tensors,
                    inputs=wrt,
                    grad_outputs=upstream,
                    allow_unused=True,
                    retain_graph=False,
                )
                grad_float_params = grads[: len(float_params)]
                grad_positions = grads[len(float_params)]

                # Rebuild full-length grad_params (zeros for non-float slots).
                grad_params_full = [
                    torch.zeros_like(t) if t is not None else None for t in torch_params
                ]
                for slot, gp in zip(float_param_indices, grad_float_params):
                    grad_params_full[slot] = (
                        gp if gp is not None else torch.zeros_like(torch_params[slot])
                    )

                grad_buffers_full = tuple(
                    torch.zeros_like(t) if t is not None else None
                    for t in torch_buffers
                )

                jax_grad_params = jax.tree_util.tree_map(
                    torch_to_jax, tuple(grad_params_full)
                )
                jax_grad_buffers = jax.tree_util.tree_map(
                    torch_to_jax, grad_buffers_full
                )
                jax_grad_positions = torch_to_jax(
                    grad_positions
                    if grad_positions is not None
                    else torch.zeros_like(positions)
                )

            # Return grads matching apply's three arguments: (params, buffers, data).
            # For data, reconstruct an NNPInput-shaped pytree: only positions
            # carries a real gradient; other float fields get zeros; aux fields
            # (atomic_numbers, is_periodic, pair_list) pass through unchanged.

            jax_grad_data = NNPInput(
                atomic_numbers=data.atomic_numbers,
                positions=jax_grad_positions,
                atomic_subsystem_indices=data.atomic_subsystem_indices,
                per_system_total_charge=jax.numpy.zeros_like(
                    data.per_system_total_charge, dtype=jax_grad_positions.dtype
                ),
                box_vectors=jax.numpy.zeros_like(
                    data.box_vectors, dtype=jax_grad_positions.dtype
                ),
                per_system_spin_state=jax.numpy.zeros_like(
                    data.per_system_spin_state, dtype=jax_grad_positions.dtype
                ),
                is_periodic=torch_to_jax(data.is_periodic),
                pair_list=data.pair_list,
                per_atom_partial_charge=(
                    jax.numpy.zeros_like(
                        data.per_atom_partial_charge, dtype=jax_grad_positions.dtype
                    )
                    if data.per_atom_partial_charge is not None
                    else None
                ),
            )

            return (jax_grad_params, jax_grad_buffers, jax_grad_data)

        apply.defvjp(apply_fwd, apply_bwd)

        return apply, jax_params, jax_buffers


def load_inference_model_from_checkpoint(
    checkpoint_path: str,
    old_config_only_local_cutoff: Optional[bool] = False,
    only_unique_pairs: Optional[bool] = None,
    jit: bool = True,
) -> Union[Potential, JAXModel]:
    """
    Creates an inference model from a checkpoint file.
    It loads the checkpoint file, extracts the hyperparameters, and creates the model in inference mode.

    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint file.
    old_config_only_local_cutoff: Optional[bool], optional
        Old checkpoint files may have been only saved with neighborlist.cutoff .
        this will update the checkpoint to use the new neighborlist.local_cutoff, neighborlist.vdw_cutoff, and neighborlist.electrostatic_cutoff
        keys and set use_vdw_cutoff and use_electrostatic_cutoff to False, as required in the newly revised neighborlist module.
    only_unique_pairs : Optional[bool], optional
        If defined, this will set the only_unique_pairs key in the neighborlist module. This is only needed
        for models trained prior to PR #299 in modelforge. (default is None).
        In the case of ANI models, this should be set to True. Typically False for other models
    jit : bool, optional
        Whether to use JIT compilation for the model internals (default is True).
    """

    # Load the checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )

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
        jit=jit,
    )

    if only_unique_pairs is not None:
        checkpoint["state_dict"]["potential.neighborlist.local_only_unique_pairs"] = (
            torch.Tensor([only_unique_pairs])
        )
        checkpoint["state_dict"]["potential.neighborlist.vdw_only_unique_pairs"] = (
            torch.Tensor([False])
        )
        checkpoint["state_dict"][
            "potential.neighborlist.electrostatic_only_unique_pairs"
        ] = torch.Tensor([True])
    if old_config_only_local_cutoff:
        cutoff = checkpoint["state_dict"].get("potential.neighborlist.cutoff")
        # remove the old key
        del checkpoint["state_dict"]["potential.neighborlist.cutoff"]

        checkpoint["state_dict"]["potential.neighborlist.local_cutoff"] = cutoff
        checkpoint["state_dict"]["potential.neighborlist.vdw_cutoff"] = torch.Tensor(
            [-1.0]
        )
        checkpoint["state_dict"]["potential.neighborlist.electrostatic_cutoff"] = (
            torch.Tensor([-1.0])
        )
        checkpoint["state_dict"]["potential.neighborlist.use_vdw_cutoff"] = (
            torch.Tensor([False])
        )
        checkpoint["state_dict"]["potential.neighborlist.use_electrostatic_cutoff"] = (
            torch.Tensor([False])
        )
        checkpoint["state_dict"]["potential.neighborlist.largest_cutoff"] = cutoff

    # Load the state dict into the model
    potential.load_state_dict(checkpoint["state_dict"])

    # Return the model
    return potential

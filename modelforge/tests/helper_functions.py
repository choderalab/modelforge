from typing import Optional, Literal


def _core_provides_property(config, property_name: str) -> bool:
    """True if the potential's core computes `property_name` itself, so it must
    not be added to predicted_properties (the core raises if it is)."""
    from modelforge.potential import _Implemented_NNPs

    core_class = _Implemented_NNPs.get_neural_network_class(
        config["potential"].potential_name
    )
    return property_name in getattr(core_class, "core_output_properties", ())


def _add_to_predicted_properties(config, property_name: str, dim: int = 1):
    """Add a readout head for `property_name` unless the core already provides it."""
    p_config = config["potential"].core_parameter
    if property_name in p_config.predicted_properties or _core_provides_property(
        config, property_name
    ):
        return config
    p_config.predicted_properties.append(property_name)
    p_config.predicted_dim.append(dim)
    return config


def _add_per_atom_charge_to_predicted_properties(config):
    return _add_to_predicted_properties(config, "per_atom_charge")


def _add_per_atom_charge_to_properties_to_process(config):
    config["potential"].postprocessing_parameter.properties_to_process.append(
        "per_atom_charge"
    )
    from modelforge.potential.parameters import PerAtomCharge

    config["potential"].postprocessing_parameter.per_atom_charge = PerAtomCharge(
        conserve=True, conserve_strategy="default"
    )

    return config


def _add_electrostatic_to_predicted_properties(config):
    from modelforge.potential.parameters import ElectrostaticPotential
    from openff.units import unit

    config["potential"].postprocessing_parameter.properties_to_process.append(
        "per_system_electrostatic_energy"
    )
    config["potential"].postprocessing_parameter.per_system_electrostatic_energy = (
        ElectrostaticPotential(
            electrostatic_strategy="coulomb",
            maximum_interaction_radius=10.0 * unit.angstrom,
        )
    )

    return config


def setup_potential_for_test(
    potential_name: str,
    use: str,
    use_default_dataset_statistic: bool = True,
    use_training_mode_neighborlist: bool = True,
    jit: bool = False,
    potential_seed: Optional[int] = None,
    simulation_environment: Literal["PyTorch", "JAX"] = "PyTorch",
    local_cache_dir: Optional[str] = None,
    dataset_cache_dir: Optional[str] = None,
):
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.utils.misc import load_configs_into_pydantic_models

    if simulation_environment == "JAX":
        assert use == "inference", "JAX only supports inference mode"

    # read default parameters
    config = load_configs_into_pydantic_models(potential_name, "qm9")
    # override defaults to match reference implementation in spk

    if local_cache_dir is not None:
        config["runtime"].local_cache_dir = local_cache_dir

    if use == "training":
        trainer = NeuralNetworkPotentialFactory.generate_trainer(
            potential_parameter=config["potential"],
            runtime_parameter=config["runtime"],
            training_parameter=config["training"],
            dataset_parameter=config["dataset"],
            potential_seed=potential_seed,
            use_default_dataset_statistic=use_default_dataset_statistic,
        )
        potential = trainer.lightning_module.potential
    else:
        potential = NeuralNetworkPotentialFactory.generate_potential(
            potential_parameter=config["potential"],
            training_parameter=config["training"],
            dataset_parameter=config["dataset"],
            potential_seed=potential_seed,
            simulation_environment=simulation_environment,
            use_training_mode_neighborlist=use_training_mode_neighborlist,
            jit=jit,
        )

    return potential

from typing import Optional


def _add_per_atom_charge_to_predicted_properties(config):
    config["potential"].core_parameter.predicted_properties.append("per_atom_charge")
    config["potential"].core_parameter.predicted_dim.append(1)
    return config


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
        "electrostatic_potential"
    )
    config["potential"].postprocessing_parameter.electrostatic_potential = (
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
    simulation_environment="PyTorch",
    local_cache_dir: Optional[str] = None,
):
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.tests.test_potentials import load_configs_into_pydantic_models

    if simulation_environment == "JAX":
        assert use == "inference", "JAX only supports inference mode"

    # read default parameters
    config = load_configs_into_pydantic_models(potential_name, "qm9")
    # override defaults to match reference implementation in spk

    if local_cache_dir is not None:
        config["runtime"].local_cache_dir = local_cache_dir

    model = NeuralNetworkPotentialFactory.generate_potential(
        use=use,
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
        potential_seed=potential_seed,
        simulation_environment=simulation_environment,
        use_training_mode_neighborlist=use_training_mode_neighborlist,
        use_default_dataset_statistic=use_default_dataset_statistic,
        jit=jit,
    )

    if use == "training":
        potential = model.lightning_module.potential
    else:
        potential = model
    return potential

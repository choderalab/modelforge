from typing import Optional


def setup_potential(
    potential_name: str,
    use: str,
    use_default_dataset_statistic: bool = True,
    use_training_mode_neighborlist: bool = True,
    jit: bool = False,
    potential_seed: Optional[int] = None,
    simulation_environmen="PyTorch",
    only_unique_pairs: bool = False,
):
    from modelforge.potential import NeuralNetworkPotentialFactory
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    if simulation_environmen == "JAX":
        assert use == "inference", "JAX only supports inference mode"

    # read default parameters
    config = load_configs_into_pydantic_models(potential_name, "qm9")
    # override defaults to match reference implementation in spk

    model = NeuralNetworkPotentialFactory.generate_potential(
        use=use,
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
        potential_seed=potential_seed,
        simulation_environment=simulation_environmen,
        use_training_mode_neighborlist=use_training_mode_neighborlist,
        use_default_dataset_statistic=use_default_dataset_statistic,
        jit=jit,
        only_unique_pairs=only_unique_pairs
    )

    if use == "training":
        model = model.model.potential
    return model

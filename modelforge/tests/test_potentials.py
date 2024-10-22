from typing import Literal

import pytest
import torch
from openff.units import unit

from modelforge.dataset import _ImplementedDatasets
from modelforge.potential import NeuralNetworkPotentialFactory, _Implemented_NNPs
from modelforge.tests.helper_functions import (
    setup_potential_for_test,
    _add_electrostatic_to_predicted_properties,
    _add_per_atom_charge_to_predicted_properties,
    _add_per_atom_charge_to_properties_to_process,
)
from modelforge.utils.io import import_
from modelforge.utils.misc import load_configs_into_pydantic_models


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_models_temp")
    return fn


def initialize_model(
    simulation_environment: Literal["PyTorch", "JAX"], config, jit: bool
):
    """Initialize the model based on the simulation environment and configuration."""
    return NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment=simulation_environment,
        potential_parameter=config["potential"],
        jit=jit,
        use_training_mode_neighborlist=True,
    )


def prepare_input_for_model(nnp_input, model):
    """Prepare the input for the model based on the simulation environment."""
    if "JAX" in str(type(model)):
        from modelforge.jax import convert_NNPInput_to_jax

        return convert_NNPInput_to_jax(nnp_input)
    return nnp_input


def validate_output_shapes(output, nr_of_mols: int, energy_expression: str):
    """Validate the output shapes to ensure they are correct."""
    assert len(output["per_system_energy"]) == nr_of_mols
    assert "per_atom_energy" in output
    if energy_expression == "short_range_and_long_range_electrostatic":
        assert "per_atom_charge" in output
        assert "per_atom_charge_uncorrected" in output
        assert "electrostatic_energy" in output


def validate_charge_conservation(
    per_system_total_charge: torch.Tensor,
    per_system_total_charge_uncorrected: torch.Tensor,
    per_system_total_charge_from_dataset: torch.Tensor,
    model_name: str,
):
    """Ensure charge conservation by validating the corrected charges."""

    if "PhysNet".lower() in model_name.lower():
        print(
            "Physnet starts with all zero partial charges"
        )  # NOTE: I am not sure if this is correct
    else:
        assert not torch.allclose(
            per_system_total_charge, per_system_total_charge_uncorrected
        )
    assert torch.allclose(
        per_system_total_charge_from_dataset.to(torch.float32),
        per_system_total_charge,
        atol=1e-5,
    )


from typing import Dict


def validate_per_atom_and_per_system_properties(output: Dict[str, torch.Tensor]):
    """Ensure that the total energy is the sum of atomic energies."""
    assert torch.allclose(
        output["per_system_energy"][0],
        output["per_atom_energy"][0:5].sum(dim=0),
        atol=1e-5,
    )
    assert torch.allclose(
        output["per_system_energy"][1],
        output["per_atom_energy"][5:9].sum(dim=0),
        atol=1e-5,
    )


def validate_chemical_equivalence(output):
    """Ensure that chemically equivalent hydrogens have equal energies."""
    assert torch.allclose(
        output["per_atom_energy"][1:4], output["per_atom_energy"][1], atol=1e-4
    )
    assert torch.allclose(
        output["per_atom_energy"][6:8], output["per_atom_energy"][6], atol=1e-4
    )


def retrieve_molecular_charges(output, atomic_subsystem_indices):
    """Retrieve per-molecule charge from per-atom charges."""
    per_system_total_charge = torch.zeros_like(output["per_system_energy"]).index_add_(
        0, atomic_subsystem_indices, output["per_atom_charge"]
    )
    per_system_total_charge_uncorrected = torch.zeros_like(
        output["per_system_energy"]
    ).index_add_(0, atomic_subsystem_indices, output["per_atom_charge_uncorrected"])
    return per_system_total_charge, per_system_total_charge_uncorrected


def convert_to_pytorch_if_needed(output, nnp_input, model):
    """Convert output to PyTorch tensors if the model is in JAX."""
    if "JAX" in str(type(model)):
        convert_to_pyt = import_("pytorch2jax").pytorch2jax.convert_to_pyt
        output["per_system_energy"] = convert_to_pyt(output["per_system_energy"])
        output["per_atom_energy"] = convert_to_pyt(output["per_atom_energy"])

        if "per_atom_charge" in output:
            output["per_atom_charge"] = convert_to_pyt(output["per_atom_charge"])
        if "per_system_total_charge" in output:
            output["per_system_total_charge"] = convert_to_pyt(
                output["per_system_total_charge"]
            ).to(torch.float32)

        atomic_subsystem_indices = convert_to_pyt(nnp_input.atomic_subsystem_indices)
    else:
        atomic_subsystem_indices = nnp_input.atomic_subsystem_indices
    return output, atomic_subsystem_indices


def test_electrostatics():
    from modelforge.potential.processing import CoulombPotential

    e_elec = CoulombPotential(1.0)
    per_atom_charge = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    # FIXME: this thest has to be implemented


"""     
pairlist =     PairListOutputs(
pair_indices=torch.tensor([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
                            [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]]),
d_ij = torch.tensor([
    )

    pairwise_properties = {}
    pairwise_properties["maximum_interaction_radius"] = 
 """


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_JAX_wrapping(potential_name, single_batch_with_batchsize, prep_temp_dir):

    batch = single_batch_with_batchsize(
        batch_size=1, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )

    # read default parameters
    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="JAX",
        local_cache_dir=str(prep_temp_dir),
    )
    from modelforge.jax import convert_NNPInput_to_jax

    nnp_input = convert_NNPInput_to_jax(batch.nnp_input)
    out = potential(nnp_input)["per_system_energy"]
    import jax

    assert "JAX" in str(type(potential))

    grad_fn = jax.grad(lambda pos: out.sum())  # Create a gradient function
    forces = -grad_fn(
        nnp_input.positions
    )  # Evaluate gradient function and apply negative sign


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_model_factory(potential_name, prep_temp_dir):
    # inference model
    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="PyTorch",
        local_cache_dir=str(prep_temp_dir),
    )
    assert (
        potential_name.upper() in str(type(potential.core_network)).upper()
        or "JAX" in str(type(potential)).upper()
    )
    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="PyTorch",
        jit=True,
        use_default_dataset_statistic=False,
        local_cache_dir=str(prep_temp_dir),
    )

    # trainers model
    trainer = setup_potential_for_test(
        use="training",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="PyTorch",
        local_cache_dir=str(prep_temp_dir),
    )
    assert (
        potential_name.upper() in str(type(trainer.core_network)).upper()
        or "JAX" in str(type(trainer)).upper()
    )


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_energy_scaling_and_offset(
    potential_name, single_batch_with_batchsize, prep_temp_dir
):
    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    # read default parameters
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")

    config["runtime"].local_cache_dir = str(prep_temp_dir)

    # inference model
    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
    )

    batch = single_batch_with_batchsize(
        batch_size=1, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )
    methane = batch.nnp_input

    # load dataset statistic
    import toml

    dataset_statistic = toml.load(trainer.datamodule.dataset_statistic_filename)
    # -------------------------------#
    # initialize model without any postprocessing
    # -------------------------------#

    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        potential_seed=42,
    )
    output_no_postprocessing = potential(methane)
    # -------------------------------#
    # Scale output
    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        dataset_statistic=trainer.dataset_statistic,
        potential_seed=42,
    )
    scaled_output = potential(methane)

    # make sure that the scaled output equals the unscaled output

    mean = unit.Quantity(
        dataset_statistic["training_dataset_statistics"]["per_atom_energy_mean"]
    ).m
    stddev = unit.Quantity(
        dataset_statistic["training_dataset_statistics"]["per_atom_energy_stddev"]
    ).m

    # NOTE: only the per_system_energy is scaled
    compare_to = output_no_postprocessing["per_atom_energy"] * stddev + mean
    assert torch.allclose(scaled_output["per_system_energy"], compare_to.sum())


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_state_dict_saving_and_loading(potential_name, prep_temp_dir):
    import torch

    # give this a unique name so that we can run tests in parallel
    file_path = f"{str(prep_temp_dir)}/{potential_name.lower()}_tsdsal_potential.pth"
    from modelforge.potential import NeuralNetworkPotentialFactory

    # read default parameters
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")

    config["runtime"].local_cache_dir = str(prep_temp_dir)
    # ------------------------------------------------------------- #
    # Use case 1:
    # train a model, save the state_dict and load it again
    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        runtime_parameter=config["runtime"],
        dataset_parameter=config["dataset"],
    )
    torch.save(trainer.lightning_module.state_dict(), file_path)
    trainer.lightning_module.load_state_dict(torch.load(file_path))

    # ------------------------------------------------------------- #
    # Use case 2:
    # load the model in inference mode
    potential = NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment="PyTorch",
        potential_parameter=config["potential"],
    )
    potential.load_state_dict(torch.load(file_path))

    # ------------------------------------------------------------- #
    # Use case 3
    # generate a new trainer and load it
    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        runtime_parameter=config["runtime"],
        dataset_parameter=config["dataset"],
    )

    trainer.lightning_module.load_state_dict(torch.load(file_path))


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_dataset_statistic(potential_name, prep_temp_dir):
    # Test that the scaling parmaeters are propagated from the dataset to the
    # runtime_defaults model and then via the state_dict to the inference model

    import numpy as np
    import torch
    from openff.units import unit

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # read default parameters
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")

    # Extract parameters
    potential_parameter = config["potential"]
    training_parameter = config["training"]
    dataset_parameter = config["dataset"]
    runtime_parameter = config["runtime"]

    runtime_parameter.local_cache_dir = str(prep_temp_dir)

    # test the self energy calculation on the QM9 dataset
    dataset = DataModule(
        name="QM9",
        batch_size=64,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )
    dataset.prepare_data()
    dataset.setup()

    # load dataset stastics from file
    from modelforge.potential.utils import read_dataset_statistics

    dataset_statistic = read_dataset_statistics(dataset.dataset_statistic_filename)
    # extract value to compare against
    toml_E_i_mean = unit.Quantity(
        dataset_statistic["training_dataset_statistics"]["per_atom_energy_mean"]
    ).m

    trainer = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )
    # check that the per_atom_energy_mean is the same as in the dataset statistics
    assert np.isclose(
        toml_E_i_mean,
        unit.Quantity(
            trainer.dataset_statistic["training_dataset_statistics"][
                "per_atom_energy_mean"
            ]
        ).m,
    )
    # give this a unique filename based on potential and the test we are in so we can run test in parallel
    file_path = f"{str(prep_temp_dir)}/{potential_name.lower()}_tsd_potential.pth"

    torch.save(trainer.lightning_module.state_dict(), file_path)

    # NOTE: we are passing dataset statistics explicit to the constructor
    # this is not saved with the state_dict
    potential = NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment="PyTorch",
        potential_parameter=config["potential"],
        dataset_statistic=dataset_statistic,
    )
    potential.load_state_dict(torch.load(file_path))

    assert np.isclose(
        toml_E_i_mean,
        unit.Quantity(
            potential.postprocessing.dataset_statistic["training_dataset_statistics"][
                "per_atom_energy_mean"
            ]
        ).m,
    )


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_energy_between_simulation_environments(
    potential_name, single_batch_with_batchsize, prep_temp_dir
):
    # compare that the energy is the same for the JAX and PyTorch Model
    import numpy as np

    batch = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )
    nnp_input = batch.nnp_input
    # test the forward pass through each of the models
    # cast input and model to torch.float64
    # read default parameters
    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="PyTorch",
        local_cache_dir=str(prep_temp_dir),
    )
    output_torch = potential(nnp_input)["per_system_energy"]

    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment="JAX",
        local_cache_dir=str(prep_temp_dir),
    )
    from modelforge.jax import convert_NNPInput_to_jax

    nnp_input = convert_NNPInput_to_jax(batch.nnp_input)
    output_jax = potential(nnp_input)["per_system_energy"]

    # test tat we get an energie per molecule
    assert np.isclose(output_torch.sum().detach().numpy(), output_jax.sum())


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_forward_pass_with_all_datasets(
    potential_name, dataset_name, datamodule_factory, prep_temp_dir
):
    """Test forward pass with all datasets."""
    import toml
    import torch

    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    # -------------------------------#
    # setup dataset
    # use a subset of the SPICE2 dataset for ANI2x
    if dataset_name.lower().startswith("spice"):
        print("using subset")
        dataset = datamodule_factory(
            dataset_name=dataset_name,
            version_select="nc_1000_v0_HCNOFClS",
            local_cache_dir=str(prep_temp_dir),
        )
    else:
        dataset = datamodule_factory(
            dataset_name=dataset_name, local_cache_dir=str(prep_temp_dir)
        )

    dataset_statistic = toml.load(dataset.dataset_statistic_filename)
    train_dataloader = dataset.train_dataloader()
    batch = next(iter(train_dataloader))
    # -------------------------------#
    # setup model
    config = load_configs_into_pydantic_models(
        f"{potential_name.lower()}", dataset_name.lower()
    )
    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        dataset_statistic=dataset_statistic,
        use_training_mode_neighborlist=True,
        jit=False,
    )
    # -------------------------------#
    # test the forward pass through each of the models
    output = potential(batch.nnp_input)

    # test that the output has the following keys and following dim
    assert "per_system_energy" in output
    assert "per_atom_energy" in output

    assert output["per_system_energy"].shape[0] == 64
    assert output["per_atom_energy"].shape == batch.nnp_input.atomic_numbers.shape

    pair_list = batch.nnp_input.pair_list
    # pairlist is in ascending order in row 0
    assert torch.all(pair_list[0, 1:] >= pair_list[0, :-1])


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_jit(potential_name, single_batch_with_batchsize, prep_temp_dir):
    # setup dataset
    batch = single_batch_with_batchsize(
        batch_size=1, dataset_name="qm9", local_cache_dir=str(prep_temp_dir)
    )
    nnp_input = batch.nnp_input

    # -------------------------------#
    # setup model
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")
    # test the forward pass through each of the models
    potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
    )
    potential = torch.jit.script(potential)
    # -------------------------------#
    potential(nnp_input)


@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("mode", ["training", "inference"])
def test_chemical_equivalency(
    dataset_name, potential_name, mode, single_batch_with_batchsize, prep_temp_dir
):
    nnp_input = single_batch_with_batchsize(
        32, dataset_name, str(prep_temp_dir)
    ).nnp_input

    potential = setup_potential_for_test(
        potential_name,
        mode,
        potential_seed=42,
        local_cache_dir=str(prep_temp_dir),
    )

    output = potential(nnp_input)
    validate_chemical_equivalence(output)
    validate_per_atom_and_per_system_properties(output)


@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize("potential_name", ["SchNet"])
def test_different_neighborlists_for_inference(
    dataset_name, potential_name, single_batch_with_batchsize, prep_temp_dir
):

    # NOTE: the training pairlist only works for a batchsize of 1
    nnp_input = single_batch_with_batchsize(
        1, dataset_name, str(prep_temp_dir)
    ).nnp_input

    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=True,
        local_cache_dir=str(prep_temp_dir),
    )

    output_1 = potential(nnp_input)

    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=False,
        local_cache_dir=str(prep_temp_dir),
    )

    output_2 = potential(nnp_input)

    assert torch.allclose(output_1["per_system_energy"], output_2["per_system_energy"])


@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize(
    "energy_expression",
    [
        "short_range",
        "short_range_and_long_range_electrostatic",
    ],
)
@pytest.mark.parametrize("potential_name", ["SchNet"])
@pytest.mark.parametrize("simulation_environment", ["PyTorch"])
@pytest.mark.parametrize("jit", [False])
def test_multiple_output_heads(
    dataset_name,
    energy_expression,
    potential_name,
    simulation_environment,
    single_batch_with_batchsize,
    jit,
    prep_temp_dir,
):
    """Test models with multiple output heads."""
    # Get input and set up model
    nnp_input = single_batch_with_batchsize(
        32, dataset_name, str(prep_temp_dir)
    ).nnp_input
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")
    config["runtime"].local_cache_dir = str(prep_temp_dir)
    # Modify the config based on the energy expression
    config = _add_per_atom_charge_to_predicted_properties(config)
    if energy_expression == "short_range_and_long_range_electrostatic":
        config = _add_per_atom_charge_to_properties_to_process(config)
        config = _add_electrostatic_to_predicted_properties(config)

    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    model = initialize_model(simulation_environment, config, jit)

    # Perform the forward pass through the model
    output = model(nnp_input)

    # Validate outputs
    validate_output_shapes(output, nr_of_mols, energy_expression)
    validate_chemical_equivalence(output)
    validate_per_atom_and_per_system_properties(output)

    # Test charge correction
    if energy_expression == "short_range_and_long_range_electrostatic":
        per_system_total_charge, per_system_total_charge_uncorrected = (
            retrieve_molecular_charges(output, nnp_input.atomic_subsystem_indices)
        )
        validate_charge_conservation(
            per_system_total_charge,
            per_system_total_charge_uncorrected,
            output["per_system_total_charge"],
            potential_name,
        )


@pytest.mark.parametrize("dataset_name", ["QM9"])
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_forward_pass(
    dataset_name,
    potential_name,
    simulation_environment,
    single_batch_with_batchsize,
    prep_temp_dir,
):
    # this test sends a single batch from different datasets through the model

    # get input and set up model
    nnp_input = single_batch_with_batchsize(
        64, dataset_name, str(prep_temp_dir)
    ).nnp_input
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=True,
        simulation_environment=simulation_environment,
        local_cache_dir=str(prep_temp_dir),
    )
    nnp_input = prepare_input_for_model(nnp_input, potential)

    # perform the forward pass through each of the models
    output = potential(nnp_input)

    # validate the output
    validate_output_shapes(output, nr_of_mols, "short_range")
    output, atomic_subsystem_indices = convert_to_pytorch_if_needed(
        output, nnp_input, potential
    )
    validate_chemical_equivalence(output)


import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="torchviz is not installed")
@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_vis(potential_name, single_batch_with_batchsize, prep_temp_dir):
    batch = single_batch_with_batchsize(
        batch_size=32, dataset_name="SPICE2", local_cache_dir=str(prep_temp_dir)
    )
    nnp_input = batch.nnp_input
    from modelforge.utils.vis import visualize_model

    visualize_model(nnp_input, potential_name, str(prep_temp_dir))


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_calculate_energies_and_forces(
    potential_name, single_batch_with_batchsize, prep_temp_dir
):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch

    batch = single_batch_with_batchsize(
        batch_size=32, dataset_name="SPICE2", local_cache_dir=str(prep_temp_dir)
    )
    nnp_input = batch.nnp_input

    # read default parameters
    trainer = setup_potential_for_test(
        potential_name,
        "training",
        potential_seed=42,
        local_cache_dir=str(prep_temp_dir),
    )
    # get energy and force
    E_training = trainer(nnp_input)["per_system_energy"]
    F_training = -torch.autograd.grad(
        E_training.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]

    # compare to inference model
    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=True,
        jit=False,
        local_cache_dir=str(prep_temp_dir),
    )

    # get energy and force
    E_inference = potential(nnp_input)["per_system_energy"]
    F_inference = -torch.autograd.grad(
        E_inference.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]

    print(f"Energy training: {E_training}")
    print(f"Energy inference: {E_inference}")

    # make sure that dimension are as expected
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]

    assert E_inference.shape == torch.Size([nr_of_mols])
    assert F_inference.shape == (nr_of_atoms_per_batch, 3)

    # make sure that both agree on E and F
    assert torch.allclose(E_inference, E_training, atol=1e-4)
    assert torch.allclose(F_inference, F_training, atol=1e-4)

    # now compare agains the compiled inference model using the neighborlist
    # optimized for MD. NOTE: this requires to reduce the batch size to 1
    # since the neighborlist is not batched

    # reduce batchsize
    batch = single_batch_with_batchsize(
        batch_size=1, dataset_name="SPICE2", local_cache_dir=str(prep_temp_dir)
    )
    nnp_input = batch.nnp_input

    # get the inference model with inference neighborlist and compilre
    # everything
    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=False,
        jit=True,
        local_cache_dir=str(prep_temp_dir),
    )

    # get energy and force
    E_inference = potential(nnp_input)["per_system_energy"]
    F_inference = -torch.autograd.grad(
        E_inference.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]
    # get energy and force
    E_training = potential(nnp_input)["per_system_energy"]
    F_training = -torch.autograd.grad(
        E_training.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]

    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]

    assert E_inference.shape == torch.Size([nr_of_mols])
    assert F_inference.shape == (nr_of_atoms_per_batch, 3)  #  only one molecule

    # make sure that both agree on E and F
    assert torch.allclose(E_inference, E_training, atol=1e-4)
    assert torch.allclose(F_inference, F_training, atol=1e-4)


def get_nr_of_mols(nnp_input):
    import torch
    import jax
    import jax.numpy as jnp

    atomic_subsystem_indices = nnp_input.atomic_subsystem_indices

    if isinstance(atomic_subsystem_indices, torch.Tensor):
        unique_indices = torch.unique(atomic_subsystem_indices)
        nr_of_mols = unique_indices.shape[0]

    elif isinstance(atomic_subsystem_indices, jax.Array):
        unique_indices = jnp.unique(atomic_subsystem_indices)
        nr_of_mols = unique_indices.shape[0]

    else:
        raise TypeError("Unsupported type. Expected a PyTorch tensor or a JAX array.")

    return nr_of_mols


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_calculate_energies_and_forces_with_jax(
    potential_name, single_batch_with_batchsize, prep_temp_dir
):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch
    from modelforge.jax import convert_NNPInput_to_jax

    # get input and set up model
    nnp_input = single_batch_with_batchsize(
        batch_size=1, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input

    # conver tinput to jax
    nnp_input = convert_NNPInput_to_jax(nnp_input)

    potential = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=False,
        jit=False,
        simulation_environment="JAX",
        local_cache_dir=str(prep_temp_dir),
    )

    # forward pass
    result = potential(nnp_input)["per_system_energy"]

    from modelforge.utils.io import import_

    jax = import_("jax")

    grad_fn = jax.grad(lambda pos: result.sum())  # Create a gradient function
    forces = -grad_fn(
        nnp_input.positions
    )  # Evaluate gradient function and apply negative sign

    # test output shapes
    nr_of_mols = get_nr_of_mols(nnp_input)
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]
    assert result.shape == torch.Size([nr_of_mols])  #  only one molecule
    assert forces.shape == (nr_of_atoms_per_batch, 3)  #  only one molecule


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_casting(potential_name, single_batch_with_batchsize, prep_temp_dir):
    # test dtype casting
    import torch

    batch = batch = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )
    batch_ = batch.to_dtype(dtype=torch.float64)
    assert batch_.nnp_input.positions.dtype == torch.float64
    batch_ = batch_.to_dtype(dtype=torch.float32)
    assert batch_.nnp_input.positions.dtype == torch.float32

    nnp_input = batch.nnp_input.to_dtype(dtype=torch.float64)
    assert nnp_input.positions.dtype == torch.float64
    nnp_input = batch.nnp_input.to_dtype(dtype=torch.float32)
    assert nnp_input.positions.dtype == torch.float32
    nnp_input = batch.metadata.to_dtype(dtype=torch.float64)

    # cast input and model to torch.float64
    # read default parameters
    config = load_configs_into_pydantic_models(f"{potential_name.lower()}", "qm9")

    potential = NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment="PyTorch",
        potential_parameter=config["potential"],
        use_training_mode_neighborlist=True,  # can handel batched data
    )
    model = potential.to(dtype=torch.float64)
    nnp_input = batch.to_dtype(dtype=torch.float64).nnp_input

    potential(nnp_input)

    # cast input and model to torch.float64
    potential = NeuralNetworkPotentialFactory.generate_potential(
        simulation_environment="PyTorch",
        potential_parameter=config["potential"],
        use_training_mode_neighborlist=True,  # can handel batched data
    )
    potential = potential.to(dtype=torch.float32)
    nnp_input = batch.to_dtype(dtype=torch.float32).nnp_input

    potential(nnp_input)


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
@pytest.mark.parametrize("simulation_environment", Literal["PyTorch"])
def test_equivariant_energies_and_forces(
    potential_name,
    simulation_environment,
    single_batch_with_batchsize,
    equivariance_utils,
    prep_temp_dir,
):
    """
    Test the calculation of energies and forces for a molecule.
    NOTE: test will be adapted once we have a trained model.
    """
    import torch

    precision = torch.float64

    # initialize the models
    potential = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name=potential_name,
        simulation_environment=simulation_environment,
        local_cache_dir=str(prep_temp_dir),
    ).to(dtype=precision)

    # define the symmetry operations
    translation, rotation, reflection = equivariance_utils
    # define the tolerance
    atol = 1e-3

    # ------------------- #
    # start the test
    # reference values
    nnp_input = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input.to_dtype(dtype=precision)

    reference_result = potential(nnp_input)["per_system_energy"]
    reference_forces = -torch.autograd.grad(
        reference_result.sum(),
        nnp_input.positions,
    )[0]

    # --------------------------------------- #
    # translation test
    # set up input
    nnp_input = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input.to_dtype(dtype=precision)
    translation_nnp_input = nnp_input.to_dtype(dtype=precision)
    translation_nnp_input.positions = translation(translation_nnp_input.positions)

    translation_result = potential(translation_nnp_input)["per_system_energy"]
    assert torch.allclose(
        translation_result,
        reference_result,
        atol=atol,
    )

    translation_forces = -torch.autograd.grad(
        translation_result.sum(),
        translation_nnp_input.positions,
    )[0]

    for t, r in zip(translation_forces, reference_forces):
        if not torch.allclose(t, r, atol=atol):
            print(t, r)

    assert torch.allclose(
        translation_forces,
        reference_forces,
        atol=atol,
    )

    # --------------------------------------- #
    # rotation test
    # set up input
    nnp_input = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input.to_dtype(dtype=precision)
    rotation_input_data = nnp_input.to_dtype(dtype=precision)
    rotation_input_data.positions = rotation(rotation_input_data.positions)
    rotation_result = potential(rotation_input_data)["per_system_energy"]

    for t, r in zip(rotation_result, reference_result):
        if not torch.allclose(t, r, atol=atol):
            print(t, r)

    assert torch.allclose(
        rotation_result,
        reference_result,
        atol=atol,
    )

    rotation_forces = -torch.autograd.grad(
        rotation_result.sum(),
        rotation_input_data.positions,
        create_graph=True,
        retain_graph=True,
    )[0]

    rotate_reference = rotation(reference_forces)
    assert torch.allclose(
        rotation_forces,
        rotate_reference,
        atol=atol,
    )

    # --------------------------------------- #
    # reflection test
    # set up input
    nnp_input = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    ).nnp_input.to_dtype(dtype=precision)
    reflection_input_data = nnp_input.to_dtype(dtype=precision)
    reflection_input_data.positions = reflection(reflection_input_data.positions)
    reflection_result = potential(reflection_input_data)["per_system_energy"]
    reflection_forces = -torch.autograd.grad(
        reflection_result.sum(),
        reflection_input_data.positions,
        create_graph=True,
        retain_graph=True,
    )[0]
    for t, r in zip(reflection_result, reference_result):
        if not torch.allclose(t, r, atol=atol):
            print(t, r)

    assert torch.allclose(
        reflection_result,
        reference_result,
        atol=atol,
    )

    assert torch.allclose(
        reflection_forces,
        reflection(reference_forces),
        atol=atol,
    )


def test_loading_from_checkpoint_file():
    from importlib import resources

    from modelforge.tests import data

    # checkpoint file is saved in tests/data
    chkp_file = resources.files(data) / "best_SchNet-PhAlkEthOH-epoch=00.ckpt"

    from modelforge.potential.potential import load_inference_model_from_checkpoint

    model = load_inference_model_from_checkpoint(chkp_file)
    assert model is not None

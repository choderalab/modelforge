import pytest

from modelforge.potential import _Implemented_NNPs
from modelforge.dataset import _ImplementedDatasets
from modelforge.potential import NeuralNetworkPotentialFactory


def load_configs(model_name: str, dataset_name: str):
    from modelforge.tests.data import (
        potential_defaults,
        training_defaults,
        dataset_defaults,
    )
    from importlib import resources
    from modelforge.train.training import return_toml_config

    potential_path = resources.files(potential_defaults) / f"{model_name.lower()}.toml"
    dataset_path = resources.files(dataset_defaults) / f"{dataset_name}.toml"
    training_path = resources.files(training_defaults) / "default.toml"

    return return_toml_config(
        potential_path=potential_path,
        dataset_path=dataset_path,
        training_path=training_path,
    )


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_JAX_wrapping(model_name, single_batch_with_batchsize_64):
    from modelforge.potential.models import (
        NeuralNetworkPotentialFactory,
    )

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    # inference model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="JAX",
        model_parameter=potential_parameter,
    )

    assert "JAX" in str(type(model))
    nnp_input = single_batch_with_batchsize_64.nnp_input.as_jax_namedtuple()
    out = model(nnp_input)["E"]

    import jax

    grad_fn = jax.grad(lambda pos: out.sum())  # Create a gradient function
    forces = -grad_fn(
        nnp_input.positions
    )  # Evaluate gradient function and apply negative sign


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_model_factory(model_name, simulation_environment):
    from modelforge.potential.models import (
        NeuralNetworkPotentialFactory,
    )
    from modelforge.train.training import TrainingAdapter

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    # Setup loss
    from modelforge.train.training import return_toml_config

    # inference model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment=simulation_environment,
        model_parameter=potential_parameter,
    )
    assert (
        model_name.upper() in str(type(model)).upper()
        or "JAX" in str(type(model)).upper()
    )

    # Extract parameters
    training_parameter = config["training"].get("training_parameter", {})
    # training model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_type=model_name,
        simulation_environment=simulation_environment,
        model_parameter=potential_parameter,
        training_parameter=training_parameter,
    )
    assert type(model) == TrainingAdapter


def test_energy_scaling_and_offset():
    # setup test dataset
    from modelforge.dataset.dataset import DataModule
    from modelforge.potential.ani import ANI2x
    import torch

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # initialize model
    # read default parameters
    from modelforge.train.training import return_toml_config
    from modelforge.tests.data import potential_defaults
    from importlib import resources

    config = load_configs("ani2x_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    import toml

    dataset_statistic = toml.load(dataset.dataset_statistic_filename)
    torch.manual_seed(42)
    model = ANI2x(**potential_parameter, dataset_statistic=dataset_statistic)

    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    # let's predict without any further postprocessing
    output_no_postprocessing = model(methane)

    # let's add self energies
    import toml

    # load dataset statistic
    dataset_statistic = toml.load(dataset.dataset_statistic_filename)
    # load potential parameter
    config = load_configs("ani2x", "qm9")
    potential_parameter = config["potential"].get("potential_parameter", {})
    torch.manual_seed(42)
    model = ANI2x(**potential_parameter, dataset_statistic=dataset_statistic)
    output_with_ase = model(methane)

    # make sure that the raw prediction is the same
    import torch

    assert torch.isclose(output_no_postprocessing["E"], output_with_ase["E"])

    assert torch.isclose(output_with_ase["mse"], torch.tensor([-707050.0]))


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_state_dict_saving_and_loading(model_name):
    from modelforge.potential import NeuralNetworkPotentialFactory
    import torch

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    training_parameter = config["training"].get("training_parameter", {})
    # Setup loss
    from modelforge.train.training import return_toml_config

    model1 = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
        training_parameter=training_parameter,
    )
    torch.save(model1.state_dict(), "model.pth")

    model2 = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
    )
    model2.load_state_dict(torch.load("model.pth"))


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_dataset_statistic(model_name):
    # Test that the scaling parmaeters are propagated from the dataset to the
    # training model and then via the state_dict to the inference model

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    training_parameter = config["training"].get("training_parameter", {})

    # test the self energy calculation on the QM9 dataset
    dataset = DataModule(
        name="QM9",
        batch_size=64,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
        regenerate_dataset_statistic=True,
    )
    dataset.prepare_data()
    dataset.setup()

    import toml
    from openff.units import unit

    dataset_statistic = toml.load(dataset.dataset_statistic_filename)
    toml_E_i_mean = unit.Quantity(
        dataset_statistic["atomic_energies_stats"]["E_i_mean"]
    ).m

    # set up training model
    model = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_statistic=dataset_statistic,
    )
    import torch
    import numpy as np

    # check that the E_i_mean is the same than in the dataset statistics
    assert np.isclose(
        toml_E_i_mean, model.model.postprocessing.per_atom_operations[0].mean
    )

    torch.save(model.state_dict(), "model.pth")

    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
    )
    model.load_state_dict(torch.load("model.pth"))
    assert np.isclose(toml_E_i_mean, model.postprocessing.per_atom_operations[0].mean)


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_energy_between_simulation_environments(
    model_name, single_batch_with_batchsize_64
):
    # compare that the energy is the same for the JAX and PyTorch Model
    import numpy as np
    import torch

    nnp_input = single_batch_with_batchsize_64.nnp_input
    # test the forward pass through each of the models
    # cast input and model to torch.float64
    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    # Setup loss
    from modelforge.train.training import return_toml_config

    torch.manual_seed(42)
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
    )

    output_torch = model(nnp_input)["E"]

    torch.manual_seed(42)
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="JAX",
        model_parameter=potential_parameter,
    )
    nnp_input = nnp_input.as_jax_namedtuple()
    output_jax = model(nnp_input)["E"]

    # test tat we get an energie per molecule
    assert np.isclose(output_torch.sum().detach().numpy(), output_jax.sum())


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_forward_pass_with_all_datasets(model_name, dataset_name, datamodule_factory):
    """Test forward pass with all datasets."""
    import torch

    if dataset_name.lower().startswith("spice"):
        print("using subset")
        dataset = datamodule_factory(
            dataset_name=dataset_name, version_select="nc_1000_v0_HCNOFClS"
        )
    else:
        dataset = datamodule_factory(dataset_name=dataset_name)

    train_dataloader = dataset.train_dataloader()
    batch = next(iter(train_dataloader))

    # test that the neighborlist is correctly generated
    # cast input and model to torch.float64
    # read default parameters
    config = load_configs(f"{model_name.lower()}", dataset_name.lower())

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    from modelforge.potential.models import NeuralNetworkPotentialFactory
    import toml

    dataset_statistic = toml.load(dataset.dataset_statistic_filename)

    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
        dataset_statistic=dataset_statistic,
    )
    output = model(batch.nnp_input)

    # test that the output has the following keys
    assert "E" in output
    assert "E_i" in output
    assert "mse" in output
    assert "ase" in output

    assert output["E"].shape[0] == 64
    assert output["mse"].shape[0] == 64
    assert output["ase"].shape == batch.nnp_input.atomic_numbers.shape
    assert output["E_i"].shape == batch.nnp_input.atomic_numbers.shape

    pair_list = batch.nnp_input.pair_list
    # pairlist is in ascending order in row 0
    assert torch.all(pair_list[0, 1:] >= pair_list[0, :-1])


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_forward_pass(
    model_name, simulation_environment, single_batch_with_batchsize_64
):
    # this test sends a single batch from different datasets through the model
    import torch

    nnp_input = single_batch_with_batchsize_64.nnp_input
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    # Setup loss
    from modelforge.train.training import return_toml_config

    # test the forward pass through each of the models
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment=simulation_environment,
        model_parameter=potential_parameter,
    )
    if "JAX" in str(type(model)):
        nnp_input = nnp_input.as_jax_namedtuple()

    output = model(nnp_input)

    # test tat we get an energie per molecule
    assert len(output["E"]) == nr_of_mols

    # the batch consists of methane (CH4) and amamonium (NH3)
    # which has symmetric hydrogens.
    # This has to be reflected in the atomic energies E_i, which
    # has to be equal for all hydrogens
    if "JAX" not in str(type(model)):
        # assert that the following tensor has equal values for dim=0 index 1 to 4 and 6 to 8
        assert torch.allclose(output["E_i"][1:4], output["E_i"][1], atol=1e-5)
        assert torch.allclose(output["E_i"][6:8], output["E_i"][6], atol=1e-5)

        # make sure that the total energy is \sum E_i
        assert torch.allclose(output["E"][0], output["E_i"][0:5].sum(dim=0), atol=1e-5)
        assert torch.allclose(output["E"][1], output["E_i"][5:9].sum(dim=0), atol=1e-5)


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_calculate_energies_and_forces(model_name, single_batch_with_batchsize_64):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    training_parameter = config["training"].get("training_parameter", {})
    nnp_input = single_batch_with_batchsize_64.nnp_input

    # test the backward pass through each of the models
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]
    # set seed manually
    torch.manual_seed(42)
    model_inference = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        model_parameter=potential_parameter,
    )
    E_inference = model_inference(nnp_input)["E"]

    # backpropagation
    F_inference = -torch.autograd.grad(
        E_inference.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]

    assert E_inference.shape == torch.Size([nr_of_mols])
    assert F_inference.shape == (nr_of_atoms_per_batch, 3)  #  only one molecule

    torch.manual_seed(42)
    model_training = NeuralNetworkPotentialFactory.generate_model(
        use="training",
        model_type=model_name,
        model_parameter=potential_parameter,
        training_parameter=training_parameter,
    )

    E_training = model_training.model.forward(nnp_input)["E"]
    F_training = -torch.autograd.grad(
        E_training.sum(), nnp_input.positions, create_graph=True, retain_graph=True
    )[0]

    # make sure that both agree on E and F
    assert torch.allclose(E_inference, E_training, atol=1e-4)
    assert torch.allclose(F_inference, F_training, atol=1e-4)


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_calculate_energies_and_forces_with_jax(
    model_name, single_batch_with_batchsize_64
):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch

    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    nnp_input = single_batch_with_batchsize_64.nnp_input
    # test the backward pass through each of the models
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]

    # The inference_model fixture now returns a function that expects an environment
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        model_parameter=potential_parameter,
        simulation_environment="JAX",
    )

    nnp_input = nnp_input.as_jax_namedtuple()

    result = model(nnp_input)["E"]

    from modelforge.utils.io import import_

    jax = import_("jax")

    grad_fn = jax.grad(lambda pos: result.sum())  # Create a gradient function
    forces = -grad_fn(
        nnp_input.positions
    )  # Evaluate gradient function and apply negative sign
    assert result.shape == torch.Size([nr_of_mols])  #  only one molecule
    assert forces.shape == (nr_of_atoms_per_batch, 3)  #  only one molecule


def test_pairlist_logic():
    import torch

    # dummy data for illustration
    positions = torch.tensor(
        [
            [0.4933, 0.4460, 0.5762],
            [0.2340, 0.2053, 0.5025],
            [0.6566, 0.1263, 0.8792],
            [0.1656, 0.0338, 0.6708],
            [0.5696, 0.4790, 0.9622],
            [0.3499, 0.4241, 0.8818],
            [0.8400, 0.9389, 0.1888],
            [0.4983, 0.0793, 0.8639],
            [0.6605, 0.7567, 0.1938],
            [0.7725, 0.9758, 0.7063],
        ]
    )
    molecule_indices = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    )  # molecule index for each atom

    # generate index grid
    n = len(molecule_indices)
    i_indices, j_indices = torch.triu_indices(n, n, 1)

    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = molecule_indices[i_indices] == molecule_indices[j_indices]

    # Apply mask to get final pair indices
    i_final_pairs = i_indices[same_molecule_mask]
    j_final_pairs = j_indices[same_molecule_mask]

    # Concatenate to form final (2, n_pairs) tensor
    final_pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    assert torch.allclose(
        final_pair_indices,
        torch.tensor([[0, 0, 1, 3, 5, 5, 6, 8], [1, 2, 2, 4, 6, 7, 7, 9]]),
    )

    # Create pair_coordinates tensor
    pair_coordinates = positions[final_pair_indices.T]
    pair_coordinates = pair_coordinates.view(-1, 2, 3)

    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )
    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )

    # Define a cutoff
    cutoff = 1.0

    # Find pairs within the cutoff
    in_cutoff = (distances <= cutoff).nonzero(as_tuple=False).squeeze()

    # Get the atom indices within the cutoff
    atom_pairs_withing_cutoff = final_pair_indices[:, in_cutoff]
    assert torch.allclose(
        atom_pairs_withing_cutoff,
        torch.tensor([[0, 0, 1, 3, 5, 5, 8], [1, 2, 2, 4, 6, 7, 9]]),
    )


def test_pairlist():
    from modelforge.potential.models import Pairlist, Neighborlist
    import torch

    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )
    from openff.units import unit

    cutoff = 5.0 * unit.nanometer  # no relevant cutoff
    pairlist = Neighborlist(cutoff, only_unique_pairs=True)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices

    # pairlist describes the pairs of interacting atoms within a batch
    # that means for the pairlist provided below:
    # pair1: pairlist[0][0] and pairlist[1][0], i.e. (0,1)
    # pair2: pairlist[0][1] and pairlist[1][1], i.e. (0,2)
    # pair3: pairlist[0][2] and pairlist[1][2], i.e. (1,2)

    assert torch.allclose(
        pair_indices, torch.tensor([[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]])
    )
    # NOTE: pairs are defined on axis=1 and not axis=0
    assert torch.allclose(
        r.r_ij,
        torch.tensor(
            [
                [1.0, 1.0, 1.0],  # pair1, [1.0, 1.0, 1.0] - [0.0, 0.0, 0.0]
                [2.0, 2.0, 2.0],  # pair2, [2.0, 2.0, 2.0] - [0.0, 0.0, 0.0]
                [1.0, 1.0, 1.0],  # pair3, [3.0, 3.0, 3.0] - [0.0, 0.0, 0.0]
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )

    # test with cutoff
    cutoff = 2.0 * unit.nanometer
    pairlist = Neighborlist(cutoff, only_unique_pairs=True)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices

    assert torch.equal(pair_indices, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
    # pairs that are excluded through cutoff: (0,2) and (3,5)
    assert torch.equal(
        r.r_ij,
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )

    assert torch.allclose(
        r.d_ij, torch.tensor([1.7321, 1.7321, 1.7321, 1.7321]), atol=1e-3
    )

    # test with complete pairlist
    cutoff = 2.0 * unit.nanometer
    pairlist = Neighborlist(cutoff, only_unique_pairs=False)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices

    print(pair_indices, flush=True)
    assert torch.equal(
        pair_indices, torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    )

    # make sure that Pairlist and Neighborlist behave the same for large cutoffs
    cutoff = 10.0 * unit.nanometer
    only_unique_pairs = False
    neighborlist = Neighborlist(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # make sure that they are the same also for non-redundant pairs
    cutoff = 10.0 * unit.nanometer
    only_unique_pairs = True
    neighborlist = Neighborlist(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # this should fail
    cutoff = 2.0 * unit.nanometer
    only_unique_pairs = True
    neighborlist = Neighborlist(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r.pair_indices

    assert not pair_indices.shape == neighbor_indices.shape


def test_pairlist_precomputation():
    from modelforge.potential.models import Pairlist
    import torch
    import numpy as np

    atomic_subsystem_indices = torch.tensor([0, 0, 0])

    pairlist = Pairlist()

    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 6)
    assert nr_pairs[0] == 6

    # 3 molecules, 3 atoms each
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 18)
    assert np.all(nr_pairs == [6, 6, 6])

    # 3 molecules, 3,4, and 5 atoms each
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 38)
    assert np.all(nr_pairs == [6, 12, 20])


def test_pairlist_on_dataset():
    # Set up a dataset
    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # get methane input
    batch = next(iter(dataset.train_dataloader(shuffle=False))).nnp_input
    import torch

    # make sure that the pairlist of methane is correct (single molecule)
    assert torch.equal(
        batch.pair_list,
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
            ]
        ),
    )

    # test that the pairlist of 2 molecules is correct (which can then be expected also to be true for N molecules)
    dataset = DataModule(
        name="QM9",
        batch_size=2,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # get methane input
    batch = next(iter(dataset.train_dataloader(shuffle=False))).nnp_input

    assert torch.equal(
        batch.pair_list,
        torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                ],
                [
                    1,
                    2,
                    3,
                    4,
                    0,
                    2,
                    3,
                    4,
                    0,
                    1,
                    3,
                    4,
                    0,
                    1,
                    2,
                    4,
                    0,
                    1,
                    2,
                    3,
                    6,
                    7,
                    8,
                    5,
                    7,
                    8,
                    5,
                    6,
                    8,
                    5,
                    6,
                    7,
                ],
            ]
        ),
    )

    # check that the pairlist maximum value for i is the number of atoms in the batch
    assert (
        int(batch.pair_list[0][-1].item()) + 1 == 8 + 1 == len(batch.atomic_numbers)
    )  # +1 because of 0-based indexing


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
def test_casting(model_name, single_batch_with_batchsize_64):
    # test dtype casting
    import torch

    batch = single_batch_with_batchsize_64
    batch_ = batch.to(dtype=torch.float64)
    assert batch_.nnp_input.positions.dtype == torch.float64
    batch_ = batch_.to(dtype=torch.float32)
    assert batch_.nnp_input.positions.dtype == torch.float32

    nnp_input = batch.nnp_input.to(dtype=torch.float64)
    assert nnp_input.positions.dtype == torch.float64
    nnp_input = batch.nnp_input.to(dtype=torch.float32)
    assert nnp_input.positions.dtype == torch.float32
    nnp_input = batch.metadata.to(dtype=torch.float64)

    # cast input and model to torch.float64
    # read default parameters
    config = load_configs(f"{model_name.lower()}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
    )
    model = model.to(dtype=torch.float64)
    nnp_input = batch.nnp_input.to(dtype=torch.float64)

    model(nnp_input)

    # cast input and model to torch.float64
    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameter=potential_parameter,
    )
    model = model.to(dtype=torch.float32)
    nnp_input = batch.nnp_input.to(dtype=torch.float32)

    model(nnp_input)


@pytest.mark.parametrize("model_name", _Implemented_NNPs.get_all_neural_network_names())
@pytest.mark.parametrize("simulation_environment", ["PyTorch"])
def test_equivariant_energies_and_forces(
    model_name,
    simulation_environment,
    single_batch_with_batchsize_64,
    equivariance_utils,
):
    """
    Test the calculation of energies and forces for a molecule.
    NOTE: test will be adapted once we have a trained model.
    """
    import torch
    from dataclasses import replace

    # cast input and model to torch.float64
    # read default parameters
    config = load_configs(f"{model_name}_without_ase", "qm9")

    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})
    # Setup loss
    from modelforge.train.training import return_toml_config

    model = NeuralNetworkPotentialFactory.generate_model(
        use="inference",
        model_type=model_name,
        simulation_environment=simulation_environment,
        model_parameter=potential_parameter,
    )

    # define the symmetry operations
    translation, rotation, reflection = equivariance_utils
    # define the tolerance
    atol = 1e-3
    nnp_input = single_batch_with_batchsize_64.nnp_input

    # initialize the models
    model = model.to(dtype=torch.float64)

    # ------------------- #
    # start the test
    # reference values
    nnp_input = single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float64)
    reference_result = model(nnp_input)["E"].to(dtype=torch.float64)
    reference_forces = -torch.autograd.grad(
        reference_result.sum(),
        nnp_input.positions,
    )[0]

    # translation test
    translation_nnp_input = replace(nnp_input)
    translation_nnp_input.positions = translation(translation_nnp_input.positions)
    translation_result = model(translation_nnp_input)["E"]
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

    # rotation test
    rotation_input_data = replace(nnp_input)
    rotation_input_data.positions = rotation(rotation_input_data.positions)
    rotation_result = model(rotation_input_data)["E"]

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

    # reflection test
    reflection_input_data = replace(nnp_input)
    reflection_input_data.positions = reflection(reflection_input_data.positions)
    reflection_result = model(reflection_input_data)["E"]
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


def test_pairlist_calculate_r_ij_and_d_ij():
    # Define inputs
    from modelforge.potential.models import Neighborlist
    import torch

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 1.0]]
    )
    atomic_subsystem_indices = torch.tensor([0, 0, 1, 1])
    from openff.units import unit

    cutoff = 3.0 * unit.nanometer

    # Create Pairlist instance
    # --------------------------- #
    # Only unique pairs
    pairlist = Neighborlist(cutoff, only_unique_pairs=True)
    pair_indices = pairlist.enumerate_all_pairs(atomic_subsystem_indices)

    # Calculate r_ij and d_ij
    r_ij = pairlist.calculate_r_ij(pair_indices, positions)
    d_ij = pairlist.calculate_d_ij(r_ij)

    # Check if the calculated r_ij and d_ij are correct
    expected_r_ij = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
    expected_d_ij = torch.tensor([[2.0000], [2.2361]])

    assert torch.allclose(r_ij, expected_r_ij, atol=1e-3)
    assert torch.allclose(d_ij, expected_d_ij, atol=1e-3)

    normalized_r_ij = r_ij / d_ij
    expected_normalized_r_ij = torch.tensor(
        [[1.0000, 0.0000, 0.0000], [0.0000, 0.8944, 0.4472]]
    )
    assert torch.allclose(expected_normalized_r_ij, normalized_r_ij, atol=1e-3)

    # --------------------------- #
    # ALL pairs
    pairlist = Neighborlist(cutoff, only_unique_pairs=False)
    pair_indices = pairlist.enumerate_all_pairs(atomic_subsystem_indices)

    # Calculate r_ij and d_ij
    r_ij = pairlist.calculate_r_ij(pair_indices, positions)
    d_ij = pairlist.calculate_d_ij(r_ij)

    # Check if the calculated r_ij and d_ij are correct
    expected_r_ij = torch.tensor(
        [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, -2.0, -1.0]]
    )
    expected_d_ij = torch.tensor([[2.0000], [2.0000], [2.2361], [2.2361]])

    assert torch.allclose(r_ij, expected_r_ij, atol=1e-3)
    assert torch.allclose(d_ij, expected_d_ij, atol=1e-3)

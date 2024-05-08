import pytest

from modelforge.potential import _IMPLEMENTED_NNPS


@pytest.mark.parametrize("model_name", _IMPLEMENTED_NNPS)
def test_JAX_wrapping(model_name, batch):
    from modelforge.potential.models import (
        NeuralNetworkPotentialFactory,
    )

    # inference model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="inference",
        nnp_name=model_name,
        simulation_environment="JAX",
    )
    assert "JAX" in str(type(model))
    nnp_input = batch.nnp_input.as_jax_namedtuple()
    out = model(nnp_input).E
    import jax

    grad_fn = jax.grad(lambda pos: out.sum())  # Create a gradient function
    forces = -grad_fn(
        nnp_input.positions
    )  # Evaluate gradient function and apply negative sign


@pytest.mark.parametrize("model_name", _IMPLEMENTED_NNPS)
@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_model_factory(model_name, simulation_environment):
    from modelforge.potential.models import (
        NeuralNetworkPotentialFactory,
    )
    from modelforge.train.training import TrainingAdapter

    # inference model
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="inference",
        nnp_name=model_name,
        simulation_environment=simulation_environment,
    )
    assert model_name in str(type(model)) or "JAX" in str(type(model))

    # training model
    model = NeuralNetworkPotentialFactory.create_nnp(
        "training", model_name, simulation_environment
    )
    assert type(model) == TrainingAdapter


def test_energy_scaling_and_offset():
    # setup test dataset
    from modelforge.dataset.dataset import DataModule
    from modelforge.potential.ani import ANI2x

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        for_unit_testing=True,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        normalize=False,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # initialize model
    model = ANI2x()

    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    # let's predict without any further postprocessing
    output_no_postprocessing = model(methane)

    # let's add self energies
    model.dataset_statistics = dataset.dataset_statistics
    output_with_ase = model(methane)

    # make sure that the raw prediction is the same
    import torch

    assert torch.isclose(output_no_postprocessing.raw_E, output_with_ase.raw_E)

    # make sure that the difference in E_predict is the ase
    assert torch.isclose(
        output_with_ase.E - output_no_postprocessing.E,
        output_with_ase.molecular_ase,
    )


@pytest.mark.parametrize("model_name", _IMPLEMENTED_NNPS)
def test_state_dict_saving_and_loading(model_name):
    from modelforge.potential import NeuralNetworkPotentialFactory
    import torch

    model1 = NeuralNetworkPotentialFactory.create_nnp("training", model_name, "PyTorch")
    torch.save(model1.state_dict(), "model.pth")

    model2 = NeuralNetworkPotentialFactory.create_nnp(
        "inference", model_name, "PyTorch"
    )
    model2.load_state_dict(torch.load("model.pth"))


def test_energy_between_simulation_environments(inference_model, batch):
    # compare that the energy is the same for the JAX and PyTorch Model
    import numpy as np
    import torch

    nnp_input = batch.nnp_input
    # test the forward pass through each of the models
    torch.manual_seed(42)
    model = inference_model("PyTorch")

    output_torch = model(nnp_input).E

    torch.manual_seed(42)
    model = inference_model("JAX")
    nnp_input = nnp_input.as_jax_namedtuple()
    output_jax = model(nnp_input).E

    # test tat we get an energie per molecule
    assert np.isclose(output_torch.sum().detach().numpy(), output_jax.sum())


@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_forward_pass(simulation_environment, inference_model, batch):
    # this test sends a single batch from different datasets through the model

    nnp_input = batch.nnp_input
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    # test the forward pass through each of the models
    model = inference_model(simulation_environment)
    if "JAX" in str(type(model)):
        nnp_input = nnp_input.as_jax_namedtuple()

    output = model(nnp_input).E

    # test tat we get an energie per molecule
    assert len(output) == nr_of_mols

    # visualize the compute graph
    from torchviz import make_dot

    if "JAX" not in str(type(model)):

        yhat = model(nnp_input)
        make_dot(yhat, params=dict(list(model.named_parameters()))).render(
            f"compute_graph_{model.__class__.__name__}", format="png"
        )


@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
def test_calculate_energies_and_forces(simulation_environment, inference_model, batch):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch

    nnp_input = batch.nnp_input
    # test the backward pass through each of the models
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]

    # The inference_model fixture now returns a function that expects an environment
    model = inference_model(simulation_environment)
    if "JAX" in str(type(model)):
        nnp_input = nnp_input.as_jax_namedtuple()

    result = model(nnp_input).E

    import jax

    if "JAX" in str(type(model)):
        grad_fn = jax.grad(lambda pos: result.sum())  # Create a gradient function
        forces = -grad_fn(
            nnp_input.positions
        )  # Evaluate gradient function and apply negative sign
    else:
        # backpropagation
        forces = -torch.autograd.grad(
            result.sum(), nnp_input.positions, create_graph=True, retain_graph=True
        )[0]

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

    atomic_subsystem_indices = torch.tensor([80, 80, 80, 11, 11, 11])
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


def test_pairlist_on_dataset(initialized_dataset):
    from modelforge.potential.models import Neighborlist

    for data in initialized_dataset.train_dataloader():
        nnp_input = data.nnp_input
        positions = nnp_input.positions
        atomic_subsystem_indices = nnp_input.atomic_subsystem_indices
        print(atomic_subsystem_indices)
        from openff.units import unit

        pairlist = Neighborlist(cutoff=5.0 * unit.angstrom)
        r = pairlist(positions, atomic_subsystem_indices)
        print(r)
        shapePairlist = r.pair_indices.shape
        shape_distance = r.d_ij.shape

        assert shapePairlist[1] == shape_distance[0]
        assert shapePairlist[0] == 2


def test_casting(batch_QM9_ANI2x, inference_model):
    # test dtype casting
    import torch

    batch = batch_QM9_ANI2x
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
    inference_model = inference_model("PyTorch")
    model = inference_model.to(dtype=torch.float64)
    nnp_input = batch.nnp_input.to(dtype=torch.float64)

    model(nnp_input)

    # cast input and model to torch.float64
    model = inference_model.to(dtype=torch.float32)
    nnp_input = batch.nnp_input.to(dtype=torch.float32)

    model(nnp_input)


@pytest.mark.parametrize("simulation_environment", ["PyTorch"])
def test_equivariant_energies_and_forces(
    simulation_environment, batch, inference_model, equivariance_utils
):
    """
    Test the calculation of energies and forces for a molecule.
    NOTE: test will be adapted once we have a trained model.
    """
    import torch
    from dataclasses import replace

    # define the symmetry operations
    translation, rotation, reflection = equivariance_utils
    # define the tolerance
    atol = 1e-3
    nnp_input = batch.nnp_input

    # initialize the models
    inference_model = inference_model(simulation_environment)
    model = inference_model.to(dtype=torch.float64)

    # ------------------- #
    # start the test
    # reference values
    nnp_input = batch.nnp_input.to(dtype=torch.float64)
    reference_result = model(nnp_input).E.to(dtype=torch.float64)
    reference_forces = -torch.autograd.grad(
        reference_result.sum(),
        nnp_input.positions,
    )[0]

    # translation test
    translation_nnp_input = replace(nnp_input)
    translation_nnp_input.positions = translation(translation_nnp_input.positions)
    translation_result = model(translation_nnp_input).E
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
    rotation_result = model(rotation_input_data).E

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
    reflection_result = model(reflection_input_data).E
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

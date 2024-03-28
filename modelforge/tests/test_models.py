import pytest

from .helper_functions import (
    DATASETS,
    MODELS_TO_TEST,
    SIMPLIFIED_INPUT_DATA,
    return_single_batch,
    equivariance_test_utils,
)


def test_energy_scaling_and_offset():
    # setup test dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.potential.ani import ANI2x

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    # -------------------------------#
    # initialize model
    model = ANI2x()

    # -------------------------------#
    # Test that we can add the reference energy correctly
    dataset.prepare_data(
        remove_self_energies=True, normalize=False, regression_ase=False
    )
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

    # -------------------------------#
    # Test energy scaling


@pytest.mark.parametrize("default_model", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(default_model, dataset):
    # this test sends a single batch from different datasets through the model

    # initialize default model
    model = default_model()
    # return a single batch
    batch = return_single_batch(
        dataset,
    )  # split_file="modelforge/tests/qm9tut/split.npz")
    nnp_input = batch.nnp_input
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    # test the forward pass through each of the models
    output = model(nnp_input).E

    # test tat we get an energie per molecule
    assert len(output) == nr_of_mols


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize("default_model", MODELS_TO_TEST)
def test_calculate_energies_and_forces(input_data, default_model):
    """
    Test the calculation of energies and forces for a molecule.
    """
    import torch

    nnp_input = input_data.nnp_input
    # test the backward pass through each of the models
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]
    nr_of_atoms_per_batch = nnp_input.atomic_subsystem_indices.shape[0]

    # initialize model with default parameters
    model = default_model()

    # forward pass
    result = model(nnp_input).E

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
    pairlist = Neighborlist(cutoff)
    r = pairlist(positions, atomic_subsystem_indices, only_unique_pairs=True)
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
    pairlist = Neighborlist(cutoff)
    r = pairlist(positions, atomic_subsystem_indices, only_unique_pairs=True)
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
    pairlist = Neighborlist(cutoff)
    r = pairlist(positions, atomic_subsystem_indices, only_unique_pairs=False)
    pair_indices = r.pair_indices

    print(pair_indices, flush=True)
    assert torch.equal(
        pair_indices, torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    )

    # make sure that Pairlist and Neighborlist behave the same for large cutoffs
    cutoff = 10.0 * unit.nanometer
    only_unique_pairs = False
    neighborlist = Neighborlist(cutoff)
    pairlist = Pairlist()
    r = pairlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    pair_indices = r.pair_indices
    r = neighborlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # make sure that they are the same also for non-redundant pairs
    cutoff = 10.0 * unit.nanometer
    only_unique_pairs = True
    neighborlist = Neighborlist(cutoff)
    pairlist = Pairlist()
    r = pairlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    pair_indices = r.pair_indices
    r = neighborlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # this should fail
    cutoff = 2.0 * unit.nanometer
    only_unique_pairs = True
    neighborlist = Neighborlist(cutoff)
    pairlist = Pairlist()
    r = pairlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    pair_indices = r.pair_indices
    r = neighborlist(
        positions, atomic_subsystem_indices, only_unique_pairs=only_unique_pairs
    )
    neighbor_indices = r.pair_indices

    assert not pair_indices.shape == neighbor_indices.shape


@pytest.mark.parametrize("dataset", DATASETS)
def test_pairlist_on_dataset(dataset):
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.potential.models import Neighborlist

    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    for data in data_module.train_dataloader():
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


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize("default_model", MODELS_TO_TEST)
def test_equivariant_energies_and_forces(input_data, default_model):
    """
    Test the calculation of energies and forces for a molecule.
    NOTE: test will be adapted once we have a trained model.
    """
    import torch
    from dataclasses import replace

    # define the symmetry operations
    translation, rotation, reflection = equivariance_test_utils()
    # define the tolerance
    atol = 1e-4
    # set seed manually
    torch.manual_seed(1234)
    # initialize the models
    model = default_model().to(torch.float64)

    # ------------------- #
    # start the test
    # reference values
    nnp_input = input_data.nnp_input
    reference_result = model(nnp_input).E.double()
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

    assert torch.allclose(
        translation_forces,
        reference_forces,
        atol=atol,
    )

    # rotation test
    rotation_input_data = replace(nnp_input)
    rotation_input_data.positions = rotation(
        rotation_input_data.positions.to(torch.float32)
    ).double()
    rotation_result = model(rotation_input_data).E

    print(rotation_result)
    print(reference_result, flush=True)

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

    rotate_reference = rotation(reference_forces.to(torch.float32)).double()
    assert torch.allclose(
        rotation_forces,
        rotate_reference,
        atol=atol,
    )

    # reflection test
    reflection_input_data = replace(nnp_input)
    reflection_input_data.positions = reflection(
        reflection_input_data.positions.to(torch.float32)
    ).double()
    reflection_result = model(reflection_input_data).E
    reflection_forces = -torch.autograd.grad(
        reflection_result.sum(),
        reflection_input_data.positions,
        create_graph=True,
        retain_graph=True,
    )[0]

    assert torch.allclose(
        reflection_result,
        reference_result,
        atol=atol,
    )

    assert torch.allclose(
        reflection_forces,
        reflection(reference_forces.to(torch.float32)).double(),
        atol=atol,
    )


def testPairlist_calculate_r_ij_and_d_ij():
    # Define inputs
    from modelforge.potential.models import Pairlist, Neighborlist
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
    pairlist = Neighborlist(cutoff)
    pair_indices = pairlist.calculate_pairs(
        positions, atomic_subsystem_indices, pairlist.cutoff, only_unique_pairs=True
    )

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
    pairlist = Neighborlist(cutoff)
    pair_indices = pairlist.calculate_pairs(
        positions, atomic_subsystem_indices, pairlist.cutoff, only_unique_pairs=False
    )

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

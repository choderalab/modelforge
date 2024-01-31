import pytest

from .helper_functions import (
    DATASETS,
    MODELS_TO_TEST,
    SIMPLIFIED_INPUT_DATA,
    return_single_batch,
    setup_simple_model,
    equivariance_test_utils,
)


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    import torch

    for lightning in [True, False]:
        initialized_model = setup_simple_model(model_class, lightning)
        inputs = return_single_batch(
            dataset,
            mode="fit",
        )  # split_file="modelforge/tests/qm9tut/split.npz")
        nr_of_mols = inputs["atomic_subsystem_indices"].unique().shape[0]
        nr_of_atoms_per_batch = inputs["atomic_subsystem_indices"].shape[0]
        print(f"nr_of_mols: {nr_of_mols}")
        output = initialized_model(inputs)
        print(output)
        if isinstance(output, dict):
            assert (
                output["scalar_representation"].shape[0] == nr_of_atoms_per_batch
            )  # 642
        else:
            assert output.shape[0] == nr_of_mols
            assert output.shape[1] == 1


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_calculate_energies_and_forces(input_data, model_class):
    """
    Test the calculation of energies and forces for a molecule.
    This test will be adapted once we have a trained model.
    """
    import torch

    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]
    nr_of_atoms_per_batch = input_data["atomic_subsystem_indices"].shape[0]
    for lightning in [True, False]:
        model = setup_simple_model(model_class, lightning)  # .double()
        result = model(input_data)
        print(result.sum())
        forces = -torch.autograd.grad(
            result.sum(), input_data["positions"], create_graph=True, retain_graph=True
        )[0]

        assert result.shape == (nr_of_mols, 1)  #  only one molecule
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
    from modelforge.potential.models import _PairList, _NeighbourList
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
    cutoff = 5.0  # no relevant cutoff
    pairlist = _NeighbourList(cutoff, only_unique_pairs=True)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]

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
        r["r_ij"],
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
    cutoff = 2.0  #
    pairlist = _NeighbourList(cutoff, only_unique_pairs=True)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]

    assert torch.equal(pair_indices, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
    # pairs that are excluded through cutoff: (0,2) and (3,5)
    assert torch.equal(
        r["r_ij"],
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
        r["d_ij"], torch.tensor([1.7321, 1.7321, 1.7321, 1.7321]), atol=1e-3
    )

    # test with complete pairlist
    cutoff = 2.0  #
    pairlist = _NeighbourList(cutoff, only_unique_pairs=False)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]

    print(pair_indices, flush=True)
    assert torch.equal(
        pair_indices, torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    )

    # make sure that Pairlist and Neighborlist behave the same for large cutoffs
    cutoff = 10.0  #
    only_unique_pairs = False
    neighborlist = _NeighbourList(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = _PairList(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r["pair_indices"]

    assert torch.equal(pair_indices, neighbor_indices)

    # make sure that they are the same also for non-redundant pairs
    cutoff = 10.0  #
    only_unique_pairs = True
    neighborlist = _NeighbourList(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = _PairList(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r["pair_indices"]

    assert torch.equal(pair_indices, neighbor_indices)

    # this should fail
    cutoff = 2.0  #
    only_unique_pairs = True
    neighborlist = _NeighbourList(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = _PairList(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]
    r = neighborlist(positions, atomic_subsystem_indices)
    neighbor_indices = r["pair_indices"]

    assert not pair_indices.shape == neighbor_indices.shape


@pytest.mark.parametrize("dataset", DATASETS)
def test_pairlist_on_dataset(dataset):
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.potential.models import _NeighbourList

    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    for data in data_module.train_dataloader():
        positions = data["positions"]
        atomic_subsystem_indices = data["atomic_subsystem_indices"]
        print(atomic_subsystem_indices)
        pairlist = _NeighbourList(cutoff=5.0)
        r = pairlist(positions, atomic_subsystem_indices)
        print(r)
        shape_pairlist = r["pair_indices"].shape
        shape_distance = r["d_ij"].shape

        assert shape_pairlist[1] == shape_distance[0]
        assert shape_pairlist[0] == 2


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_equivariant_energies_and_forces(input_data, model_class):
    """
    Test the calculation of energies and forces for a molecule.
    This test will be adapted once we have a trained model.
    """
    import torch
    import torch.nn as nn

    translation, rotation, reflection = equivariance_test_utils()

    for lightning in [True, False]:
        # increase precision to 64 bit
        torch.manual_seed(1234)
        model = setup_simple_model(model_class, lightning).double()
        input_data["positions"] = input_data["positions"]
        # reference values
        reference_result = model(input_data).double()
        reference_forces = -torch.autograd.grad(
            reference_result.sum(),
            input_data["positions"],
            create_graph=True,
            retain_graph=True,
        )[0]

        # translation test
        translation_input_data = input_data.copy()
        translation_input_data["positions"] = translation(
            translation_input_data["positions"]
        )
        translation_result = model(translation_input_data)
        assert torch.allclose(
            translation_result,
            reference_result,
            atol=1e-5,
        )

        translation_forces = -torch.autograd.grad(
            translation_result.sum(),
            translation_input_data["positions"],
            create_graph=True,
            retain_graph=True,
        )[0]

        assert torch.allclose(
            translation_forces,
            reference_forces,
            atol=1e-5,
        )

        # rotation test
        rotation_input_data = input_data.copy()
        rotation_input_data["positions"] = rotation(
            rotation_input_data["positions"].to(torch.float32)
        ).double()
        rotation_result = model(rotation_input_data)

        print(rotation_result)
        print(reference_result, flush=True)

        assert torch.allclose(
            rotation_result,
            reference_result,
            atol=1e-4,
        )

        rotation_forces = -torch.autograd.grad(
            rotation_result.sum(),
            rotation_input_data["positions"],
            create_graph=True,
            retain_graph=True,
        )[0]

        rotate_reference = rotation(reference_forces.to(torch.float32)).double()
        assert torch.allclose(
            rotation_forces,
            rotate_reference,
            atol=1e-4,
        )

        # reflection test
        reflection_input_data = input_data.copy()
        reflection_input_data["positions"] = reflection(
            reflection_input_data["positions"].to(torch.float32)
        ).double()
        reflection_result = model(reflection_input_data)
        reflection_forces = -torch.autograd.grad(
            reflection_result.sum(),
            reflection_input_data["positions"],
            create_graph=True,
            retain_graph=True,
        )[0]

        assert torch.allclose(
            reflection_result,
            reference_result,
            atol=1e-4,
        )

        assert torch.allclose(
            reflection_forces,
            reflection(reference_forces.to(torch.float32)).double(),
            atol=1e-4,
        )


def test_pairlist_calculate_r_ij_and_d_ij():
    # Define inputs
    from modelforge.potential.models import _PairList, _NeighbourList
    import torch

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 1.0]]
    )
    atomic_subsystem_indices = torch.tensor([0, 0, 1, 1])
    cutoff = 3.0

    # Create Pairlist instance
    # --------------------------- #
    # Only unique pairs
    pairlist = _NeighbourList(cutoff, only_unique_pairs=True)
    pair_indices = pairlist.calculate_pairs(
        positions, atomic_subsystem_indices, pairlist.cutoff, only_unique_pairs=True
    )

    # Calculate r_ij and d_ij
    r_ij = pairlist._calculate_r_ij(pair_indices, positions)
    d_ij = pairlist._calculate_d_ij(r_ij)

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
    pairlist = _NeighbourList(cutoff, only_unique_pairs=False)
    pair_indices = pairlist.calculate_pairs(
        positions, atomic_subsystem_indices, pairlist.cutoff, only_unique_pairs=False
    )

    # Calculate r_ij and d_ij
    r_ij = pairlist._calculate_r_ij(pair_indices, positions)
    d_ij = pairlist._calculate_d_ij(r_ij)

    # Check if the calculated r_ij and d_ij are correct
    expected_r_ij = torch.tensor(
        [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, -2.0, -1.0]]
    )
    expected_d_ij = torch.tensor([[2.0000], [2.0000], [2.2361], [2.2361]])

    assert torch.allclose(r_ij, expected_r_ij, atol=1e-3)
    assert torch.allclose(d_ij, expected_d_ij, atol=1e-3)

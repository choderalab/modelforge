import pytest

import numpy as np
from .schnetpack_pain_implementation import setup_painn
from .helper_functions import (
    DATASETS,
    MODELS_TO_TEST,
    return_single_batch,
    setup_simple_model,
)


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    for lightning in [True, False]:
        initialized_model = setup_simple_model(model_class, lightning)
        inputs = return_single_batch(
            dataset,
            mode="fit",
        )  # split_file="modelforge/tests/qm9tut/split.npz")
        output = initialized_model(inputs)
        print(output)
        if isinstance(output, dict):
            assert output["scalar_representation"].shape[0] == 1088
        else:
            assert output.shape[0] == 64
            assert output.shape[1] == 1


# @pytest.mark.parametrize("dataset", DATASETS)
# def test_forward_pass_schnetpack_painn(dataset):
#     initialized_model = setup_painn()
#     inputs = return_single_batch(
#         dataset,
#         mode="fit",
#     )  # split_file="modelforge/tests/qm9tut/split.npz")
#     output = initialized_model(inputs)
#     print(output)
#     assert output.shape[0] == 64
#     assert output.shape[1] == 1


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

    torch.allclose(
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
    from modelforge.potential.models import PairList
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
    pairlist = PairList(cutoff)
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
                [-1.0, -1.0, -1.0],  # pair1, [0.0, 0.0, 0.0] - [1.0, 1.0, 1.0]
                [-2.0, -2.0, -2.0],  # pair2, [0.0, 0.0, 0.0] - [2.0, 2.0, 2.0]
                [-1.0, -1.0, -1.0],  # pair3, [0.0, 0.0, 0.0] - [3.0, 3.0, 3.0]
                [-1.0, -1.0, -1.0],
                [-2.0, -2.0, -2.0],
                [-1.0, -1.0, -1.0],
            ]
        ),
    )

    # test with cutoff
    cutoff = 2.0  #
    pairlist = PairList(cutoff)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r["pair_indices"]

    torch.allclose(pair_indices, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
    # pairs that are excluded through cutoff: (0,2) and (3,5)
    torch.allclose(
        r["r_ij"],
        torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ]
        ),
    )

    torch.allclose(r["d_ij"], torch.tensor([1.7321, 1.7321, 1.7321, 1.7321]))


@pytest.mark.parametrize("dataset", DATASETS)
def test_pairlist_on_dataset(dataset):
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.potential.models import PairList

    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    for data in data_module.train_dataloader():
        positions = data["positions"]
        atomic_subsystem_indices = data["atomic_subsystem_indices"]
        print(atomic_subsystem_indices)
        pairlist = PairList(cutoff=5.0)
        r = pairlist(positions, atomic_subsystem_indices)
        print(r)
        shape_pairlist = r["pair_indices"].shape
        shape_distance = r["d_ij"].shape

        assert shape_pairlist[1] == shape_distance[0]
        assert shape_pairlist[0] == 2

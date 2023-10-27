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
    positions = torch.rand((10, 3))  # 10 atoms with 3D coordinates
    molecule_indices = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    )  # molecule index for each atom

    # generate index grid
    n = len(molecule_indices)
    i_indices, j_indices = torch.meshgrid(torch.arange(n), torch.arange(n))
    # create and apply upper triangular mask to only include pairs (i, j) where i < j
    upper_triangle_mask = i_indices < j_indices
    i_upper_triangle = i_indices[upper_triangle_mask]
    j_upper_triangle = j_indices[upper_triangle_mask]

    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = (
        molecule_indices[i_upper_triangle] == molecule_indices[j_upper_triangle]
    )

    # Apply mask to get final pair indices
    i_final_pairs = i_upper_triangle[same_molecule_mask]
    j_final_pairs = j_upper_triangle[same_molecule_mask]

    # Concatenate to form final (2, n_pairs) tensor
    final_pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    torch.allclose(
        final_pair_indices,
        torch.tensor([[0, 0, 1, 3, 5, 5, 6, 8], [1, 2, 2, 4, 6, 7, 7, 9]]),
    )


def test_pairlist():
    from modelforge.potential.models import PairList
    import torch

    # start with tensor without masking
    mask = torch.tensor([[0, 0, 0], [0, 0, 0]])  # no maksing
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        ]
    )
    cutoff = 5.0  # no relevant cutoff
    pairlist = PairList(cutoff)
    r = pairlist(mask, positions)
    pairlist = r["pairlist"]

    # pairlist describes the pairs of interacting atoms within a batch
    # that means for the pairlist provided below:
    # pair1: pairlist[0][0] and pairlist[1][0], i.e. (0,1)
    # pair2: pairlist[0][1] and pairlist[1][1], i.e. (0,2)
    # pair3: pairlist[0][2] and pairlist[1][2], i.e. (1,2)
    assert torch.allclose(
        pairlist, torch.tensor([[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]])
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

    # test with cutoff, no masking
    cutoff = 2.0  #
    pairlist = PairList(cutoff)
    r = pairlist(mask, positions)
    pairlist = r["pairlist"]

    torch.allclose(pairlist, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
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

    # use masking
    mask = torch.tensor(
        [[1, 0, 0], [0, 1, 0]]
    )  # entries with 1 are masked, that means idx_i = 0 and idx_j = 1 (these values can't be present below)
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        ]
    )
    cutoff = 5.0  # no relevant cutoff
    pairlist = PairList(cutoff)
    r = pairlist(mask, positions)
    pairlist = r["pairlist"]

    torch.allclose(pairlist, torch.tensor([[1, 3], [2, 5]]))
    torch.allclose(r["r_ij"], torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]]))
    torch.allclose(r["d_ij"], torch.tensor([1.7321, 3.4641]))


@pytest.mark.parametrize("dataset", DATASETS)
def test_pairlist_on_dataset(dataset):
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.potential.models import PairList

    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    for b in data_module.train_dataloader():
        print(b.keys())
        R = b["positions"]
        mask = b["atomic_numbers"] == 0
        pairlist = PairList(cutoff=5.0)
        l = pairlist(mask, R)
        print(l)
        shape_pairlist = l["pairlist"].shape
        shape_distance = l["d_ij"].shape

        assert shape_pairlist[1] == shape_distance[0]
        assert shape_pairlist[0] == 2

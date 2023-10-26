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

    assert torch.allclose(
        pairlist, torch.tensor([[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]])
    )
    # NOTE: pairs are defined on axis=1 and not axis=0
    assert torch.allclose(
        r["r_ij"],
        torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-2.0, -2.0, -2.0],
                [-1.0, -1.0, -1.0],
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
    mask = torch.tensor([[1, 0, 0], [0, 1, 0]])
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


def test_pairlist_nopbc():
    import torch
    from modelforge.potential.utils import neighbor_pairs_nopbc

    mask = torch.tensor(
        [[False, False, True], [True, False, False]]
    )  # masking [0][2] and [1][0]
    R = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        ]
    )
    cutoff = (
        2.0  # entry [0][0] and [0][1] as well as [1][1] and [1][2] are within cutoff
    )
    neighbor_idx_below_cutoff = neighbor_pairs_nopbc(mask, R, cutoff)
    assert neighbor_idx_below_cutoff[0][0] == 0 and neighbor_idx_below_cutoff[0][1] == 4
    assert neighbor_idx_below_cutoff[1][0] == 1 and neighbor_idx_below_cutoff[1][1] == 5

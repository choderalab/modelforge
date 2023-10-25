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


def test_pairlist_simple_data():
    from modelforge.potential.models import PairList
    import torch

    mask = torch.tensor([[0, 0, 0], [0, 0, 0]])  # masking [0][0] and [1][2]
    R = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        ]
    )
    cutoff = 3.0
    pairlist = PairList(cutoff)
    r = pairlist(mask, R)
    atom_index12 = r["pairlist"].tolist()
    assert (atom_index12[0][0], atom_index12[1][0]) == (0, 1)
    assert (atom_index12[0][-1], atom_index12[1][-1]) == (4, 5)

    assert r["d_ij"].shape == torch.Size([4])
    assert np.isclose(r["d_ij"].tolist()[0], 1.7320507764816284)
    assert np.isclose(r["d_ij"].tolist()[-1], 1.7320507764816284)

    assert r["r_ij"].shape == (4, 3)


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

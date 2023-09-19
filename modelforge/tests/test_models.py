import pytest

from modelforge.potential.models import BaseNNP
import numpy as np

from .helper_functinos import (
    DATASETS,
    MODELS_TO_TEST,
    return_single_batch,
    setup_simple_model,
)


def test_BaseNNP():
    nnp = BaseNNP()


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_forward_pass(model_class, dataset):
    initialized_model = setup_simple_model(model_class)
    inputs = return_single_batch(dataset, mode="fit")
    output = initialized_model(inputs)
    print(output.shape)
    assert output.shape[0] == 64
    assert output.shape[1] == 1


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
    atom_index12 = r["atom_index12"].tolist()
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
        R = b["R"]
        mask = b["Z"] == 0
        pairlist = PairList(cutoff=5.0)
        pairlist(mask, R)


def test_pairlist_nopbc():
    import torch
    from modelforge.potential.utils import neighbor_pairs_nopbc

    mask = torch.tensor([[0, 0, 1], [1, 0, 0]])  # masking [0][0] and [1][2]
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

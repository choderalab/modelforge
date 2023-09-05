import numpy as np
from modelforge.potential.utils import scatter_add, Dense, GaussianRBF
import torch

def test_scatter_add():
    x = torch.tensor([1, 4, 3, 2], dtype=torch.float32)
    idx_i = torch.tensor([0, 2, 2, 1], dtype=torch.int64)
    result = scatter_add(x, idx_i, dim_size=3)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 7.0]))


def test_Dense():
    layer = Dense(2, 3)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y = layer(x)
    assert y.shape == (2, 3)


def test_GaussianRBF():
    layer = GaussianRBF(10, 5.0)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = layer(x)
    assert y.shape == (3, 10)

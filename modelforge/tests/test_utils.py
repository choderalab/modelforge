import numpy as np
from modelforge.potential.utils import scatter_add
import torch


def test_rbf():
    from modelforge.potential.utils import GaussianRBF
    from .helper_functinos import prepare_pairlist_for_single_batch

    radial_basis = GaussianRBF(n_rbf=20, cutoff=5.0)

    pairlist = prepare_pairlist_for_single_batch()
    radial_basis(pairlist["d_ij"])


def test_scatter_add():
    dim_size = 3
    dim = 0
    x = torch.tensor([1, 4, 3, 2], dtype=torch.float32)
    idx_i = torch.tensor([0, 2, 2, 1], dtype=torch.int64)
    custom_result = scatter_add(x, idx_i, dim_size=dim_size, dim=dim)
    assert torch.equal(custom_result, torch.tensor([1.0, 2.0, 7.0]))

    # Using PyTorch's native scatter_add function
    shape = list(x.shape)
    shape[dim] = dim_size
    native_result = torch.zeros(shape, dtype=x.dtype)
    native_result.scatter_add_(dim, idx_i, x)

    # Check if the results are equal
    assert torch.equal(custom_result, native_result)


def test_GaussianRBF():
    from modelforge.potential.utils import GaussianRBF

    n_rbf = 10
    dim_of_x = 3
    layer = GaussianRBF(10, 5.0)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = layer(x)
    assert y.shape == (dim_of_x, n_rbf)

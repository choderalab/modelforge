from modelforge.potential import _GaussianRBF
import torch


def test_gaussian_rbf_1D():
    dist = torch.tensor([[[1.0]]])

    rbf_expension = _GaussianRBF(n_rbf=6, cutoff=5.0)
    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, rbf_expension(dist), atol=0.0, rtol=1.0e-7)
    assert list(rbf_expension.parameters()) == []

    rbf_expension = _GaussianRBF(n_rbf=6, cutoff=5.0, trainable=True)
    assert list(rbf_expension.parameters()) != []


def test_gaussian_rbf_3D():
    dist = torch.tensor([[[0.0, 1.0, 1.5], [0.5, 1.5, 3.0]]])
    # smear using 4 Gaussian functions with 1. spacing
    smear = _GaussianRBF(start=1.0, cutoff=4.0, n_rbf=4)
    # absolute value of centered distances
    expt = torch.tensor(
        [
            [
                [[1, 2, 3, 4], [0, 1, 2, 3], [0.5, 0.5, 1.5, 2.5]],
                [[0.5, 1.5, 2.5, 3.5], [0.5, 0.5, 1.5, 2.5], [2, 1, 0, 1]],
            ]
        ]
    )
    expt = torch.exp(-0.5 * expt**2)
    assert torch.allclose(expt, smear(dist), atol=0.0, rtol=1.0e-7)
    assert list(smear.parameters()) == []

from modelforge.potential import GaussianRBF
import torch


def test_gaussian_rbf_1D():
    dist = (
        torch.tensor([[[1.0]]]) / 10
    )  # NOTE: converting to nanometer, NOTE: invariant as long as consisten units are used

    from openff.units import unit

    rbf_expension = GaussianRBF(n_rbf=6, cutoff=unit.Quantity(5.0, unit.angstrom))

    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, rbf_expension(dist), atol=1e-4)
    assert list(rbf_expension.parameters()) == []

    rbf_expension = GaussianRBF(n_rbf=6, cutoff=5.0 * unit.angstrom, trainable=True)
    assert list(rbf_expension.parameters()) != []


def test_gaussian_rbf_3D():
    from openff.units import unit

    dist = torch.tensor([[[0.0, 1.0, 1.5], [0.5, 1.5, 3.0]]]) / 10
    # smear using 4 Gaussian functions with 1. spacing
    smear = GaussianRBF(start=0.1, cutoff=4.0 * unit.angstrom, n_rbf=4)
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
    assert torch.allclose(expt, smear(dist), atol=1e-5)
    assert list(smear.parameters()) == []

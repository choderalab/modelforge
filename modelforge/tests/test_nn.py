from modelforge.potential import RadialSymmetryFunction
import torch


def test_gaussian_rbf_1D():
    dist = (
        torch.tensor([[[1.0]]]) / 10
    )  # NOTE: converting to nanometer, NOTE: invariant as long as consisten units are used

    from openff.units import unit

    rbf_expension = RadialSymmetryFunction(
        number_of_gaussians=6, radial_cutoff=unit.Quantity(5.0, unit.angstrom)
    )

    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, rbf_expension(dist), atol=1e-4)
    assert list(rbf_expension.parameters()) == []

    rbf_expension = RadialSymmetryFunction(
        number_of_gaussians=6, radial_cutoff=5.0 * unit.angstrom
    )


def test_gaussian_rbf_3D():
    from openff.units import unit

    dist = torch.tensor([[[0.0, 1.0, 1.5], [0.5, 1.5, 3.0]]])
    # smear using 4 Gaussian functions with 1. spacing
    smear = RadialSymmetryFunction(
        radial_start=0.1 * unit.angstrom,
        radial_cutoff=4.0 * unit.angstrom,
        number_of_gaussians=4,
    )
    # absolute value of centered distances
    expt = torch.tensor(
        [
            [
                [[1, 2, 3, 4], [0, 1, 2, 3], [0.5, 0.5, 1.5, 2.5]],
                [[0.5, 1.5, 2.5, 3.5], [0.5, 0.5, 1.5, 2.5], [2, 1, 0, 1]],
            ]
        ]
    )
    expt = torch.exp(-1 * expt**2)
    print(expt)
    print(smear(dist))
    assert torch.allclose(expt, smear(dist), atol=1e-5)
    assert list(smear.parameters()) == []

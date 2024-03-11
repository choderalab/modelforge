from modelforge.potential import RadialSymmetryFunction
import torch


def test_gaussian_rbf_1D():

    dist = torch.tensor([[[1.0]]]) / 10  # NOTE: converting to nanometer

    from openff.units import unit

    rbf_expension = RadialSymmetryFunction(
        number_of_gaussians=6, radial_cutoff=unit.Quantity(5.0, unit.angstrom)
    )

    expt = torch.exp(-0.5 * torch.tensor([[[1.0, 0.0, 1.0, 4.0, 9.0, 16.0]]]))
    assert torch.allclose(expt, rbf_expension(dist), atol=1e-4)
    assert list(rbf_expension.parameters()) == []

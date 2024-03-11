from modelforge.potential import RadialSymmetryFunction
import torch


def test_radial_symmetry_function():

    from modelforge.potential import RadialSymmetryFunction, CosineCutoff
    import torch
    from openff.units import unit

    # set cutoff and radial symmetry function
    cutoff = CosineCutoff(cutoff=unit.Quantity(5.0, unit.angstrom))
    rbf_expension = RadialSymmetryFunction(
        number_of_gaussians=18, radial_cutoff=unit.Quantity(5.0, unit.angstrom)
    )

    # calculate expension and cutoff
    d = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    vs = rbf_expension(d) * cutoff(d).T

    # make sure that this matches the output of SchNETRepresentation
    from modelforge.potential.schnet import SchNETRepresentation

    rep = SchNETRepresentation(
        radial_cutoff=5 * unit.angstrom,
        number_of_gaussians=18,
        device=torch.device("cpu"),
    )
    d = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    rep_ = rep(d)

    assert torch.allclose(vs, rep_)

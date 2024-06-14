import pytest


def test_tensornet_init():
    from modelforge.potential.tensornet import TensorNet

    net = TensorNet()
    assert net is not None


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function
    # to the output of the modelforge radial symmetry function

    import torch
    from openff.units import unit
    from torchmdnet.models.utils import ExpNormalSmearing

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.utils import TensorNetRadialSymmetryFunction

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5

    # TensorNet constants
    radial_cutoff = 5.0  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8

    rsf = TensorNetRadialSymmetryFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    r_mf = rsf(d_ij / 10)  # torch.Size([5, 1, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)
    
    rcut_ij = cutoff_module(d_ij / 10)  # torch.Size([5]) # NOTE: nanometer
    r_mf = r_mf * rcut_ij
    
    rsf_tn = ExpNormalSmearing(
        cutoff_lower=radial_start,
        cutoff_upper=radial_cutoff,
        num_rbf=radial_dist_divisions,
    )
    r_tn = rsf_tn(d_ij)

    assert torch.allclose(r_mf, r_tn)


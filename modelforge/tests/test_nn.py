def test_radial_symmetry_function():

    from modelforge.potential.utils import SchnetRadialBasisFunction, CosineCutoff
    import torch
    from openff.units import unit

    # set cutoff and radial symmetry function
    cutoff = CosineCutoff(cutoff=unit.Quantity(5.0, unit.angstrom))
    rbf_expension = SchnetRadialBasisFunction(
        number_of_radial_basis_functions=18,
        max_distance=unit.Quantity(5.0, unit.angstrom),
    )

    # calculate expension and cutoff
    d_ij = torch.tensor(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    )  # distances have the dimensions [nr_of_pairs, 1] (because displacement vectors have the dimensions [nr_of_pairs, 3])

    f_ij_cutoff = cutoff(d_ij)
    f_ij = rbf_expension(d_ij)
    vs = f_ij * f_ij_cutoff

    # make sure that this matches the output of SchNETRepresentation
    from modelforge.potential.schnet import SchNETRepresentation

    rep = SchNETRepresentation(
        radial_cutoff=5 * unit.angstrom,
        number_of_radial_basis_functions=18,
    )

    representation = rep(d_ij)
    f_ij_cutoff = representation["f_ij"] * representation["f_cutoff"]

    assert torch.allclose(vs, f_ij_cutoff)

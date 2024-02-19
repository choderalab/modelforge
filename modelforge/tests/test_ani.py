import pytest


@pytest.fixture
def setup_methane():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coordinates = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
        requires_grad=True,
        device=device,
    )
    # In periodic table, C = 6 and H = 1
    species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
    return species, coordinates, device


def test_ani(setup_methane):
    import torch
    import torchani

    species, coordinates, device = setup_methane
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative


def test_compare_radial_symmetry_features(setup_methane):

    import torch
    from modelforge.potential.utils import RadialSymmetryFunction
    from modelforge.potential.models import _PairList
    from openff.units import unit

    _, coordinates, device = setup_methane
    pairlist = _PairList(only_unique_pairs=True)
    distances = pairlist(
        coordinates[0], torch.tensor([0, 0, 0, 0, 0], dtype=torch.int8)
    )
    d_ij = distances["d_ij"]

    # ANI constants
    Rcr = radial_cutoff = 5.1  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8
    EtaR = torch.tensor([19.7], device=device)  # radial eta

    # ShfR is ths shift in the ani implementation
    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]

    # testing the ANI Radial Symmetry Function implemented
    # in modelforge with the same input
    rsf = RadialSymmetryFunction(
        radial_dist_divisions,
        radial_cutoff * unit.angstrom,
        radial_start * unit.angstrom,
        ani_style=True,
    )
    r_mf = rsf(distances["d_ij"]/10)

    from torchani.aev import radial_terms

    r_ani = radial_terms(1, EtaR, ShfR, distances["d_ij"])

    assert torch.allclose(r_mf, r_ani)

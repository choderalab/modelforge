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


def test_compare_radial_symmetry_features():

    import torch
    from modelforge.potential.utils import RadialSymmetryFunction
    from openff.units import unit

    r = torch.rand(5, 3)

    # ANI constants
    radial_cutoff = 5.1  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8
    EtaR = torch.tensor([19.7])  # radial eta
    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]

    rsf = RadialSymmetryFunction(
        radial_dist_divisions,
        radial_cutoff * unit.angstrom,
        radial_start * unit.angstrom,
        ani_style=True,
    )
    r_mf = rsf(r / 10)
    from torchani.aev import radial_terms

    r_ani = radial_terms(1, EtaR, ShfR, r)
    print(r_ani)
    print(r_mf)
    assert torch.allclose(r_mf, r_ani)


def test_compare_angular_symmetry_features(setup_methane):

    import torch
    from modelforge.potential.utils import AngularSymmetryFunction, triple_by_molecule
    from openff.units import unit
    from modelforge.potential.models import _PairList

    species, r, device = setup_methane
    pairlist = _PairList(only_unique_pairs=True)
    pairs = pairlist(r[0], torch.tensor([0, 0, 0, 0, 0]))
    d_ij = pairs["d_ij"].squeeze(1)
    r_ij = pairs["r_ij"].squeeze(1)

    # reformat for input
    species = species.flatten()
    atom_index12 = pairs["pair_indices"]
    species12 = species[atom_index12]
    # ANI constants
    angular_cutoff = Rca = 3.5  # angular_cutoff
    angular_start = 0.9
    EtaA = angular_eta = 19.7
    angular_dist_divisions = 4
    ShfA = torch.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[
        :-1
    ]
    radial_cutoff = 5.1  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8
    EtaA = angular_eta = 19.7
    Zeta = 32.0

    even_closer_indices = (d_ij <= Rca).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, even_closer_indices)
    species12 = species12.index_select(1, even_closer_indices)
    r_ij = r_ij.index_select(0, even_closer_indices)
    central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
    species12_small = species12[:, pair_index12]
    vec12 = r_ij.index_select(0, pair_index12.view(-1)).view(
        2, -1, 3
    ) * sign12.unsqueeze(-1)
    species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])
    asf = AngularSymmetryFunction(
        angular_dist_divisions,
        angular_cutoff * unit.angstrom,
        angular_start * unit.angstrom,
        radial_dist_divisions,
        radial_start * unit.angstrom,
        radial_cutoff * unit.angstrom,
        ani_style=True,
    )

    from torchani.aev import angular_terms
    import math

    angle_sections = 8
    angle_start = math.pi / (2 * angle_sections)
    ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]

    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)

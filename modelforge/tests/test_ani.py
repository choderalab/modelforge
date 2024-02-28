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
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0], dtype=torch.int32, device=device
    )
    mf_input = {
        "atomic_numbers": species.squeeze(),
        "positions": coordinates.squeeze(),
        "atomic_subsystem_indices": atomic_subsystem_indices,
    }

    return species, coordinates, device, mf_input


def test_torchani_ani(setup_methane):
    import torch
    import torchani

    species, coordinates, device, _ = setup_methane
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative


def test_modelforge_ani(setup_methane):
    from modelforge.potential.ani import ANI2x as mf_ANI2x

    _, _, _, mf_input = setup_methane
    model = mf_ANI2x()
    model(mf_input)


def test_compare_radial_symmetry_features():
    # Compare the ANI radial symmetry function
    # agsint the output of the Modelforge radial symmetry function
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
    # Compare the Modelforge angular symmetry function
    # against the original torchani implementation

    import torch
    from modelforge.potential.utils import AngularSymmetryFunction, triple_by_molecule
    from openff.units import unit
    from modelforge.potential.models import _PairList

    # set up relevant system properties
    species, r, _, _ = setup_methane
    pairlist = _PairList(only_unique_pairs=True)
    pairs = pairlist(r[0], torch.tensor([0, 0, 0, 0, 0]))
    d_ij = pairs["d_ij"].squeeze(1)
    r_ij = pairs["r_ij"].squeeze(1)

    # reformat for input
    species = species.flatten()
    atom_index12 = pairs["pair_indices"]
    species12 = species[atom_index12]
    # ANI constants
    # for angular features
    angular_cutoff = Rca = 3.5  # angular_cutoff
    angular_start = 0.8
    EtaA = angular_eta = 19.7
    angular_dist_divisions = 8
    ShfA = torch.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[
        :-1
    ]
    angle_sections = 4
    import math

    angle_start = math.pi / (2 * angle_sections)
    ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]

    # other constants
    Zeta = 32.0

    # get index in right order
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

    # now use formated indices and inputs to calculate the
    # angular terms, both with the modelforge AngularSymmetryFunction
    # and with its implementation in torchani
    from torchani.aev import angular_terms

    # First with ANI
    angular_feature_vector_ani = angular_terms(
        Rca, ShfZ.unsqueeze(0).unsqueeze(0), EtaA, Zeta, ShfA.unsqueeze(1), vec12
    )

    # set up modelforge angular features
    asf = AngularSymmetryFunction(
        angular_cutoff * unit.angstrom,
        angular_start * unit.angstrom,
        angular_dist_divisions,
        angle_sections,
    )
    # NOTE: ANI works with Angstrom, modelforge with nanometer
    angular_feature_vector_mf = asf(vec12 / 10)
    # make sure that the output is the same
    assert angular_feature_vector_ani.dim() == angular_feature_vector_mf.dim()
    assert torch.allclose(angular_feature_vector_ani, angular_feature_vector_mf)



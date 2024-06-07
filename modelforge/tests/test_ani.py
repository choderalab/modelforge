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
    species = torch.tensor([[1, 0, 0, 0, 0]], device=device)
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0], dtype=torch.int32, device=device
    )

    from modelforge.potential.utils import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], device=device),
        positions=coordinates.squeeze(0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        total_charge=torch.tensor([0.0]),
    )

    return species, coordinates, device, nnp_input


@pytest.fixture
def setup_two_methanes():
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
            ],
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ],
        ],
        requires_grad=True,
        device=device,
    )
    # In periodic table, C = 6 and H = 1
    mf_species = torch.tensor([6, 1, 1, 1, 1, 6, 1, 1, 1, 1], device=device)
    ani_species = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], device=device)
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
    )

    atomic_numbers = mf_species
    from modelforge.potential.utils import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=torch.cat((coordinates[0], coordinates[1]), dim=0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        total_charge=torch.tensor([0.0, 0.0]),
    )
    return ani_species, coordinates, device, nnp_input


def test_torchani_ani(setup_two_methanes):
    # Test torchani ANI implementation
    # Test forward pass and backpropagation through network

    import torch
    import torchani

    species, coordinates, device, _ = setup_two_methanes
    model = torchani.models.ANI2x(periodic_table_index=False).to(device)

    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative


def test_modelforge_ani(setup_two_methanes):
    # Test modelforge ANI implementation
    # Test forward pass and backpropagation through network
    from modelforge.potential.ani import ANI2x as mf_ANI2x
    import torch

    # read default parameters
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import potential_defaults

    file_path = resources.files(potential_defaults) / f"ani2x_defaults.toml"
    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameters = config["potential"].get("potential_parameters", {})

    _, _, _, mf_input = setup_two_methanes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mf_ANI2x(**potential_parameters).to(device=device)
    energy = model(mf_input)
    derivative = torch.autograd.grad(energy.E.sum(), mf_input.positions)[0]
    force = -derivative


def test_compare_radial_symmetry_features():
    # Compare the ANI radial symmetry function
    # to the output of the modelforge radial symmetry function
    import torch
    from modelforge.potential.utils import AniRadialSymmetryFunction, CosineCutoff
    from openff.units import unit

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5

    # ANI constants
    radial_cutoff = 5.0  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8
    EtaR = torch.tensor([19.7])  # radial eta
    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]

    # NOTE: we pass in Angstrom to ANI and in nanometer to mf
    rsf = AniRadialSymmetryFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    r_mf = rsf(d_ij / 10)  # torch.Size([5,1, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)
    from torchani.aev import radial_terms

    rcut_ij = cutoff_module(d_ij / 10)  # torch.Size([5]) # NOTE: nanometer

    r_mf = r_mf * rcut_ij
    r_ani = radial_terms(5, EtaR, ShfR, d_ij)  # torch.Size([5,8]) # NOTE: Angstrom
    assert torch.allclose(r_mf, r_ani)


def test_radial_with_diagonal_batching(setup_two_methanes):
    import torch
    from modelforge.potential.utils import AniRadialSymmetryFunction, CosineCutoff
    from openff.units import unit
    from modelforge.potential.models import Pairlist
    from torchani.aev import neighbor_pairs_nopbc

    # ------------ general setup -------------#
    ani_species, ani_coordinates, _, mf_input = setup_two_methanes
    pairlist = Pairlist(only_unique_pairs=True)
    pairs = pairlist(
        mf_input.positions,
        mf_input.atomic_subsystem_indices,
    )
    d_ij = pairs.d_ij

    # ANI constants
    radial_cutoff = 5.1  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 16
    # --------------- ANI setup --------------- #
    EtaR = torch.tensor([19.7])  # radial eta
    ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]

    ani_coordinates_ = ani_coordinates
    ani_coordinates = ani_coordinates_.flatten(0, 1)

    species = ani_species
    atom_index12 = neighbor_pairs_nopbc(species == -1, ani_coordinates_, radial_cutoff)
    selected_coordinates = ani_coordinates.index_select(0, atom_index12.view(-1)).view(
        2, -1, 3
    )
    vec = selected_coordinates[0] - selected_coordinates[1]
    distances = vec.norm(2, -1)
    # ------------ Modelforge calculation ----------#

    radial_symmetry_function = AniRadialSymmetryFunction(
        radial_dist_divisions,
        radial_cutoff * unit.angstrom,
        radial_start * unit.angstrom,
    )

    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)
    rcut_ij = cutoff_module(d_ij)

    radial_symmetry_feature_vector_mf = radial_symmetry_function(d_ij)
    radial_symmetry_feature_vector_mf = radial_symmetry_feature_vector_mf * rcut_ij
    # ------------ ANI calculation ----------#
    from torchani.aev import radial_terms

    assert torch.allclose(distances, d_ij.squeeze(1) * 10)  # NOTE: unit mismatch
    radial_symmetry_feature_vector_ani = radial_terms(
        radial_cutoff, EtaR, ShfR, distances
    )
    # test that both ANI and MF obtain the same radial symmetry outpu
    assert torch.allclose(
        radial_symmetry_feature_vector_mf, radial_symmetry_feature_vector_ani
    )

    assert radial_symmetry_feature_vector_mf.shape == torch.Size(
        [20, radial_dist_divisions]
    )


def test_compare_angular_symmetry_features(setup_methane):
    # Compare the Modelforge angular symmetry function
    # against the original torchani implementation

    import torch
    from modelforge.potential.utils import AngularSymmetryFunction, triple_by_molecule
    from openff.units import unit
    from modelforge.potential.models import Pairlist
    import math

    # set up relevant system properties
    species, r, _, _ = setup_methane
    pairlist = Pairlist(only_unique_pairs=True)
    pairs = pairlist(r[0], torch.tensor([0, 0, 0, 0, 0]))
    d_ij = pairs.d_ij.squeeze(1)
    r_ij = pairs.r_ij.squeeze(1)

    # reformat for input
    species = species.flatten()
    atom_index12 = pairs.pair_indices
    species12 = species[atom_index12]
    # ANI constants
    # for angular features
    angular_cutoff = Rca = 3.5  # angular_cutoff
    angular_start = 0.8
    EtaA = angular_eta = 12.5
    angular_dist_divisions = 8
    ShfA = torch.linspace(angular_start, angular_cutoff, angular_dist_divisions + 1)[
        :-1
    ]
    angle_sections = 4

    angle_start = math.pi / (2 * angle_sections)
    ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]

    # other constants
    Zeta = 14.1

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
    vec12 = vec12 / 10
    # NOTE: ANI operates on a [nr_of_molecules, nr_of_atoms, 3] tensor
    angular_feature_vector_mf = asf(vec12)
    # make sure that the output is the same
    assert angular_feature_vector_ani.size() == angular_feature_vector_mf.size()
    assert torch.allclose(angular_feature_vector_ani, angular_feature_vector_mf)


def test_representation(setup_methane):
    # Compare the Modelforge angular symmetry function
    # against the original torchani implementation

    # methane input
    species, coordinates, device, mf_input = setup_methane

    # generate torchani representation
    import torchani
    import torch

    torchani_model = torchani.models.ANI2x(periodic_table_index=False)

    # calculate aev
    (species, tochani_aev) = torchani_model.aev_computer(
        (species, coordinates), cell=None, pbc=None
    )

    # generate modelforge ani representation
    from modelforge.potential import ANI2x

    # read default parameters
    from modelforge.train.training import return_toml_config
    from importlib import resources
    from modelforge.tests.data import potential_defaults

    file_path = resources.files(potential_defaults) / f"ani2x_defaults.toml"
    config = return_toml_config(file_path)

    # Extract parameters
    potential_parameters = config["potential"].get("potential_parameters", {})

    mf_model = ANI2x(**potential_parameters)
    # perform input checks
    mf_model.input_preparation._input_checks(mf_input)
    # prepare the input for the forward pass
    pairlist_output = mf_model.input_preparation.prepare_inputs(mf_input)
    nnp_input = mf_model.core_module._model_specific_input_preparation(
        mf_input, pairlist_output
    )
    representation = mf_model.core_module.ani_representation_module(nnp_input)

    tochani_aev = tochani_aev.squeeze(0)

    # test for equivalenc
    assert tochani_aev.shape == representation.aevs.shape
    assert torch.allclose(tochani_aev, representation.aevs, atol=1e-4)

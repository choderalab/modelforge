import pytest
from modelforge.tests.helper_functions import setup_potential_for_test
from importlib import resources
from modelforge.tests import data

file_path = resources.files(data) / f"torchani_parameters2.state"


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_ani_temp")
    return fn


def setup_methane():
    import torch

    device = torch.device("cpu")
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

    from modelforge.utils.prop import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], device=device),
        positions=coordinates.squeeze(0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=torch.tensor([0.0]),
    )

    return species, coordinates, device, nnp_input


def setup_two_methanes():
    import torch

    device = torch.device("cpu")

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
    # Specify the translation vector
    translation_vector = torch.tensor([1.0, 1.0, 1.0], device=device)
    # Translate the second "molecule" without in-place modification
    translated_coordinates = (
        coordinates.clone()
    )  # Clone the tensor to avoid in-place modification
    translated_coordinates[1] = translated_coordinates[1] + translation_vector

    print(translated_coordinates)

    # In periodic table, C = 6 and H = 1
    mf_species = torch.tensor([6, 1, 1, 1, 1, 6, 1, 1, 1, 1], device=device)
    ani_species = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], device=device)
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
    )

    atomic_numbers = mf_species
    from modelforge.utils.prop import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=torch.cat(
            (translated_coordinates[0], translated_coordinates[1]), dim=0
        )
        / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=torch.tensor([0.0, 0.0]),
    )
    return ani_species, translated_coordinates, device, nnp_input


@pytest.mark.xfail
def test_ani(prep_temp_dir):
    import torch
    import torchani

    local_cache_dir = str(prep_temp_dir) + "/ani2x_test"
    # NOTE: in the following the input data is scaled to provide both
    # torchani and modelforge ani the same input but in different units
    # NOTE: output unit is Hartree

    # load reference implementation

    # get input
    species, coordinates, device, mf_input = setup_two_methanes()

    # get single model
    model = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    # calculate energy for methane
    energy = model((species, coordinates)).energies
    # get per atom energy
    w, torchani_atomic_energies = model.atomic_energies((species, coordinates))

    # compare to reference energy
    assert torch.allclose(
        torchani_atomic_energies,
        torch.tensor(
            [
                [-38.0841, -0.5797, -0.5898, -0.6034, -0.6027],
                [-38.0841, -0.5797, -0.5898, -0.6034, -0.6027],
            ]
        ),
        rtol=1e-4,
    )

    # calculate reference ase (substract per atom energy without ase from per
    # atom energy with ase)
    # NOTE: this is in Hartree
    reference_ase = torch.tensor(
        [
            [
                -38.08933878049795,
                0.5978583943827134,
                0.5978583943827134,
                0.5978583943827134,
                0.5978583943827134,
            ],
            [
                -38.08933878049795,
                0.5978583943827134,
                0.5978583943827134,
                0.5978583943827134,
                0.5978583943827134,
            ],
        ],
    )

    # ------------------------------------------ #
    # setup modelforge potential
    potential = setup_potential_for_test(
        use="training",
        potential_seed=42,
        potential_name="ani2x",
        jit=False,
        local_cache_dir=local_cache_dir,
    )
    # load the original ani2x parameter set
    potential.load_state_dict(torch.load(file_path))
    # compare to original ani2x dataset
    atomic_energies = potential(mf_input)["per_atom_energy"]
    modelforge_atomic_energies = (
        atomic_energies.flatten() + reference_ase.squeeze(0).flatten()
    )

    print(atomic_energies.flatten())
    print(torchani_atomic_energies.flatten() - reference_ase.flatten())

    print(modelforge_atomic_energies)
    print(torchani_atomic_energies.flatten())

    assert torch.allclose(
        modelforge_atomic_energies,
        torchani_atomic_energies.flatten(),
        rtol=1e-3,
    )


def test_ani_against_torchani_reference(prep_temp_dir):
    import torch

    local_cache_dir = str(prep_temp_dir) + "/ani2x_test_against_torchani_reference"
    # get input
    species, coordinates, device, mf_input = setup_two_methanes()

    # ------------------------------------------ #
    # setup modelforge potential
    potential = setup_potential_for_test(
        use="training",
        potential_seed=42,
        potential_name="ani2x",
        jit=False,
        local_cache_dir=local_cache_dir,
    )
    # load the original ani2x parameter set
    potential.load_state_dict(torch.load(file_path))
    # compare to original ani2x dataset
    atomic_energies = potential(mf_input)["per_atom_energy"]

    assert torch.allclose(
        atomic_energies,
        torch.tensor(
            [
                [0.0052],
                [0.0181],
                [0.0080],
                [-0.0055],
                [-0.0048],
                [0.0052],
                [0.0181],
                [0.0080],
                [-0.0055],
                [-0.0048],
            ]
        ),
        rtol=1e-2,
    )  # that's the atomic energies for the two methane molecules obtained with torchani

    a = 7


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_forward_and_backward(mode, prep_temp_dir):
    # Test modelforge ANI implementation
    # Test forward and backward pass

    import torch

    local_cache_dir = str(prep_temp_dir) + "/ani2x_test_forward_and_backward"
    model = setup_potential_for_test(
        use=mode,
        potential_seed=42,
        potential_name="ani2x",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        jit=False,
        local_cache_dir=local_cache_dir,
    )

    _, _, _, mf_input = setup_two_methanes()

    energy = model(mf_input)
    derivative = torch.autograd.grad(
        energy["per_system_energy"].sum(), mf_input.positions
    )[0]
    per_atom_force = -derivative

    # same input, same output
    assert torch.isclose(
        energy["per_system_energy"][0], energy["per_system_energy"][1], rtol=1e-4
    )
    assert torch.allclose(per_atom_force[0:5], per_atom_force[5:10], rtol=1e-4)


def test_representation():
    # Compare the reference radial symmetry function output against the
    # implemented radial symmetry function
    import torch
    from modelforge.potential import (
        AniRadialBasisFunction,
        CosineAttenuationFunction,
    )
    from openff.units import unit
    from .precalculated_values import (
        provide_reference_values_for_test_ani_test_compare_rsf,
    )

    # set up relevant variables
    d_ij = unit.Quantity(
        torch.tensor([[3.5201], [2.6756], [2.1641], [3.0990], [4.5180]]), unit.angstrom
    )
    max_distance = unit.Quantity(5.0, unit.angstrom)
    min_distance = unit.Quantity(0.8, unit.angstrom)
    radial_dist_divisions = 8

    # pass parameters to the radial symmetry function
    rsf = AniRadialBasisFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=max_distance.to(unit.nanometer).m,
        min_distance=min_distance.to(unit.nanometer).m,
    )

    calculated_rsf = rsf(d_ij.to(unit.nanometer).m)  # torch.Size([5,1, 8])
    cutoff_module = CosineAttenuationFunction(max_distance.to(unit.nanometer).m)

    rcut_ij = cutoff_module(d_ij.to(unit.nanometer).m)  # torch.Size([5])
    calculated_rsf = calculated_rsf * rcut_ij

    # get the precalculated output obtained from torchani for the same d_ij and
    # cutoff values
    reference_rsf = provide_reference_values_for_test_ani_test_compare_rsf()
    assert torch.allclose(calculated_rsf, reference_rsf, rtol=1e-4)


def test_dimer_representation():
    import torch
    from modelforge.potential.ani import ANIRepresentation
    from modelforge.potential.neighbors import NeighborListForTraining
    from modelforge.utils.prop import NNPInput

    # set up representation
    angle_sections = 4
    maximum_interaction_radius = 0.51
    minimum_interaction_radius = 0.08
    number_of_radial_basis_functions = 16
    maximum_interaction_radius_for_angular_features = 0.35
    minimum_interaction_radius_for_angular_features = 0.08
    angular_dist_divisions = 8

    representation = ANIRepresentation(
        radial_max_distance=maximum_interaction_radius,
        radial_min_distance=minimum_interaction_radius,
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        maximum_interaction_radius_for_angular_features=maximum_interaction_radius_for_angular_features,
        minimum_interaction_radius_for_angular_features=minimum_interaction_radius_for_angular_features,
        angular_dist_divisions=angular_dist_divisions,
        angle_sections=angle_sections,
    )

    # set up NNP input
    atomic_numbers = torch.tensor([17, 17], dtype=torch.int32)
    positions = torch.tensor(
        [[0.0000e00, 0.0000e00, 1.0118e-01], [1.2391e-17, 0.0000e00, -1.0118e-01]]
    )
    atomic_subsystem_indices = torch.tensor([0, 0], dtype=torch.int32)
    per_system_total_charge = torch.tensor([0], dtype=torch.int32)
    pair_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    dimer_system = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=per_system_total_charge,
        pair_list=pair_list,
    )

    # set up neighbor list
    nlist = NeighborListForTraining(
        cutoff=0.51,
        only_unique_pairs=True,
    )

    nlist_output = nlist.forward(dimer_system)

    representation_output = representation(
        dimer_system, nlist_output, torch.tensor([6, 6])
    )
    assert representation_output.aevs.shape == (2, 1008)

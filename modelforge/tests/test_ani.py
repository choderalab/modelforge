import pytest
from modelforge.tests.helper_functions import setup_potential_for_test


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
        positions=torch.cat((coordinates[0], coordinates[1]), dim=0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=torch.tensor([0.0, 0.0]),
    )
    return ani_species, coordinates, device, nnp_input


@pytest.mark.parametrize("mode", ["inference", "training"])
@pytest.mark.parametrize("simulation_environment", ["JAX", "PyTorch"])
@pytest.mark.parametrize("jit", [False, True])
def test_init(mode, simulation_environment, jit, prep_temp_dir):

    if simulation_environment == "JAX" and mode == "training":
        pass
    else:
        model = setup_potential_for_test(
            use=mode,
            potential_seed=42,
            potential_name="ani2x",
            simulation_environment=simulation_environment,
            jit=jit,
            local_cache_dir=str(prep_temp_dir),
        )


@pytest.mark.xfail
def test_forward_and_backward_using_torchani():
    # Test torchani ANI implementation
    # Test forward pass and backpropagation through network

    import torch
    import torchani

    species, coordinates, device, _ = setup_two_methanes()
    model = torchani.models.ANI2x(periodic_table_index=False).to(device)

    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    per_atom_force = -derivative


@pytest.mark.parametrize("mode", ["inference", "training"])
def test_forward_and_backward(mode, prep_temp_dir):
    # Test modelforge ANI implementation
    # Test forward pass and backpropagation through network

    import torch

    model = setup_potential_for_test(
        use=mode,
        potential_seed=42,
        potential_name="ani2x",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        jit=False,
        local_cache_dir=str(prep_temp_dir),
    )

    _, _, _, mf_input = setup_two_methanes()

    energy = model(mf_input)
    derivative = torch.autograd.grad(
        energy["per_system_energy"].sum(), mf_input.positions
    )[0]
    per_atom_force = -derivative


def test_representation():
    # Compare the reference radial symmetry function output against the the
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
    calculated_rsf = rsf(
        d_ij.to(unit.nanometer).m
    )  # torch.Size([5,1, 8]) # NOTE: nanometer
    cutoff_module = CosineAttenuationFunction(max_distance.to(unit.nanometer).m)

    rcut_ij = cutoff_module(d_ij.to(unit.nanometer).m)  # torch.Size([5])

    # get the precalculated output obtained from torchani for the same d_ij and
    # cutoff values
    reference_rsf = provide_reference_values_for_test_ani_test_compare_rsf()
    calculated_rsf = calculated_rsf * rcut_ij
    assert torch.allclose(calculated_rsf, reference_rsf, rtol=1e-4)


def test_representation_with_diagonal_batching():
    # Compare the reference radial symmetry function output against the the
    # implemented radial symmetry function for multiple molecules
    import torch
    from modelforge.potential import (
        AniRadialBasisFunction,
        CosineAttenuationFunction,
    )
    from openff.units import unit
    from modelforge.potential.neighbors import Pairlist
    from .precalculated_values import (
        provide_reference_values_for_test_ani_test_compute_rsf_with_diagonal_batching,
    )

    # ------------ general setup -------------#
    ani_species, ani_coordinates, _, mf_input = setup_two_methanes()
    pairlist = Pairlist(only_unique_pairs=True)
    pairs = pairlist(
        mf_input.positions,
        mf_input.atomic_subsystem_indices,
    )
    d_ij = pairs.d_ij

    # ANI constants
    max_distance = unit.Quantity(5.1, unit.angstrom)
    min_distance = unit.Quantity(0.8, unit.angstrom)
    radial_dist_divisions = 16
    # ------------ Modelforge calculation ----------#
    radial_symmetry_function = AniRadialBasisFunction(
        radial_dist_divisions,
        max_distance.to(unit.nanometer).m,
        min_distance.to(unit.nanometer).m,
    )

    cutoff_module = CosineAttenuationFunction(max_distance.to(unit.nanometer).m)
    rcut_ij = cutoff_module(d_ij)

    calculated_rbf_output = radial_symmetry_function(d_ij)
    calculated_rbf_output = calculated_rbf_output * rcut_ij

    # test that both ANI and MF obtain the same radial symmetry outpu
    reference_rbf_output, ani_d_ij = (
        provide_reference_values_for_test_ani_test_compute_rsf_with_diagonal_batching()
    )
    assert torch.allclose(calculated_rbf_output, reference_rbf_output, atol=1e-4)
    assert torch.allclose(
        ani_d_ij, d_ij.squeeze(1) * 10, atol=1e-4
    )  # NOTE: unit mismatch

    assert calculated_rbf_output.shape == torch.Size([20, radial_dist_divisions])


def test_compare_angular_symmetry_features():
    # Compare the calculated angular symmetry function output
    # against the reference angular symmetry functino output

    import torch
    from modelforge.potential.representation import AngularSymmetryFunction
    from modelforge.potential.ani import ANIRepresentation
    from openff.units import unit
    from modelforge.potential.neighbors import Pairlist

    # set up relevant system properties
    species, _, _, nnp_input = setup_methane()
    pairlist = Pairlist(only_unique_pairs=True)
    pairs = pairlist(nnp_input.positions, torch.tensor([0, 0, 0, 0, 0]))
    d_ij = pairs.d_ij.squeeze(1)
    r_ij = pairs.r_ij.squeeze(1)

    # reformat for input
    species = species.flatten()
    atom_index12 = pairs.pair_indices
    # ANI constants
    # for angular features
    angular_cutoff = Rca = unit.Quantity(3.5, unit.angstrom)
    angular_start = unit.Quantity(0.8, unit.angstrom)
    angular_dist_divisions = 8

    # get index in right order
    even_closer_indices = (d_ij <= Rca.to(unit.nanometer).m).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, even_closer_indices)
    r_ij = r_ij.index_select(0, even_closer_indices)
    _, pair_index12, sign12 = ANIRepresentation.triple_by_molecule(atom_index12)
    vec12 = r_ij.index_select(0, pair_index12.view(-1)).view(
        2, -1, 3
    ) * sign12.unsqueeze(-1)

    # now use formated indices and inputs to calculate the
    # angular terms, both with the modelforge AngularSymmetryFunction
    # and with its implementation in torchani

    # ref value
    from .precalculated_values import (
        provide_input_for_test_test_compare_angular_symmetry_features,
    )

    reference_angular_feature_vector = (
        provide_input_for_test_test_compare_angular_symmetry_features()
    )

    # set up modelforge angular features
    asf = AngularSymmetryFunction(
        angular_cutoff.to(unit.nanometer).m,
        angular_start.to(unit.nanometer).m,
        angular_dist_divisions,
        angle_sections=4,
    )
    # NOTE: ANI works with Angstrom, modelforge with nanometer
    # NOTE: ANI operates on a [nr_of_molecules, nr_of_atoms, 3] tensor
    calculated_angular_feature_vector = asf(vec12)
    # make sure that the output is the same
    assert (
        calculated_angular_feature_vector.size()
        == reference_angular_feature_vector.size()
    )

    # NOTE: the order of the angular_feature_vector is not guaranteed as the
    # triple_by_molecule function  used to prepare the inputs does not use
    # stable sorting. When stable sorting is used, the output is identical
    # across platforms, but will not be used here as it is slower and the order
    # of the output is not important in practrice. As such, to check for
    # equivalence in a way that is not order dependent, we can just consider the
    # sum.
    assert torch.isclose(
        torch.sum(calculated_angular_feature_vector),
        torch.sum(reference_angular_feature_vector),
        atol=1e-4,
    )


def test_compare_aev(prep_temp_dir):
    """
    Compare the atomic enviornment vector generated by the reference
    implementation (torchani) and modelforge for the same input
    """
    import torch
    from .precalculated_values import provide_input_for_test_ani_test_compare_aev

    # methane input
    _, _, _, mf_input = setup_methane()

    # generate modelforge ani representation
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="ani2x",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
        only_unique_pairs=True,
        local_cache_dir=str(prep_temp_dir),
    )
    # prepare the input for the forward pass
    pairlist_output = model.neighborlist.forward(mf_input)

    atom_index = model.core_network.lookup_tensor[mf_input.atomic_numbers.long()]
    representation_module_output = model.core_network.ani_representation_module(
        mf_input, pairlist_output, atom_index
    )

    reference_aev = provide_input_for_test_ani_test_compare_aev()
    # test for equivalence
    assert torch.Size([5, 1008]) == representation_module_output.aevs.shape
    # compare a selected subsection
    assert torch.allclose(
        reference_aev, representation_module_output.aevs[::2, :50:5], atol=1e-4
    )

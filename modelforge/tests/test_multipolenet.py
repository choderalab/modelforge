import pytest
import torch
from openff.units import unit

from modelforge.tests.helper_functions import setup_potential_for_test
from modelforge.potential.neighbors import NeighborListForTraining, PairlistData
from modelforge.utils.prop import NNPInput


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    """
    Creates a temporary directory for this test session.
    Parameters
    ----------
    tmp_path_factory

    Returns
    -------

    """
    fn = tmp_path_factory.mktemp("test_multipolenet_temp")
    return fn


def setup_test_system_nnp_input():

    # 12 atoms in 3 systems
    atomic_numbers = torch.tensor([
        26, 8, 8, 29, 17, 17, 30, 16, 8, 8, 8, 8
    ])
    atomic_subsystem_indices = torch.tensor([
        0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2
    ])

    # random positions for 12 atoms
    positions = torch.rand(12, 3)

    per_system_total_charge = torch.tensor([-1, 0, 1])
    per_system_spin_multiplicity = torch.tensor([1, 3, 5])

    data = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=per_system_total_charge,
        per_system_spin_multiplicity=per_system_spin_multiplicity,
    )

    neighborlist_module = NeighborListForTraining(
        local_cutoff=5,
    )
    neighborlist = neighborlist_module(
        data,
    )

    return data, neighborlist


def test_initialize_model(prep_temp_dir):
    """
    Test initialization of the MultipoleNet model.
    Parameters
    ----------
    prep_temp_dir

    Returns
    -------

    """

    model = setup_potential_for_test(
        "multipolenet",
        "training",
        local_cache_dir=str(prep_temp_dir),
    )

    assert model is not None, "MultipoleNet model should be initialized."


def test_shift_monopole(prep_temp_dir):
    multipolenet = setup_potential_for_test(
        "multipolenet",
        "training",
        local_cache_dir=str(prep_temp_dir),
    )

    representation_module = multipolenet.core_network.representation_module

    # two systems
    # 3 atoms, 4 atoms
    multipole_input = torch.tensor(
        [
            [3, 1, 2, 3, 1, 2, 3, 4, 5],
            [4, 1, 2, 3, 1, 2, 3, 4, 5],
            [5, 1, 2, 3, 1, 2, 3, 4, 5],
            [6, 1, 2, 3, 1, 2, 3, 4, 5],
            [6, 1, 2, 3, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 1, 2, 3, 4, 5],
            [14, 1, 2, 3, 1, 2, 3, 4, 5],
        ],
        dtype=torch.float32,
    )
    monopole_target = torch.tensor([-1, 2])
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    shifted_multipole = representation_module._scale_monopole(
        multipole_input,
        monopole_target,
        atomic_subsystem_indices
    )

    shifted_multipole_true = torch.tensor(
        [
            [-0.25, 1, 2, 3, 1, 2, 3, 4, 5],
            [-0.3333, 1, 2, 3, 1, 2, 3, 4, 5],
            [-0.4167, 1, 2, 3, 1, 2, 3, 4, 5],
            [0.3429, 1, 2, 3, 1, 2, 3, 4, 5],
            [0.3429, 1, 2, 3, 1, 2, 3, 4, 5],
            [0.5143, 1, 2, 3, 1, 2, 3, 4, 5],
            [0.8, 1, 2, 3, 1, 2, 3, 4, 5],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(shifted_multipole, shifted_multipole_true, atol=1e-4)


def test_representation_forward(prep_temp_dir):
    multipolenet = setup_potential_for_test(
        "multipolenet",
        "training",
        local_cache_dir=str(prep_temp_dir),
    )

    representation_module = multipolenet.core_network.representation_module

    nnp_input, neighborlist = setup_test_system_nnp_input()

    representation = representation_module(nnp_input, neighborlist.local_cutoff)

    assert representation[0].shape == torch.Size([12, 9])  # 12 atoms, 1+3+5 channels
    assert representation[1].shape == torch.Size([12, 9])  # 12 atoms, 1+3+5 channels


def test_multipole_invariants(prep_temp_dir):
    multipolenet = setup_potential_for_test(
        "multipolenet",
        "training",
        local_cache_dir=str(prep_temp_dir),
    )
    representation_module = multipolenet.core_network.representation_module

    nnp_input, neighborlist = setup_test_system_nnp_input()

    charge_representation = representation_module(nnp_input, neighborlist.local_cutoff)[0]
    charge_invariants = multipolenet.core_network._multipole_invariants(charge_representation)

    assert charge_invariants.shape == torch.Size([12, 3])


def test_core_forward(prep_temp_dir):
    multipolenet = setup_potential_for_test(
        "multipolenet",
        "training",
        local_cache_dir=str(prep_temp_dir),
    )

    nnp_input, neighborlist = setup_test_system_nnp_input()

    output = multipolenet.core_network(nnp_input, neighborlist.local_cutoff)

    assert "per_atom_charge_multipole" in output.keys()
    assert "per_atom_spin_multipole" in output.keys()
    assert "per_atom_scalar_representation" in output.keys()
    assert "per_atom_charge" in output.keys()
    assert "per_system_charge" in output.keys()
    assert "per_system_spin" in output.keys()
    assert "per_system_dipole_moment" in output.keys()
import pytest
import torch
from openff.units import unit

from modelforge.tests.helper_functions import setup_potential_for_test

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
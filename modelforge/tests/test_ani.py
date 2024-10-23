import pytest
from modelforge.tests.helper_functions import setup_potential_for_test
from importlib import resources
from modelforge.tests import data

file_path = resources.files(data) / f"torchani_parameters.state"


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


@pytest.mark.xfail
def test_ani():
    import torch
    import torchani
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
    w, ref_atomic_energies = model.atomic_energies((species, coordinates))

    # compare to reference energy
    assert torch.allclose(
        ref_atomic_energies,
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
            [0.0052, 0.0181, 0.0080, -0.0055, -0.0048],
            [0.0052, 0.0181, 0.0080, -0.0055, -0.0048],
        ],
    ) - torch.tensor(
        [
            [-38.0841, -0.5797, -0.5898, -0.6034, -0.6027],
            [-38.0841, -0.5797, -0.5898, -0.6034, -0.6027],
        ]
    )
    # ------------------------------------------ #
    # setup modelforge potential
    potential = setup_potential_for_test(
        use="training",
        potential_seed=42,
        potential_name="ani2x",
        jit=False,
        local_cache_dir=str(prep_temp_dir),
    )
    # load the original ani2x parameter set
    potential.load_state_dict(torch.load(file_path))
    # compare to original ani2x dataset
    atomic_energies = potential(mf_input)["per_atom_energy"]
    assert torch.allclose(
        atomic_energies.flatten() - reference_ase.flatten(),
        ref_atomic_energies.flatten(),
        rtol=1e-3,
    )

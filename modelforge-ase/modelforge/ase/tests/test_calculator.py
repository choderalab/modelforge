import torch
import pytest
from ase import Atoms
from modelforge.potential import _Implemented_NNPs as Implemented_NNPs


@pytest.mark.parametrize("is_periodic", [False])
# @pytest.mark.parametrize(
#     "potential_name", Implemented_NNPs.get_all_neural_network_names()
# )
@pytest.mark.parametrize(
    "potential_name", ["SCHNET", "ANI2X", "PHYSNET", "PAINN", "AIMNET2"]
)
def test_potential_wrapping(is_periodic, potential_name, prep_temp_dir):
    from modelforge.ase.examples.helper_functions import smiles_to_ase

    ase_system = smiles_to_ase("C", optimize=False)

    from modelforge.utils.misc import load_configs_into_pydantic_models

    # just use the helper function even though we will not do anything with the dataset
    config = load_configs_into_pydantic_models(potential_name, "phalkethoh")

    # generate the potential using the NeuralNetworkPotentialFactory
    # note this is not a trained potential

    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    modelforge_potential = NeuralNetworkPotentialFactory.generate_potential(
        potential_parameter=config["potential"],
        potential_seed=42,
        jit=False,
    )

    from modelforge.utils.prop import NNPInput

    if is_periodic:
        box_vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        box_vectors = torch.zeros(3, 3)

    n_atoms = len(ase_system)
    atomic_numbers = torch.tensor(
        [atom.number for atom in ase_system], dtype=torch.int64
    )
    positions = torch.tensor(
        [atom.position * 0.1 for atom in ase_system], dtype=torch.float32
    )

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=torch.zeros(n_atoms),
        is_periodic=torch.tensor([is_periodic]),
        per_system_total_charge=torch.zeros(1, 1),
        box_vectors=box_vectors,
    )

    modelforge_energy = modelforge_potential(nnp_input)["per_system_energy"]

    # load up the ase calculator
    from modelforge.ase.calculator import ModelForgeCalculator

    ase_system.calc = ModelForgeCalculator(modelforge_potential)
    pe = ase_system.get_potential_energy()

    # convert pe from eV to kJ/mol
    pe = pe * 96.485
    assert torch.isclose(modelforge_energy, torch.tensor([[pe]]))

    # check to ensure that forces are being reported so we can use things such as optimizers
    from ase.optimize import BFGS

    #
    opt = BFGS(ase_system)
    opt.run(fmax=0.05)

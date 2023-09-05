import torch
from modelforge.potential.schnet import Schnet
from modelforge.utils import Inputs


def test_Schnet_init():
    schnet = Schnet(128, 6, 2)
    assert schnet.n_atom_basis == 128
    assert schnet.n_interactions == 6


def test_calculate_energies_and_forces():
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    schnet = Schnet(128, 6, 64)
    Z = torch.tensor([1, 8], dtype=torch.int64)
    R = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], dtype=torch.float32)
    inputs = Inputs(Z, R, torch.tensor([100]))
    result = schnet.calculate_energies_and_forces(inputs)
    assert result.shape[1] == 128  # Assuming n_atom_basis is 128

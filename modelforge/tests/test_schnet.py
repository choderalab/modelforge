from loguru import logger

from modelforge.potential.schnet import Schnet

from .helper_functinos import generate_methane_input
import torch


def test_Schnet_init():
    schnet = Schnet(128, 6, 2)
    assert schnet is not None


def test_schnet_forward():
    model = Schnet(128, 3)
    inputs = {
        "Z": torch.tensor([[1, 2], [2, 3]]),
        "R": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        ),
    }
    energy = model(inputs)
    assert energy.shape == (
        2,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_calculate_energies_and_forces():
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    schnet = Schnet(128, 6, 64)
    methane_inputs = generate_methane_input()
    result = schnet(methane_inputs)
    forces = -torch.autograd.grad(
        result, methane_inputs["R"], create_graph=True, retain_graph=True
    )[0]

    assert result.shape == (1, 1)  #  only one molecule
    assert forces.shape == (1, 5, 3)  #  only one molecule

from loguru import logger

from modelforge.potential.schnet import Schnet

from .helper_functinos import methane_input


def test_Schnet_init():
    schnet = Schnet(128, 6, 2)


def test_calculate_energies_and_forces():
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    schnet = Schnet(128, 6, 64)
    methane_inputs = methane_input()
    result = schnet.calculate_energy(methane_inputs)
    logger.debug(result)
    assert result.shape[0] == 1  # Assuming only one molecule

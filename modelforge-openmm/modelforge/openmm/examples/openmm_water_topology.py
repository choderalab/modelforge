from openmm.app import *
import numpy as np


def openmm_water_topology():
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("water", chain)
    element_O = Element.getBySymbol("O")
    element_H = Element.getBySymbol("H")

    atom_O = topology.addAtom("O", element_O, residue)
    atom_H1 = topology.addAtom("H1", element_H, residue)
    atom_H2 = topology.addAtom("H2", element_H, residue)

    # in nanometers
    positions = (
        np.array(
            [[0, -0.6556, 0], [0.75695, 0.52032, 0], [-0.75695, 0.52032, 0]], np.float32
        )
        * 0.1
    )
    return topology, positions

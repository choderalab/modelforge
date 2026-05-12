from openmm.app import Topology, Element
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def openmm_topology_from_smiles(smiles: str, optimize=False, seed=12345):
    """
    Create an openmm topology from a SMILES string.

    Parameters
    ----------
    smiles: str
        SMILES string.
    optimize: bool
        Should the topology be optimized using MMFF.
    seed: int
        Random seed for the distance-geometry embedder.
    Returns
    -------
    topology: openmm.app.Topology
        OpenMM Topology object representing the molecule.
    positions: np.ndarray
        positions of each atom in the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    topology, positions = openmm_topology_from_rdkit(mol, optimize=optimize, seed=seed)

    return topology, positions


def openmm_topology_from_rdkit(mol, optimize=False, seed=12345):
    """
    Create an openmm topology from a SMILES string.

    Parameters
    ----------
    mol:
        RDKit molecule
    optimize: bool
        Should the topology be optimized using MMFF.
    seed: int
        Random seed for the distance-geometry embedder.
    Returns
    -------
    topology: openmm.app.Topology
        OpenMM Topology object representing the molecule.
    positions: np.ndarray
        positions of each atom in the molecule.
    """

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        raise RuntimeError("3-D embedding failed for molecule.")

    if optimize:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
        if ff_result == -1:
            print("Warning: MMFF optimisation did not converge.")

    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer. Embed 3D coordinates first.")

    conf = mol.GetConformer()
    positions = conf.GetPositions() / 10.0  # (N, 3) float array, converted to nanometer

    # symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL1", chain)

    for atom in mol.GetAtoms():
        atom_species = Element.getBySymbol(atom.GetSymbol())
        atom_name = f"{atom.GetSymbol()}{atom.GetIdx()}"
        topology.addAtom(atom_name, atom_species, residue)

    return topology, positions


def openmm_topology_from_xyz(xyz_file: str, optimize=False, seed=12345):
    """
    Create an openmm topology from a xyz file.

    Parameters
    ----------
    xyz_file:
        path to the xyz file to load

    Returns
    -------
    topology: openmm.app.Topology
        OpenMM Topology object representing the molecule.
    positions: np.ndarray
        positions of each atom in the molecule.

    """

    mol = Chem.MolFromMolFile(xyz_file)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {xyz_file!r}")

    topology, positions = openmm_topology_from_rdkit(mol, optimize=False, seed=seed)
    return topology, positions

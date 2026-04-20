# Helper functions to convert between RDKit and ASE

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from ase import Atoms


def rdkit_mol_to_ase(mol: Mol, smiles: str = "") -> Atoms:
    """
    Convert an RDKit Mol (with 3D conformer) to an ASE Atoms object.

    Parameters
    ----------
    mol    : RDKit Mol with at least one 3D conformer embedded
    smiles : optional label string for the Atoms object

    Returns
    -------
    ase.Atoms
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer. Embed 3D coordinates first.")

    conf = mol.GetConformer()
    positions = conf.GetPositions()  # (N, 3) float array in Angstrom

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    ase_system = Atoms(symbols=symbols, positions=positions)
    ase_system.info["smiles"] = smiles
    return ase_system


def smiles_to_ase(smiles: str, optimize: bool = True, seed: int = 42) -> Atoms:
    """
    Full pipeline: SMILES → RDKit Mol → 3-D embedding → ASE Atoms.

    Parameters
    ----------
    smiles   : SMILES string
    optimize : run MMFF94 geometry optimisation after embedding
    seed     : random seed for the distance-geometry embedder

    Returns
    -------
    ase.Atoms
    """
    # 1. Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # 2. Add explicit hydrogens (important for realistic geometry)
    mol = Chem.AddHs(mol)

    # 3. Embed 3-D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        raise RuntimeError("3-D embedding failed for molecule.")

    # 4. Geometry optimisation (optional but recommended)
    if optimize:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
        if ff_result == -1:
            print("Warning: MMFF optimisation did not converge.")

    return rdkit_mol_to_ase(mol, smiles=smiles)


def ase_to_rdkit(ase_system: Atoms) -> Chem.Mol:
    from rdkit.Chem import Conformer

    mol = Chem.RWMol()
    conf = Conformer(len(ase_system))
    for i, atom in enumerate(ase_system):
        rd_atom = Chem.Atom(int(atom.number))
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, atom.position)
    mol.AddConformer(conf)
    return mol.GetMol()

checkpoint_file_path = "/home/cri/Downloads/model.ckpt"

from modelforge.potential.potential import load_inference_model_from_checkpoint

potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)

from modelforge.ase.calculator import ModelForgeCalculator

from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = ModelForgeCalculator(potential)

pe = atoms.get_potential_energy()
forces = atoms.get_forces()
print(pe)
print(forces)

from ase.optimize import BFGS

#
opt = BFGS(atoms)
opt.run(fmax=0.05)


def ase_to_rdkit(atoms):
    from rdkit import Chem
    from rdkit.Chem import Conformer

    mol = Chem.RWMol()
    conf = Conformer(len(atoms))
    for i, atom in enumerate(atoms):
        rd_atom = Chem.Atom(int(atom.number))
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, atom.position)
    mol.AddConformer(conf)
    return mol.GetMol()


mol = ase_to_rdkit(atoms)

"""SMILES-based modelforge-ase example for new users.

Edit the USER SETTINGS section below, then run this file.
"""

from ase.optimize import BFGS

from modelforge.ase import ModelForgeCalculator, ase_to_rdkit, smiles_to_ase
from modelforge.potential.potential import load_inference_model_from_checkpoint
from modelforge.utils.io import get_path_string
import modelforge.ase.tests as ase_tests


# =========================
# USER SETTINGS (EDIT HERE)
# =========================
SMILES = "CCCCO"  # Example: "O", "CCO", "NCCCCCCO"
RDKIT_OPTIMIZE = False  # Set True to run an MMFF94 geometry optimization in RDKit.
MODEL_PATH = f"{get_path_string(ase_tests)}/data/model.ckpt"
OPT_FMAX_EV_PER_ANGSTROM = 0.05
OPT_LOGFILE = "example_from_smiles_opt.log"
OPT_TRAJECTORY = "example_from_smiles_opt.traj"
CONVERT_TO_RDKIT = True
# =========================

print("=== modelforge-ase example (from SMILES) ===")
print(f"Checkpoint: {MODEL_PATH}")
print(f"SMILES: {SMILES}")

# 1) Load the trained model checkpoint.
potential = load_inference_model_from_checkpoint(MODEL_PATH, jit=False)

# 2) Build an ASE Atoms object from a SMILES string.
atoms = smiles_to_ase(SMILES, optimize=RDKIT_OPTIMIZE)

# 3) Attach the modelforge calculator so ASE can compute energies/forces.
atoms.calc = ModelForgeCalculator(potential)

# 4) Single-point energy and forces.
pe = atoms.get_potential_energy()
forces = atoms.get_forces()
print(f"Potential energy: {pe:.6f} eV")
print("Forces (eV/angstrom):")
print(forces)

# 5) Geometry optimization.
optimizer = BFGS(atoms, logfile=OPT_LOGFILE, trajectory=OPT_TRAJECTORY)
optimizer.run(fmax=OPT_FMAX_EV_PER_ANGSTROM)
print("Optimization complete.")
print(f"Wrote optimizer log: {OPT_LOGFILE}")
print(f"Wrote optimizer trajectory: {OPT_TRAJECTORY}")

# 6) Optional conversion back to an RDKit molecule.
if CONVERT_TO_RDKIT:
    mol = ase_to_rdkit(atoms)
    print(f"Converted optimized structure to RDKit object: {type(mol).__name__}")


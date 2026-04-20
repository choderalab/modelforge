"""Simple modelforge-ase example for new users.

Edit the USER SETTINGS section below, then run this file.
"""

from ase.optimize import BFGS
from pathlib import Path

from modelforge.ase.calculator import ModelForgeCalculator
from modelforge.ase.examples.helper_functions import smiles_to_ase
from modelforge.potential.potential import load_inference_model_from_checkpoint


# =========================
# USER SETTINGS (EDIT HERE)
# =========================
SMILES = "O"  # Example: "O", "CCO", "NCCCCCCO"
RDKIT_OPTIMIZE = False  # Set True to run an MMFF94 geometry optimization in RDKit.
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "tests" / "data" / "model.ckpt")
OPT_FMAX_EV_PER_ANGSTROM = 0.05
OPT_LOGFILE = "simple_example_opt.log"
OPT_TRAJECTORY = "simple_example_opt.traj"
# =========================

print("=== modelforge-ase simple example ===")
print(f"Checkpoint: {CHECKPOINT_PATH}")
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

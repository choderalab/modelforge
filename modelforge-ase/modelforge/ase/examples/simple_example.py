"""Simple modelforge-ase example for new users.

Edit the USER SETTINGS section below, then run this file.
"""

from ase.build import molecule
from ase.optimize import BFGS

from modelforge.ase import ModelForgeCalculator
from modelforge.potential.potential import load_inference_model_from_checkpoint
from modelforge.utils.io import get_path_string
import modelforge.ase.tests as ase_tests


# =========================
# USER SETTINGS (EDIT HERE)
# =========================
# `molecule()` expects an ASE-recognized structure name string.
# Common examples include: "H2O", "NH3", "CH4", "CO2", and "C6H6".
ASE_MOLECULE_NAME = "H2O"
MODEL_PATH = f"{get_path_string(ase_tests)}/data/model.ckpt"
OPT_FMAX_EV_PER_ANGSTROM = 0.05
OPT_LOGFILE = "simple_example_opt.log"
OPT_TRAJECTORY = "simple_example_opt.traj"
# =========================

print("=== modelforge-ase simple example ===")
print(f"Checkpoint: {MODEL_PATH}")
print(f"ASE molecule name: {ASE_MOLECULE_NAME}")

# 1) Load the trained model checkpoint.
potential = load_inference_model_from_checkpoint(MODEL_PATH, jit=False)

# 2) Build an ASE Atoms object from a named ASE structure.
atoms = molecule(ASE_MOLECULE_NAME)

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

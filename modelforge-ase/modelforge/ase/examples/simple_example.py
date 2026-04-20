# initialize a potential from a checkpoint file

from modelforge.utils.io import get_path_string
from modelforge.ase.tests import data
from modelforge.potential.potential import load_inference_model_from_checkpoint

# checkpoint file is saved in tests/data
checkpoint_file_path = get_path_string(data) + "/model.ckpt"
potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)

# to use the potential wtih ASE we can need to import the ModelForgeCalculator class,
# which wraps the potential in an ASE-compatible calculator interface
from modelforge.ase.calculator import ModelForgeCalculator

# let us use one of ase's built in molecules
from ase.build import molecule

atoms = molecule("H2O")
atoms.calc = ModelForgeCalculator(potential)

# extract the energy and forces
pe = atoms.get_potential_energy()
forces = atoms.get_forces()
print("potential energy: ", pe)
print("forces: ", forces)

# let us try doing an optimization
from ase.optimize import BFGS

#
opt = BFGS(atoms)
opt.run(fmax=0.05)

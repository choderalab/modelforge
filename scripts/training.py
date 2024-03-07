# This is an example script that trains the PaiNN model on the .
from lightning import Trainer
import torch
from modelforge.potential.schnet import LightningSchNET

from modelforge.potential import CosineCutoff, RadialSymmetryFunction
from modelforge.potential.utils import Embedding

from openff.units import unit

max_atomic_number = 100
nr_atom_basis = 64
number_of_gaussians = 14
nr_interaction_blocks = 2

cutoff = unit.Quantity(5, unit.angstrom)
embedding = Embedding(max_atomic_number, nr_atom_basis)
assert embedding.embedding_dim == nr_atom_basis
radial_symmetry_function_module = RadialSymmetryFunction(
    number_of_gaussians=number_of_gaussians, radial_cutoff=cutoff
)

cutoff = CosineCutoff(cutoff=cutoff)

from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

data = QM9Dataset(for_unit_testing=False)
dataset = TorchDataModule(
    data, batch_size=128, split=FirstComeFirstServeSplittingStrategy()
)

dataset.prepare_data(remove_self_energies=True, normalize=True)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

trainer = Trainer(
    max_epochs=10_000,
    num_nodes=1,
    accelerator="cpu",
    # devices=[0],
    # callbacks=[
    #    EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001)
    # ],
)


model = LightningSchNET(
    embedding=embedding,
    nr_interaction_blocks=nr_interaction_blocks,
    radial_symmetry_function_module=radial_symmetry_function_module,
    cutoff_module=cutoff,
    nr_filters=32,
    lr=1e-6,
)

print(model)
# set scaling and ase values
model.dataset_statistics = dataset.dataset_statistics

# Move model to the appropriate dtype and device
model = model.to(torch.float32)
# Run training loop and validate
trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

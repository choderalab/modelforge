# This is an example script that trains the PaiNN model on the .
from lightning import Trainer
import torch
from modelforge.potential import SchNet, PaiNN, ANI2x
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

# Set up dataset
data = QM9Dataset(for_unit_testing=False)
dataset = TorchDataModule(
    data, batch_size=128, split=FirstComeFirstServeSplittingStrategy()
)

dataset.prepare_data(remove_self_energies=True, normalize=False)

# Set up model
model = ANI2x()  # PaiNN() # ANI2x()
model = model.to(torch.float32)

print(model)

# set up traininer

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

trainer = Trainer(
    max_epochs=10_000,
    num_nodes=1,
    accelerator="cpu",
)


# Run training loop and validate
trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

# This is an example script that trains the ANI2x model on the ANI2x dataset.
from lightning import Trainer
import torch

# import the dataset and model factory
from modelforge.potential import NeuralNetworkPotentialFactory
from modelforge.dataset.ani2x import ANI2xDataset

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.utils import RandomRecordSplittingStrategy
from pytorch_lightning.loggers import TensorBoardLogger

# set up tensor board logger
logger = TensorBoardLogger("tb_logs", name="training")

# Set up dataset
data = ANI2xDataset(force_download=False, for_unit_testing=False)

dataset = TorchDataModule(
    data, batch_size=512, splitting_strategy=RandomRecordSplittingStrategy()
)

dataset.prepare_data(remove_self_energies=True, normalize=False)

# Set up model
model = NeuralNetworkPotentialFactory.create_nnp("training", "ANI2x")

# set up traininer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

trainer = Trainer(
    max_epochs=10_000,
    num_nodes=1,
    devices=1,
    accelerator="cpu",
    logger=logger,  # Add the logger here
    callbacks=[
        EarlyStopping(monitor="val_loss", min_delta=0.05, patience=20, verbose=True)
    ],
)


# Run training loop and validate
trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

# tensorboard --logdir tb_logs

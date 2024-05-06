# This is an example script that trains an implemented model on the QM9 dataset.
from lightning import Trainer
import torch

# import the models implemented in modelforge, for now SchNet, PaiNN, ANI2x or PhysNet
from modelforge.potential import NeuralNetworkPotentialFactory
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.utils import RandomRecordSplittingStrategy
from pytorch_lightning.loggers import TensorBoardLogger

# set up tensor board logger
logger = TensorBoardLogger("tb_logs", name="training")

# Set up dataset
data = QM9Dataset(force_download=True, for_unit_testing=False)

dataset = TorchDataModule(
    data, batch_size=512, splitting_strategy=RandomRecordSplittingStrategy()
)

dataset.prepare_data(remove_self_energies=True, normalize=False)

# Set up model
model = NeuralNetworkPotentialFactory.create_nnp("training", "PhysNet")


# set up traininer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

trainer = Trainer(
    max_epochs=1000,
    num_nodes=1,
    devices=1,
    accelerator="cpu",
    logger=logger,  # Add the logger here
    callbacks=[
        EarlyStopping(
            monitor="acu_rmse_val_loss", min_delta=0.05, patience=20, verbose=True
        )
    ],
)


# Run training loop and validate
trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

# tensorboard --logdir tb_logs

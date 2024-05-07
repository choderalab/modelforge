# This is an example script that trains an implemented model on the QM9 dataset.
from lightning import Trainer
from modelforge.potential import NeuralNetworkPotentialFactory
from modelforge.dataset.dataset import DataModule
from modelforge.dataset.utils import RandomRecordSplittingStrategy
from pytorch_lightning.loggers import TensorBoardLogger

# set up tensor board logger
logger = TensorBoardLogger("tb_logs", name="training")

# Set up dataset

dm = DataModule(
    name="QM9",
    batch_size=512,
    splitting_strategy=RandomRecordSplittingStrategy(),
    normalize=False,
)


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
            monitor="epoch_rmse_val_loss", min_delta=0.05, patience=20, verbose=True
        )
    ],
)


# Run training loop and validate
trainer.fit(model, datamodule=dm)

# tensorboard --logdir tb_logs

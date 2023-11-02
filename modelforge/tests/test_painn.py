import os

import pytest

from modelforge.potential.pain import PaiNN
from modelforge.potential.utils import CosineCutoff

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
from .helper_functions import (
    SIMPLIFIED_INPUT_DATA,
)


def test_PaiNN_init():
    """Test initialization of the PaiNN neural network potential."""
    import torch

    embedding = torch.nn.Embedding(100, 128)
    painn = PaiNN(embedding, 6, 10, cutoff_fn=CosineCutoff(5.0))
    assert painn is not None, "PaiNN model should be initialized."


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
def test_painn_forward(input_data):
    """
    Test the forward pass of the Schnet model.
    """
    import torch

    embedding = torch.nn.Embedding(100, 128)

    model = PaiNN(embedding, 6, 10, cutoff_fn=CosineCutoff(5.0))
    energy = model(input_data)
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert energy.shape == (
        nr_of_mols,
        1,
    )  # Assuming energy is calculated per sample in the batch


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="This test is not intended to be performed regularly.",
)
def test_schnetpack_PaiNN():
    # NOTE: this test sets up the schnetpack Painn implementation and trains a model
    # units are not cnnsistent with the modelforge implementation yet
    import os

    import schnetpack as spk
    import schnetpack.transform as trn
    from schnetpack.datasets import QM9

    from .schnetpack_pain_implementation import setup_painn

    qm9tut = "./qm9tut"
    if not os.path.exists("qm9tut"):
        os.makedirs(qm9tut)

    qm9data = QM9(
        "./qm9.db",
        batch_size=64,
        num_train=1000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32(),
        ],
        property_units={QM9.U0: "eV"},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=False,  # set to false, when not using a GPU
        load_properties=[QM9.U0],  # only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()

    nnpot = setup_painn()
    import torch
    import torchmetrics

    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
    )

    import pytorch_lightning as pl

    logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss",
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm9tut,
        max_epochs=3,  # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=qm9data)

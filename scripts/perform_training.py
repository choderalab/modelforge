# This is an example script that trains an implemented model on the QM9 dataset.
from lightning import Trainer
from modelforge.potential import NeuralNetworkPotentialFactory
from modelforge.dataset.dataset import DataModule
from modelforge.dataset.utils import RandomRecordSplittingStrategy
from pytorch_lightning.loggers import TensorBoardLogger


def perform_training(
    model_name: str, dataset_name: str, nr_of_epochs: int, accelerator: str = "cpu"
):
    # set up tensor board logger
    logger = TensorBoardLogger("tb_logs", name="training")

    # Set up dataset
    dm = DataModule(
        name=dataset_name,
        batch_size=512,
        splitting_strategy=RandomRecordSplittingStrategy(),
        remove_self_energies=True,
        for_unit_testing=True,
    )
    # Set up model
    model = NeuralNetworkPotentialFactory.create_nnp("training", model_name)

    # set up traininer
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    trainer = Trainer(
        max_epochs=nr_of_epochs,
        num_nodes=1,
        devices=1,
        accelerator=accelerator,
        logger=logger,  # Add the logger here
        callbacks=[
            EarlyStopping(
                monitor="epoch_rmse_val_loss", min_delta=0.05, patience=20, verbose=True
            )
        ],
    )

    dm.prepare_data()
    dm.setup()

    from modelforge.utils.misc import visualize_model

    visualize_model(dm, model_name)

    # Run training loop and validate
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    trainer.test(model, dataloaders=dm.test_dataloader())

# tensorboard --logdir tb_logs


if __name__ == "__main__":
    from modelforge.potential import _Implemented_NNPs

    for model_name in _Implemented_NNPs.get_all_neural_network_names():

        dataset_name = "QM9"
        nr_of_repeats = 5
        # Run training loop and validate
        for i in range(nr_of_repeats):
            print("Running training iteration:", i)
            perform_training(
                model_name, dataset_name, nr_of_epochs=1000, accelerator="gpu"
            )

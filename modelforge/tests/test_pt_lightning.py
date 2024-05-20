def test_datamodule():
    # This is an example script that trains an implemented model on the QM9 dataset.
    from modelforge.dataset.dataset import DataModule

    # Set up dataset

    dm = DataModule(
        name="QM9",
        batch_size=512,
        normalize=False,
    )


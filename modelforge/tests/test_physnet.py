def test_physnet_init():

    from modelforge.potential.physnet import PhysNet

    model = PhysNet()


def test_physnet_forward():

    from modelforge.potential.physnet import PhysNet
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # Set up dataset
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    model = PhysNet()
    model(methane)

import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.utils import Inputs


def default_input():
    train_loader = initialize_dataloader()
    R, Z, E = train_loader.dataset[0]
    padded_values = -Z.eq(-1).sum().item()
    Z_ = Z[:padded_values]
    R_ = R[:padded_values]
    return Inputs(Z_, R_, E)


def initialize_dataloader() -> torch.utils.data.DataLoader:
    data = QM9Dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module.train_dataloader()


methane_coordinates = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.63918859, 0.63918859, 0.63918859],
        [-0.63918859, -0.63918859, 0.63918859],
        [-0.63918859, 0.63918859, -0.63918859],
        [0.63918859, -0.63918859, -0.63918859],
    ]
)

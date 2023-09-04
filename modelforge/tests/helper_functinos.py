import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential.schnet import Schnet
from modelforge.utils import Inputs
from modelforge.potential.models import BaseNNP

from typing import Optional

MODELS_TO_TEST = [Schnet]
DATASETS = [QM9Dataset]


def setup_simple_model(model_class) -> Optional[BaseNNP]:
    if model_class is Schnet:
        return Schnet(n_atom_basis=128, n_interactions=3, n_filters=64)
    else:
        raise NotImplementedError


def single_default_input(dataset, mode):
    train_loader = initialize_dataset(dataset, mode)
    v = train_loader.dataset[0]
    print(v)
    Z, R, E = v["Z"], v["R"], v["E"]
    padded_values = -Z.eq(-1).sum().item()
    Z_ = Z[:padded_values]
    R_ = R[:padded_values]
    return Inputs(Z_, R_, E)


def default_input_iterator():
    train_loader = initialize_dataset()
    for R, Z, E in train_loader:
        padded_values = -Z.eq(-1).sum().item()
        Z_ = Z[:padded_values]
        R_ = R[:padded_values]
        yield Inputs(Z_, R_, E)


def initialize_dataset(dataset, mode: str) -> TorchDataModule:
    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup(mode)
    return data_module


def methane_input():
    Z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.int64)
    R = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.63918859, 0.63918859, 0.63918859],
            [-0.63918859, -0.63918859, 0.63918859],
            [-0.63918859, 0.63918859, -0.63918859],
            [0.63918859, -0.63918859, -0.63918859],
        ]
    )
    E = torch.tensor([0.0])
    return Inputs(Z, R, E)

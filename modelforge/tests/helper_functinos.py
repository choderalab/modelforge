import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential.schnet import Schnet, LighningSchnet
from modelforge.utils import Inputs
from modelforge.potential.models import BaseNNP

from typing import Optional

MODELS_TO_TEST = [Schnet]
DATASETS = [QM9Dataset]


def setup_simple_model(model_class, lightning: bool = False) -> Optional[BaseNNP]:
    if model_class is Schnet:
        if lightning:
            return LighningSchnet(n_atom_basis=128, n_interactions=3, n_filters=64)
        return Schnet(n_atom_basis=128, n_interactions=3, n_filters=64)
    else:
        raise NotImplementedError


def return_single_batch(dataset, mode: str):
    train_loader = initialize_dataset(dataset, mode)
    for batch in train_loader.train_dataloader():
        return batch


def initialize_dataset(dataset, mode: str) -> TorchDataModule:
    data = dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup(mode)
    return data_module


def generate_methane_input():
    Z = torch.tensor([[6, 1, 1, 1, 1]], dtype=torch.int64)
    R = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.63918859, 0.63918859, 0.63918859],
                [-0.63918859, -0.63918859, 0.63918859],
                [-0.63918859, 0.63918859, -0.63918859],
                [0.63918859, -0.63918859, -0.63918859],
            ]
        ],
        requires_grad=True,
    )
    E = torch.tensor([0.0], requires_grad=True)
    return {"Z": Z, "R": R, "E": E}

from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.potential.models import NeuralNetworkPotential
import torch
import torch.nn as nn


def initialize_dataloader() -> torch.utils.data.DataLoader:
    data = QM9Dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module.train_dataloader()


def setup_simple_model() -> NeuralNetworkPotential:
    input_modules = [nn.Linear(1, 20)]
    representation = [nn.Linear(20, 20), nn.Linear(20, 10)]
    output_modules = [nn.Linear(10, 1)]
    return NeuralNetworkPotential(representation, input_modules, output_modules)


def test_base_class():
    setup_simple_model()


def test_forward_pass():
    model = setup_simple_model()
    t = torch.tensor([[1.0] * 10])
    model.forward({"inputs": t})

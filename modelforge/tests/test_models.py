from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.potential.models import NeuralNetworkPotential
from modelforge.potential.schnet import Schnet
from modelforge.utils import Inputs
import torch
import torch.nn as nn
from pytest import fixture


@fixture
def initialize_dataloader() -> torch.utils.data.DataLoader:
    data = QM9Dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module.train_dataloader()


def setup_simple_model() -> NeuralNetworkPotential:
    return Schnet(n_atom_basis=128, n_interactions=3, n_filters=128)


def test_base_class(initialize_dataloader):
    schnet = setup_simple_model()

    train_loader = initialize_dataloader
    R, Z, E = train_loader.dataset[0]
    padded_values = -Z.eq(-1).sum().item()
    Z_ = Z[:padded_values]
    R_ = R[:padded_values]
    inputs = Inputs(Z_, R_)

    schnet.forward(inputs)


def test_forward_pass():
    model = setup_simple_model()
    t = torch.tensor([[1.0] * 10])
    model.forward({"inputs": t})

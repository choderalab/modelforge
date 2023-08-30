from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.potential.schnet import Schnet
from modelforge.utils import Inputs
import torch


def initialize_dataloader() -> torch.utils.data.DataLoader:
    data = QM9Dataset(for_unit_testing=True)
    data_module = TorchDataModule(data)
    data_module.prepare_data()
    data_module.setup("fit")
    return data_module.train_dataloader()


def setup_simple_model() -> Schnet:
    return Schnet(n_atom_basis=128, n_interactions=3, n_filters=128)


def default_input():
    train_loader = initialize_dataloader()
    R, Z, E = train_loader.dataset[0]
    padded_values = -Z.eq(-1).sum().item()
    Z_ = Z[:padded_values]
    R_ = R[:padded_values]
    return Inputs(Z_, R_, E)


def test_base_class():
    schnet = setup_simple_model()
    inputs = default_input()
    output = schnet.forward(inputs)
    print(output)


def test_forward_pass():
    model = setup_simple_model()
    t = torch.tensor([[1.0] * 10])
    model.forward({"inputs": t})

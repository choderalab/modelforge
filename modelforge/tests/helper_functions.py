import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential import SchNet, PaiNN, ANI2x, PhysNet, SAKE
from typing import Optional, Dict

MODELS_TO_TEST = [SchNet, PaiNN, ANI2x, PhysNet, SAKE]
DATASETS = [QM9Dataset]

from modelforge.potential.utils import BatchData


def return_single_batch(
    dataset, split_file: Optional[str] = None, for_unit_testing: bool = True
) -> BatchData:
    """
    Return a single batch from a dataset.

    Parameters
    ----------
    dataset : class
        Dataset class.
    Returns
    -------
    Dict[str, Tensor]
        A single batch from the dataset.
    """

    train_loader = initialize_dataset(dataset, split_file, for_unit_testing)
    for batch in train_loader.train_dataloader():
        return batch


def initialize_dataset(
    dataset, split_file: Optional[str] = None, for_unit_testing: bool = True
) -> TorchDataModule:
    """
    Initialize a dataset for a given mode.

    Parameters
    ----------
    dataset : class
        Dataset class.
    Returns
    -------
    TorchDataModule
        Initialized TorchDataModule.
    """

    data = dataset(for_unit_testing=for_unit_testing)
    data_module = TorchDataModule(data, split_file=split_file)
    data_module.prepare_data()
    return data_module


def prepare_pairlist_for_single_batch(
    batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Prepare pairlist for a single batch.

    Parameters
    ----------
    batch : Dict[str, Tensor]
        Batch with keys like 'R', 'Z', etc.

    Returns
    -------
    Dict[str, Tensor]
        Pairlist with keys 'atom_index12', 'd_ij', 'r_ij'.
    """

    from modelforge.potential.models import Pairlist

    positions = batch["positions"]
    atomic_subsystem_indices = batch["atomic_subsystem_indices"]
    pairlist = Pairlist()
    return pairlist(positions, atomic_subsystem_indices)


from modelforge.potential.utils import Metadata, NNPInput, BatchData


def generate_methane_input() -> BatchData:
    """
    Generate a methane molecule input for testing.

    Returns
    -------
    BatchData
    """
    from modelforge.potential.utils import Metadata, NNPInput, BatchData

    atomic_numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.int64)
    positions = (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.63918859, 0.63918859, 0.63918859],
                [-0.63918859, -0.63918859, 0.63918859],
                [-0.63918859, 0.63918859, -0.63918859],
                [0.63918859, -0.63918859, -0.63918859],
            ],
            requires_grad=True,
        )
        / 10  # NOTE: converting to nanometer
    )
    E = torch.tensor([0.0], requires_grad=True)
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    return BatchData(
        NNPInput(
            atomic_numbers=atomic_numbers,
            positions=positions,
            atomic_subsystem_indices=atomic_subsystem_indices,
            total_charge=torch.tensor([0.0]),
        ),
        Metadata(
            E=E,
            atomic_subsystem_counts=torch.tensor([0]),
            atomic_subsystem_indices_referencing_dataset=torch.tensor([0]),
            number_of_atoms=atomic_numbers.numel(),
        ),
    )


def generate_batch_data():
    return return_single_batch(QM9Dataset)


SIMPLIFIED_INPUT_DATA = [
    generate_methane_input(),
    generate_batch_data(),
]


import torch
import math


def equivariance_test_utils():
    """
    Generates random tensors for testing equivariance of a neural network.

    Returns:
        Tuple of tensors:
        - h0 (torch.Tensor): Random tensor of shape (number_of_atoms, hidden_features).
        - x0 (torch.Tensor): Random tensor of shape (number_of_atoms, 3).
        - v0 (torch.Tensor): Random tensor of shape (5, 3).
        - translation (function): Function that translates a tensor by a random 3D vector.
        - rotation (function): Function that rotates a tensor by a random 3D rotation matrix.
        - reflection (function): Function that reflects a tensor across a random 3D plane.
    """

    # Define translation function
    torch.manual_seed(12345)
    x_translation = torch.randn(
        size=(1, 3),
    )
    translation = lambda x: x + x_translation

    # Define rotation function
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample()

    rz = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 1],
        ]
    )

    ry = torch.tensor(
        [
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [-math.sin(beta), 0, math.cos(beta)],
        ]
    )

    rx = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(gamma), -math.sin(gamma)],
            [0, math.sin(gamma), math.cos(gamma)],
        ]
    )

    rotation = lambda x: x @ rz @ ry @ rx

    # Define reflection function
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample()
    v = torch.tensor([[alpha, beta, gamma]])
    v /= (v**2).sum() ** 0.5

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p

    return translation, rotation, reflection

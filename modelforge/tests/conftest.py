import torch
import pytest
from modelforge.dataset import TorchDataModule, _IMPLEMENTED_DATASETS
from typing import Optional, Dict
from modelforge.potential import NeuralNetworkPotentialFactory, _IMPLEMENTED_NNPS


_DATASETS_TO_TEST = [name for name in _IMPLEMENTED_DATASETS]
_MODELS_TO_TEST = [name for name in _IMPLEMENTED_NNPS]
from modelforge.potential.utils import BatchData


@pytest.fixture(params=_MODELS_TO_TEST)
def train_model(request):
    model_name = request.param
    # Assuming NeuralNetworkPotentialFactory.create_nnp
    model = NeuralNetworkPotentialFactory.create_nnp(
        use="training", nnp_name=model_name, simulation_environment="PyTorch"
    )
    return model


@pytest.fixture(params=_MODELS_TO_TEST)
def inference_model(request):
    model_name = request.param
    # simulation_environment needs to be taken from another parameter, not defined here.
    # Assuming you pass simulation_environment to create_nnp in some way
    return lambda env: NeuralNetworkPotentialFactory.create_nnp(
        use="inference",
        nnp_name=model_name,
        simulation_environment=env,
    )


@pytest.fixture(params=_DATASETS_TO_TEST)
def datasets_to_test(request):
    dataset_name = request.param
    if dataset_name == "QM9":
        from modelforge.dataset import QM9Dataset

        dataset = QM9Dataset(for_unit_testing=True)
        return dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")


@pytest.fixture(params=_DATASETS_TO_TEST)
def initialized_dataset(request):
    dataset_name = request.param
    if dataset_name == "QM9":
        from modelforge.dataset import QM9Dataset

        dataset = QM9Dataset(for_unit_testing=True)

    return initialize_dataset(dataset)


@pytest.fixture(params=_DATASETS_TO_TEST)
def batch(initialized_dataset, request):
    """
    Fixture to obtain a single batch from an initialized dataset.

    This fixture depends on the `initialized_dataset` fixture for the dataset instance.
    The `request` parameter is automatically provided by pytest but is not used directly in this fixture.
    """
    batch = return_single_batch(initialized_dataset)
    return batch


# Fixture for initializing QM9Dataset
@pytest.fixture
def qm9_dataset():
    from modelforge.dataset import QM9Dataset

    dataset = QM9Dataset(for_unit_testing=True)
    return dataset


# Fixture for generating simplified input data
@pytest.fixture(params=["methane", "qm9_batch"])
def simplified_input_data(request, qm9_batch):
    if request.param == "methane":
        return generate_methane_input()
    elif request.param == "qm9_batch":
        return qm9_batch


# Fixture for equivariance test utilities
@pytest.fixture
def equivariance_utils():
    return equivariance_test_utils()


# ----------------------------------------------------------- #
# helper functions
# ----------------------------------------------------------- #


def return_single_batch(data_module) -> BatchData:
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

    batch = next(iter(data_module.train_dataloader()))
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

    data_module = TorchDataModule(dataset, split_file=split_file)
    data_module.prepare_data()
    return data_module


from modelforge.potential.utils import Metadata, NNPInput, BatchData


@pytest.fixture
def methane() -> BatchData:
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
    # torch.manual_seed(12345)
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
        ],
        dtype=torch.float64,
    )

    ry = torch.tensor(
        [
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [-math.sin(beta), 0, math.cos(beta)],
        ],
        dtype=torch.float64,
    )

    rx = torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(gamma), -math.sin(gamma)],
            [0, math.sin(gamma), math.cos(gamma)],
        ],
        dtype=torch.float64,
    )

    rotation = lambda x: x @ rz @ ry @ rx

    # Define reflection function
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample()
    v = torch.tensor([[alpha, beta, gamma]], dtype=torch.float64)
    v /= (v**2).sum() ** 0.5

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p

    return translation, rotation, reflection

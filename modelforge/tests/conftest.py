import torch
import pytest
from modelforge.dataset import DataModule

from typing import Optional, Dict
from dataclasses import dataclass


from modelforge.potential.utils import BatchData


# datamodule fixture
@pytest.fixture
def datamodule_factory():
    def create_datamodule(**kwargs):
        return initialize_datamodule(**kwargs)

    return create_datamodule


from modelforge.dataset.utils import (
    FirstComeFirstServeSplittingStrategy,
    SplittingStrategy,
)


def initialize_datamodule(
    dataset_name: str,
    for_unit_testing: bool = True,
    batch_size: int = 64,
    splitting_strategy: SplittingStrategy = FirstComeFirstServeSplittingStrategy(),
    remove_self_energies: bool = True,
    regression_ase: bool = False,
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """

    data_module = DataModule(
        dataset_name,
        splitting_strategy=splitting_strategy,
        batch_size=batch_size,
        for_unit_testing=for_unit_testing,
        remove_self_energies=remove_self_energies,
        regression_ase=regression_ase,
    )
    data_module.prepare_data()
    data_module.setup()
    return data_module


# dataset fixture
@pytest.fixture
def dataset_factory():
    def create_dataset(**kwargs):
        return initialize_dataset(**kwargs)

    return create_dataset


from modelforge.dataset.dataset import DatasetFactory, TorchDataset
from modelforge.dataset import _ImplementedDatasets


def single_batch(batch_size: int = 64):
    """
    Utility function to create a single batch of data for testing.
    """
    data_module = initialize_datamodule(
        dataset_name="QM9",
        batch_size=batch_size,
        for_unit_testing=True,
    )
    return next(iter(data_module.train_dataloader()))


@pytest.fixture(scope="session")
def single_batch_with_batchsize_64():
    """
    Utility fixture to create a single batch of data for testing.
    """
    return single_batch(batch_size=64)


@pytest.fixture(scope="session")
def single_batch_with_batchsize_1():
    """
    Utility fixture to create a single batch of data for testing.
    """
    return single_batch(batch_size=1)


def initialize_dataset(
    dataset_name: str,
    local_cache_dir: str,
    for_unit_testing: bool = True,
    force_download: bool = False,
) -> DataModule:
    """
    Initialize a dataset for a given mode.
    """

    factory = DatasetFactory()
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        local_cache_dir=local_cache_dir,
        for_unit_testing=for_unit_testing,
        force_download=force_download,
    )
    dataset = factory.create_dataset(data)

    return dataset


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    import uuid

    filename = str(uuid.uuid4())

    tmp_path_factory.mktemp(f"dataset_test/")
    return f"dataset_test"


@dataclass
class DataSetContainer:
    name: str
    expected_properties_of_interest: list
    expected_E_random_split: float
    expected_E_fcfs_split: float


from typing import Dict

from modelforge.dataset import _ImplementedDatasets


dataset_container: Dict[str, DataSetContainer] = {
    "QM9": DataSetContainer(
        name="QM9",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "internal_energy_at_0K",
            "charges",
        ],
        expected_E_random_split=-622027.790147837,
        expected_E_fcfs_split=-106277.4161215308,
    ),
    "ANI1X": DataSetContainer(
        name="ANI1x",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "wb97x_dz.energy",
            "wb97x_dz.forces",
        ],
        expected_E_random_split=-1652066.552014041,
        expected_E_fcfs_split=-1015736.8142089575,
    ),
    "ANI2x": DataSetContainer(
        name="ANI2x",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "energies",
            "forces",
        ],
        expected_E_random_split=-148410.43286007023,
        expected_E_fcfs_split=-2096692.258327173,
    ),
    "SPICE114": DataSetContainer(
        name="SPICE114",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "mbis_charges",
        ],
        expected_E_random_split=-1922185.3358204272,
        expected_E_fcfs_split=-972574.265833225,
    ),
    "SPICE2": DataSetContainer(
        name="SPICE2",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "mbis_charges",
        ],
        expected_E_random_split=-5844365.936898948,
        expected_E_fcfs_split=-3418985.278140791,
    ),
    "SPICE114_OPENFF": DataSetContainer(
        name="SPICE114_OPENFF",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "mbis_charges",
        ],
        expected_E_random_split=-2263605.616072006,
        expected_E_fcfs_split=-1516718.0904709378,
    ),
}


def get_dataset_container(dataset_name: str) -> DataSetContainer:
    datasetDC = dataset_container[dataset_name]
    return datasetDC


@pytest.fixture
def get_dataset_container_fix():

    return get_dataset_container


# Fixture for equivariance test utilities
@pytest.fixture
def equivariance_utils():
    return equivariance_test_utils()


# ----------------------------------------------------------- #
# helper functions
# ----------------------------------------------------------- #

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


def generate_uniform_quaternion(u=None):
    """
    Generates a uniform normalized quaternion.

    Adapted from numpy implementation in openmm-tools
    https://github.com/choderalab/openmmtools/blob/main/openmmtools/mcmc.py

    Parameters
    ----------
    u : torch.Tensor
        Tensor of shape (3,). Optional, default is None.
        If not provided, a random tensor is generated.

    References
    ----------
    [1] K. Shoemake. Uniform random rotations. In D. Kirk, editor,
    Graphics Gems III, pages 124-132. Academic, New York, 1992.
    [2] Described briefly here: http://planning.cs.uiuc.edu/node198.html
    """
    import torch

    if u is None:
        u = torch.rand(3)
    # import numpy for pi
    import numpy as np

    q = torch.tensor(
        [
            torch.sqrt(1 - u[0]) * torch.sin(2 * np.pi * u[1]),
            torch.sqrt(1 - u[0]) * torch.cos(2 * np.pi * u[1]),
            torch.sqrt(u[0]) * torch.sin(2 * np.pi * u[2]),
            torch.sqrt(u[0]) * torch.cos(2 * np.pi * u[2]),
        ]
    )
    return q


def rotation_matrix_from_quaternion(quaternion):
    """Compute a 3x3 rotation matrix from a given quaternion (4-vector).

    Adapted from the numpy implementation in openmm-tools

    https://github.com/choderalab/openmmtools/blob/main/openmmtools/mcmc.py

    Parameters
    ----------
    q : torch.Tensor
        Quaternion tensor of shape (4,).

    Returns
    -------
    torch.Tensor
        Rotation matrix tensor of shape (3, 3).

    References
    ----------
    [1] http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """

    w, x, y, z = quaternion.unbind()
    Nq = (quaternion**2).sum()  # Squared norm.
    if Nq > 0.0:
        s = 2.0 / Nq
    else:
        s = 0.0

    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z

    rotation_matrix = torch.tensor(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ],
        dtype=torch.float64,
    )
    return rotation_matrix


def apply_rotation_matrix(coordinates, rotation_matrix, use_center_of_mass=True):
    """
    Rotate the coordinates using the rotation matrix.

    Parameters
    ----------
    coordinates : torch.Tensor
        The coordinates to rotate.
    rotation_matrix : torch.Tensor
        The rotation matrix.
    use_center_of_mass : bool
        If True, the coordinates are rotated around the center of mass, not the origin.

    Returns
    -------
    torch.Tensor
        The rotated coordinates.
    """

    if use_center_of_mass:
        coordinates_com = torch.mean(coordinates, 0)
    else:
        coordinates_com = torch.zeros(3)

    coordinates_proposed = (
        torch.matmul(
            rotation_matrix, (coordinates - coordinates_com).transpose(0, -1)
        ).transpose(0, -1)
    ) + coordinates_com

    return coordinates_proposed


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
    # CRI: Let us manually seed the random number generator to ensure that we perfrom the same tests each time.
    # While our tests of translation and rotation should ALWAYS pass regardless of the seed,
    # if the code is correctly implemented, there may be instances where the tolerance we set is not
    # sufficient to pass the test, and without the workflow being deterministic, it may be hard to
    # debug if it is an underlying issue with the code or just the tolerance.

    torch.manual_seed(12345)
    x_translation = torch.randn(
        size=(1, 3),
    )
    translation = lambda x: x + x_translation

    # generate random quaternion and rotation matrix
    q = generate_uniform_quaternion()
    rotation_matrix = rotation_matrix_from_quaternion(q)

    rotation = lambda x: apply_rotation_matrix(x, rotation_matrix)

    # Define reflection function
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample()
    v = torch.tensor([[alpha, beta, gamma]], dtype=torch.float64)
    v /= (v**2).sum() ** 0.5

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p

    return translation, rotation, reflection

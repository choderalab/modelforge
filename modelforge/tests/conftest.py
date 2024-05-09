import torch
import pytest
from modelforge.dataset import TorchDataModule, _IMPLEMENTED_DATASETS
from modelforge.dataset.dataset import HDF5Dataset

from typing import Optional, Dict
from modelforge.potential import NeuralNetworkPotentialFactory, _IMPLEMENTED_NNPS
from dataclasses import dataclass

_DATASETS_TO_TEST = [name for name in _IMPLEMENTED_DATASETS]
_DATASETS_TO_TEST_QM9_ANI2X = ["QM9", "ANI2X"]
_DATASETS_TO_TEST_QM9 = ["QM9"]
_MODELS_TO_TEST = [name for name in _IMPLEMENTED_NNPS]
from modelforge.potential.utils import BatchData


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("dataset_testing")
    return fn


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


@dataclass
class DataSetContainer:
    dataset: HDF5Dataset
    name: str
    expected_properties_of_interest: list
    expected_E_random_split: float
    expected_E_fcfs_split: float


from typing import Dict

# This will store the cached datasets
dataset_cache: Dict[str, DataSetContainer] = {}


@pytest.fixture(scope="session", params=_DATASETS_TO_TEST_QM9)
def datasets_to_test(request, prep_temp_dir):
    dataset_name = request.param

    if datasetDC := dataset_cache.get(dataset_name, None):
        return datasetDC

    if dataset_name == "QM9":
        from modelforge.dataset.qm9 import QM9Dataset

        datasetDC = DataSetContainer(
            dataset=QM9Dataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "internal_energy_at_0K",
                "charges",
            ],
            expected_E_random_split=-622027.790147837,
            expected_E_fcfs_split=-106277.4161215308,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC
    elif dataset_name == "ANI1X":
        from modelforge.dataset.ani1x import ANI1xDataset

        datasetDC = DataSetContainer(
            dataset=ANI1xDataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "wb97x_dz.energy",
                "wb97x_dz.forces",
            ],
            expected_E_random_split=-1652066.552014041,
            expected_E_fcfs_split=-1015736.8142089575,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC
    elif dataset_name == "ANI2X":
        from modelforge.dataset.ani2x import ANI2xDataset

        datasetDC = DataSetContainer(
            dataset=ANI2xDataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "energies",
                "forces",
            ],
            expected_E_random_split=-148410.43286007023,
            expected_E_fcfs_split=-2096692.258327173,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC
    elif dataset_name == "SPICE114":
        from modelforge.dataset.spice114 import SPICE114Dataset

        datasetDC = DataSetContainer(
            dataset=SPICE114Dataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "dft_total_energy",
                "dft_total_force",
                "mbis_charges",
            ],
            expected_E_random_split=-1922185.3358204272,
            expected_E_fcfs_split=-972574.265833225,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC
    elif dataset_name == "SPICE2":
        from modelforge.dataset.spice2 import SPICE2Dataset

        datasetDC = DataSetContainer(
            dataset=SPICE2Dataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "dft_total_energy",
                "dft_total_force",
                "mbis_charges",
            ],
            expected_E_random_split=-5844365.936898948,
            expected_E_fcfs_split=-3418985.278140791,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC

    elif dataset_name == "SPICE114_OPENFF":
        from modelforge.dataset.spice114openff import SPICE114OpenFFDataset

        datasetDC = DataSetContainer(
            dataset=SPICE114OpenFFDataset(
                for_unit_testing=True, local_cache_dir=str(prep_temp_dir)
            ),
            name=dataset_name,
            expected_properties_of_interest=[
                "geometry",
                "atomic_numbers",
                "dft_total_energy",
                "dft_total_force",
                "mbis_charges",
            ],
            expected_E_random_split=-2263605.616072006,
            expected_E_fcfs_split=-1516718.0904709378,
        )
        dataset_cache[dataset_name] = datasetDC
        return datasetDC
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")


@pytest.fixture()
def initialized_dataset(datasets_to_test):
    dataset = datasets_to_test.dataset
    return initialize_dataset(dataset)


@pytest.fixture()
def batch(initialized_dataset):
    """py
    Fixture to obtain a single batch from an initialized dataset.

    This fixture depends on the `initialized_dataset` fixture for the dataset instance.
    The `request` parameter is automatically provided by pytest but is not used directly in this fixture.
    """
    batch = return_single_batch(initialized_dataset)
    return batch


@pytest.fixture(params=_DATASETS_TO_TEST_QM9_ANI2X)
def QM9_ANI2X_to_test(request, prep_temp_dir):
    dataset_name = request.param
    if dataset_name == "QM9":
        from modelforge.dataset.qm9 import QM9Dataset

        return QM9Dataset(for_unit_testing=True, local_cache_dir=str(prep_temp_dir))

    elif dataset_name == "ANI2X":
        from modelforge.dataset.ani2x import ANI2xDataset

        return ANI2xDataset(for_unit_testing=True, local_cache_dir=str(prep_temp_dir))


@pytest.fixture()
def initialized_QM9_ANI2X_dataset(QM9_ANI2X_to_test):
    return initialize_dataset(QM9_ANI2X_to_test)


@pytest.fixture()
def batch_QM9_ANI2x(initialized_QM9_ANI2X_dataset):
    batch = return_single_batch(initialized_QM9_ANI2X_dataset)
    return batch


# Fixture for setting up QM9Dataset
@pytest.fixture
def qm9_dataset(prep_temp_dir):
    from modelforge.dataset import QM9Dataset

    dataset = QM9Dataset(for_unit_testing=True, local_cache_dir=str(prep_temp_dir))
    return dataset


# fixture for initializing QM9Dataset
@pytest.fixture
def initialized_qm9_dataset(qm9_dataset):
    return initialize_dataset(qm9_dataset)


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
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # we need to use the first come first serve splitting strategy, as random is default
    # using random would make it hard to validate the expected values in the tests
    data_module = TorchDataModule(
        dataset,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        split_file=split_file,
        batch_size=2,
    )
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

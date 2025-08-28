from dataclasses import dataclass
from typing import Dict, Optional

import pytest
import torch

from modelforge.dataset import DataModule


# let us setup a few pytest options
def pytest_addoption(parser):
    parser.addoption(
        "--run_data_download",
        action="store_true",
        default=False,
        help="run slow data download tests",
    )


# create a temporary directory for storing the downloaded datasets
# so we do not need to re-download the datasets for each of the tests
# This is different from the standard fixture
# as we want this to create a single directory shared by all workers
# in the pytest-xdist parallel testing, rather than a separate directory for each worker
# see https://docs.pytest.org/en/6.2.x/how-to/xdist.html#making-fixtures-shared-between-workers
# for more details
@pytest.fixture(scope="session")
def dataset_temp_dir(tmp_path_factory):
    import os

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        fn = str(tmp_path_factory.getbasetemp().parent) + f"/dataset_dir"
        os.makedirs(fn, exist_ok=True)
    else:
        fn = tmp_path_factory.mktemp("dataset_dir")
    return fn


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_data_download"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_data_download = pytest.mark.skip(
        reason="need --run_data_download option to run"
    )
    for item in items:
        if "data_download" in item.keywords:
            item.add_marker(skip_data_download)


# datamodule fixture
@pytest.fixture
def datamodule_factory():
    def create_datamodule(
        dataset_name: str,
        batch_size: int,
        local_cache_dir: str,
        splitting_strategy,
        dataset_cache_dir: Optional[str] = None,
        remove_self_energies: Optional[bool] = True,
        element_filter: Optional[str] = None,
        local_yaml_file: Optional[str] = None,
        shift_center_of_mass_to_origin: bool = True,
        version_select: Optional[str] = None,
        regression_ase: Optional[bool] = False,
    ):

        # for simplicity of writing tests, we will have this accept the dataset name
        # and then use the toml files to load the properties of interests and association
        # and use the appropriate version of the dataset.
        # this will make it so we only have to change things in one spot if we have an
        # update to the test dataset.
        # note if version_select is not provided, we will use the test listed in the toml file in the test/data/datasets
        # directory.
        # The only reason to provide a version_select is if you want to test a specific version of the dataset
        # in the testing; e.g., the subset for spice so we can test on ani2x NNP

        from importlib import resources
        from modelforge.tests.data import dataset_defaults
        import toml
        import os

        toml_file = resources.files(dataset_defaults) / f"{dataset_name.lower()}.toml"

        # check to ensure the yaml file exists
        if not os.path.exists(toml_file):
            raise FileNotFoundError(
                f"Dataset toml file {toml_file} not found. Please check the dataset name."
            )

        config_dict = toml.load(toml_file)

        if dataset_cache_dir is None:
            dataset_cache_dir = local_cache_dir

        if version_select is None:
            version_select = config_dict["dataset"]["version_select"]
        return initialize_datamodule(
            dataset_name=dataset_name,
            splitting_strategy=splitting_strategy,
            batch_size=batch_size,
            version_select=version_select,
            properties_of_interest=config_dict["dataset"]["properties_of_interest"],
            properties_assignment=config_dict["dataset"]["properties_assignment"],
            local_cache_dir=local_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            remove_self_energies=remove_self_energies,
            element_filter=element_filter,
            local_yaml_file=local_yaml_file,
            shift_center_of_mass_to_origin=shift_center_of_mass_to_origin,
            regression_ase=regression_ase,
        )

    return create_datamodule


# dataset fixture
@pytest.fixture
def dataset_factory():
    def create_dataset(**kwargs):
        return initialize_dataset(**kwargs)

    return create_dataset


from modelforge.dataset.dataset import (
    initialize_datamodule,
    initialize_dataset,
    single_batch,
)


@pytest.fixture(scope="session")
def single_batch_with_batchsize():
    """
    Utility fixture to create a single batch of data for testing.
    """

    def _create_single_batch(
        batch_size: int,
        dataset_name: str,
        local_cache_dir: str,
        dataset_cache_dir: Optional[str] = None,
        version_select: Optional[str] = None,
    ):
        # for simplicity of writing tests, we will have this accept the dataset name
        # and then use the toml files to load the properties of interests and association
        # and use the appropriate version of the dataset.
        # this will make it so we only have to change things in one spot if we have an
        # update to the test dataset.

        from importlib import resources
        from modelforge.tests.data import dataset_defaults
        import toml
        import os

        toml_file = resources.files(dataset_defaults) / f"{dataset_name.lower()}.toml"

        # check to ensure the yaml file exists
        if not os.path.exists(toml_file):
            raise FileNotFoundError(
                f"Dataset toml file {toml_file} not found. Please check the dataset name."
            )

        config_dict = toml.load(toml_file)

        if dataset_cache_dir is None:
            dataset_cache_dir = local_cache_dir

        if version_select is None:
            version_select = config_dict["dataset"]["version_select"]
        return single_batch(
            batch_size=batch_size,
            dataset_name=dataset_name,
            local_cache_dir=local_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            version_select=version_select,
            properties_of_interest=config_dict["dataset"]["properties_of_interest"],
            properties_assignment=config_dict["dataset"]["properties_assignment"],
        )

    return _create_single_batch


@pytest.fixture(scope="session")
def load_test_dataset():
    """
    Fixture to load a test dataset.
    """

    def _load_test_dataset(
        dataset_name: str,
        local_cache_dir: str,
        dataset_cache_dir: Optional[str] = None,
        element_filter: Optional[list] = None,
    ):
        from modelforge.dataset import HDF5Dataset

        from importlib import resources
        from modelforge.tests.data import dataset_defaults
        import toml
        import os

        toml_file = resources.files(dataset_defaults) / f"{dataset_name.lower()}.toml"

        # check to ensure the yaml file exists
        if not os.path.exists(toml_file):
            raise FileNotFoundError(
                f"Dataset toml file {toml_file} not found. Please check the dataset name."
            )

        config_dict = toml.load(toml_file)
        if dataset_cache_dir is None:
            dataset_cache_dir = local_cache_dir

        return HDF5Dataset(
            dataset_name=dataset_name,
            force_download=False,
            version_select=config_dict["dataset"]["version_select"],
            properties_of_interest=config_dict["dataset"]["properties_of_interest"],
            properties_assignment=config_dict["dataset"]["properties_assignment"],
            local_cache_dir=local_cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            element_filter=element_filter,
        )

    return _load_test_dataset


# @pytest.fixture(scope="session")
# def prep_temp_dir(tmp_path_factory):
#     import uuid
#
#     filename = str(uuid.uuid4())
#
#     tmp_path_factory.mktemp(f"dataset_test/")
#     return f"dataset_test"


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
            "positions",
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
            "positions",
            "atomic_numbers",
            "wb97x_dz_energy",
            "wb97x_dz_forces",
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
    "PHALKETHOH": DataSetContainer(
        name="PHALKETHOH",
        expected_properties_of_interest=[
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "total_charge",
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

from modelforge.utils.prop import BatchData, NNPInput


@pytest.fixture
def methane() -> BatchData:
    """
    Generate a methane molecule input for testing.

    Returns
    -------
    BatchData
    """
    from modelforge.potential.utils import BatchData, Metadata

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
            per_system_total_charge=torch.tensor([0.0]),
        ),
        Metadata(
            E=E,
            atomic_subsystem_counts=torch.tensor([0]),
            atomic_subsystem_indices_referencing_dataset=torch.tensor([0]),
            number_of_atoms=atomic_numbers.numel(),
        ),
    )


import math

import torch


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

    Adapted from the numpy implementation in modelforgeopenmm-tools

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
        coordinates_com = torch.zeros(3).to(coordinates.device, coordinates.dtype)

    coordinates_proposed = (
        torch.matmul(
            rotation_matrix.to(coordinates.device, coordinates.dtype),
            (coordinates - coordinates_com).transpose(0, -1),
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
    translation = lambda x: x + x_translation.to(x.device, x.dtype)

    # generate random quaternion and rotation matrix
    q = generate_uniform_quaternion()
    rotation_matrix = rotation_matrix_from_quaternion(q)

    rotation = lambda x: apply_rotation_matrix(x, rotation_matrix.to(x.device, x.dtype))

    # Define reflection function
    alpha = torch.distributions.Uniform(-math.pi, math.pi).sample()
    beta = torch.distributions.Uniform(-math.pi, math.pi).sample()
    gamma = torch.distributions.Uniform(-math.pi, math.pi).sample()
    v = torch.tensor([[alpha, beta, gamma]], dtype=torch.float64)
    v /= (v**2).sum() ** 0.5

    p = torch.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p.to(x.device, x.dtype)

    return translation, rotation, reflection

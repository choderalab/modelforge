import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential.schnet import SchNET, LightningSchNET
from modelforge.potential.painn import PaiNN, LighningPaiNN
from modelforge.potential.models import BaseNNP

from typing import Optional, Dict

MODELS_TO_TEST = [SchNET, PaiNN]
DATASETS = [QM9Dataset]

from openmm import unit


def setup_simple_model(
    model_class,
    lightning: bool = False,
    nr_atom_basis: int = 128,
    max_atomic_number: int = 100,
    n_rbf: int = 20,
    cutoff: unit.Quantity = unit.Quantity(5.0, unit.angstrom),
    nr_interaction_blocks: int = 2,
    nr_filters: int = 2,
) -> Optional[BaseNNP]:
    """
    Setup a simple model based on the given model_class.

    Parameters
    ----------
    model_class : class
        Class of the model to be set up.
    lightning : bool, optional
        Flag to indicate if the Lightning variant should be returned.

    Returns
    -------
    Optional[BaseNNP]
        Initialized model.
    """
    from modelforge.potential import _CosineCutoff, _GaussianRBF
    from modelforge.potential.utils import SlicedEmbedding

    embedding = SlicedEmbedding(max_atomic_number, nr_atom_basis, sliced_dim=0)
    assert embedding.embedding_dim == nr_atom_basis
    rbf = _GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

    cutoff = _CosineCutoff(cutoff=cutoff)

    if model_class is SchNET:
        if lightning:
            return LightningSchNET(
                embedding=embedding,
                nr_interaction_blocks=nr_interaction_blocks,
                radial_basis=rbf,
                cutoff=cutoff,
                nr_filters=nr_filters,
            )
        return SchNET(
            embedding_module=embedding,
            nr_interaction_blocks=nr_interaction_blocks,
            radial_basis_module=rbf,
            cutoff_module=cutoff,
            nr_filters=nr_filters,
        )

    elif model_class is PaiNN:
        if lightning:
            return LighningPaiNN(
                embedding=embedding,
                nr_interaction_blocks=nr_interaction_blocks,
                radial_basis=rbf,
                cutoff=cutoff,
            )
        return PaiNN(
            embedding_module=embedding,
            nr_interaction_blocks=nr_interaction_blocks,
            radial_basis_module=rbf,
            cutoff_module=cutoff,
        )
    else:
        raise NotImplementedError


def return_single_batch(
    dataset, mode: str, split_file: Optional[str] = None, for_unit_testing: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Return a single batch from a dataset.

    Parameters
    ----------
    dataset : class
        Dataset class.
    mode : str
        Mode to setup the dataset ('fit', or 'test').

    Returns
    -------
    Dict[str, Tensor]
        A single batch from the dataset.
    """

    train_loader = initialize_dataset(dataset, mode, split_file, for_unit_testing)
    for batch in train_loader.train_dataloader():
        return batch


def initialize_dataset(
    dataset, mode: str, split_file: Optional[str] = None, for_unit_testing: bool = True
) -> TorchDataModule:
    """
    Initialize a dataset for a given mode.

    Parameters
    ----------
    dataset : class
        Dataset class.
    mode : str
        Mode to setup the dataset. Either "fit" for training/validation split
        or "test" for test split.

    Returns
    -------
    TorchDataModule
        Initialized TorchDataModule.
    """

    data = dataset(for_unit_testing=for_unit_testing)
    data_module = TorchDataModule(data, split_file=split_file)
    data_module.prepare_data()
    data_module.setup(stage=mode)
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

    from modelforge.potential.models import _PairList

    positions = batch["positions"]
    atomic_subsystem_indices = batch["atomic_subsystem_indices"]
    pairlist = _PairList()
    return pairlist(positions, atomic_subsystem_indices)


def generate_methane_input() -> Dict[str, torch.Tensor]:
    """
    Generate a methane molecule input for testing.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary with keys 'atomic_numbers', 'positions', 'atomic_subsystem_indices', 'E_labels'.
    """

    atomic_numbers = torch.tensor([[6], [1], [1], [1], [1]], dtype=torch.int64)
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
    E_labels = torch.tensor([0.0], requires_grad=True)
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "E_labels": E_labels,
        "atomic_subsystem_indices": atomic_subsystem_indices,
    }


def generate_mock_data():
    return {
        "atomic_numbers": torch.tensor([[1], [2], [2], [3]]),
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            requires_grad=True,
        ),
        "atomic_subsystem_indices": torch.tensor([0, 0, 1, 1]),
    }


def generate_batch_data():
    return return_single_batch(QM9Dataset, mode="fit")


def generate_interaction_block_data(
    nr_atom_basis: int, nr_embeddings: int, nr_rbf: int
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for testing the SchNet interaction block.

    Parameters
    ----------
    nr_atom_basis : int
        Number of atom basis.
    nr_embeddings : int
        Number of embeddings.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing tensors for the interaction block test.
    """

    import torch.nn as nn

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.potential import _GaussianRBF
    from modelforge.potential.utils import _distance_to_radial_basis

    embedding = nn.Embedding(nr_embeddings, nr_atom_basis, padding_idx=0)
    batch = return_single_batch(QM9Dataset, "fit")
    r = prepare_pairlist_for_single_batch(batch)
    radial_basis = _GaussianRBF(
        n_rbf=nr_rbf, cutoff=5.0, dtype=batch["positions"].dtype
    )

    d_ij = r["d_ij"]
    f_ij, rcut_ij = _distance_to_radial_basis(d_ij, radial_basis)
    return {
        "x": embedding(batch["atomic_numbers"].squeeze(dim=1)),
        "f_ij": f_ij,
        "pair_indices": r["pair_indices"],
        "rcut_ij": rcut_ij,
        "atomic_subsystem_indices": batch["atomic_subsystem_indices"],
    }


SIMPLIFIED_INPUT_DATA = [
    generate_methane_input(),
    generate_mock_data(),
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

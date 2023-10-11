import torch

from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential.schnet import Schnet, LighningSchnet
from modelforge.potential.pain import PaiNN, LighningPaiNN
from modelforge.potential.models import BaseNNP

from typing import Optional, Dict

MODELS_TO_TEST = [Schnet]
DATASETS = [QM9Dataset]


def setup_simple_model(model_class, lightning: bool = False) -> Optional[BaseNNP]:
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
    from modelforge.potential.utils import CosineCutoff

    if model_class is Schnet:
        if lightning:
            return LighningSchnet(n_atom_basis=32, n_interactions=3, n_filters=64)
        return Schnet(nr_atom_basis=32, nr_interactions=3, nr_filters=64)
    elif model_class is PaiNN:
        if lightning:
            return LighningPaiNN(
                n_atom_basis=32,
                n_interactions=3,
                n_rbf=16,
                cutoff_fn=CosineCutoff(5.0),
            )
        return PaiNN(
            nr_atom_basis=32, nr_interactions=3, n_rbf=16, cutoff_fn=CosineCutoff(5.0)
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
        Mode to setup the dataset ('train', 'val', or 'test').

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
        Mode to setup the dataset ('train', 'val', or 'test').

    Returns
    -------
    TorchDataModule
        Initialized TorchDataModule.
    """

    data = dataset(for_unit_testing=for_unit_testing)
    data_module = TorchDataModule(data, split_file=split_file)
    data_module.prepare_data()
    data_module.setup(mode)
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

    from modelforge.potential.models import PairList

    R = batch["R"]
    mask = batch["Z"] == 0
    pairlist = PairList(cutoff=5.0)
    return pairlist(mask, R)


def generate_methane_input() -> Dict[str, torch.Tensor]:
    """
    Generate a methane molecule input for testing.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary with keys 'Z', 'R', 'E'.
    """

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

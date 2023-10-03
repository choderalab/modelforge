from typing import Iterable, Optional
import torch

def train(
        model: torch.nn.Module,
        data: Iterable,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        loss_function: torch.nn.Module = torch.nn.MSELoss(),
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        scheduler_kwargs: Optional[dict]=None,
):
    """Train a model on a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.

    data : Iterable
        Iterable of tuples (inputs, targets).

    optimizer : torch.optim.Optimizer
        Optimizer to use.

    num_epochs : int, optional
        Number of epochs to train for.

    learning_rate : float, optional
        Learning rate to use.

    weight_decay : float, optional
        Weight decay to use.

    loss_function : torch.nn.Module, optional
        Loss function to use.

    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler to use.

    scheduler_kwargs : dict, optional
        Keyword arguments to pass to the scheduler.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    """

    return model
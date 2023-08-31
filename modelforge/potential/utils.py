import torch
from typing import Callable, Union
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
        y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Initialize the Dense layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bias : bool, optional
            If False, the layer will not adapt bias.
        activation : Callable or nn.Module, optional
            Activation function, default is None (Identity).
        dtype : torch.dtype, optional
            Data type for PyTorch tensors.
        device : torch.device, optional
            Device ("cpu" or "cuda") on which computations will be performed.
        """
        super().__init__(in_features, out_features, bias)

        # Initialize activation function
        self.activation = activation if activation is not None else nn.Identity()

        # Initialize weight matrix
        self.weight = nn.init.xavier_uniform_(self.weight).to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """

        y = F.linear(x, self.weight, self.bias)
        y = self.activation(y)
        return y


def gaussian_rbf(
    inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian radial basis function (RBF) transformation.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.
    offsets : torch.Tensor
        Offsets for Gaussian functions.
    widths : torch.Tensor
        Widths for Gaussian functions.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """

    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y.to(dtype=torch.float32)


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """
    Behler-style cosine cutoff function.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.
    cutoff : torch.Tensor
        Cutoff radius.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * np.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= input < cutoff
    return input_cut


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Compute shifted soft-plus activation function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """
    return nn.functional.softplus(x) - np.log(2.0)


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis functions (RBF).
    """

    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        start: float = 0.0,
        trainable: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Gaussian RBF layer.

        Parameters
        ----------
        n_rbf : int
            Number of radial basis functions.
        cutoff : float
            Cutoff distance for RBF.
        start : float, optional
            Starting distance for RBF, defaults to 0.0.
        trainable : bool, optional
            If True, widths and offsets are trainable parameters.
        device : torch.device, optional
            Device ("cpu" or "cuda") on which computations will be performed.
        dtype : torch.dtype, optional
            Data type for PyTorch tensors.

        """
        super().__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf, dtype=dtype, device=device)
        widths = torch.tensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset),
            device=device,
            dtype=dtype,
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return gaussian_rbf(inputs, self.offsets, self.widths)

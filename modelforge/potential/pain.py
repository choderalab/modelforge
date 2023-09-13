import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple

from .models import BaseNNP
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    cosine_cutoff,
    scatter_add,
)
import torch


class PaiNN(BaseNNP):
    def __init__(self, n_atom_basis: int, n_interactions: int, n_filters: int = 0):
        super().__init__()

    
from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class Properties:
    Z: str = "atomic_numbers"
    R: str = "positions"


@dataclass
class Inputs:
    Z: torch.Tensor
    R: torch.Tensor
    cell: Optional[torch.Tensor] = (None,)
    pbc: Optional[torch.Tensor] = (None,)

    def __post_init__(self):
        assert self.Z.shape[0] == self.R.shape[0]


@dataclass
class SpeciesEnergies:
    species: torch.Tensor
    energies: torch.Tensor

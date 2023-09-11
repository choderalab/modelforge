from dataclasses import dataclass
import torch
from typing import Optional
from loguru import logger


@dataclass
class PropertyNames:
    Z: str
    R: str
    E: str


@dataclass
class Inputs:
    Z: torch.Tensor
    R: torch.Tensor
    E: torch.Tensor
    cell: Optional[torch.Tensor] = (None,)
    pbc: Optional[torch.Tensor] = (None,)
    dtype: torch.dtype = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        assert self.Z.shape[0] == self.R.shape[0]
        logger.info(f"Transforming Z and R to {self.dtype}")
        self.Z = self.Z.to(self.device, torch.int32)
        self.R = self.R.to(self.device, self.dtype)


@dataclass
class SpeciesEnergies:
    species: torch.Tensor
    energies: torch.Tensor

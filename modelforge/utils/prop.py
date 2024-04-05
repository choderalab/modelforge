from dataclasses import dataclass
import torch
from typing import NamedTuple, Optional
from loguru import logger


@dataclass
class PropertyNames:
    Z: str
    R: str
    E: str
    F: Optional[str] = None
    Q: Optional[str] = None


class SpeciesEnergies(NamedTuple):
    species: torch.Tensor
    energies: torch.Tensor


class SpeciesAEV(NamedTuple):
    species: torch.Tensor
    aevs: torch.Tensor

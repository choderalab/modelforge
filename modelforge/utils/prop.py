"""
Module of dataclass definitions of properties.
"""

from dataclasses import dataclass
import torch
from typing import NamedTuple, Optional
from loguru import logger


@dataclass
class PropertyNames:
    atomic_numbers: str
    positions: str
    E: str
    F: Optional[str] = None
    total_charge: Optional[str] = None


class SpeciesEnergies(NamedTuple):
    species: torch.Tensor
    energies: torch.Tensor


class SpeciesAEV(NamedTuple):
    species: torch.Tensor
    aevs: torch.Tensor

from modelforge.utils import Inputs, Properties, SpeciesEnergies

import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Dict, List, Optional, Union, Callable
from ase.neighborlist import neighbor_list
from ase import Atoms
import numpy as np
from .models import BaseNNP
from .utils import GaussianRBF, shifted_softplus, cosine_cutoff, Dense, scatter_add
from torch import dtype


class Schnet(BaseNNP):
    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_filters: int = None,
        dtype: dtype = torch.float32,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(dtype, device)
        n_rbf = 20
        self.radial_basis = GaussianRBF(n_rbf=n_rbf, cutoff=5.0)
        self.cutoff = 5.0
        self.activation = shifted_softplus
        self.n_interactions = n_interactions

        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=self.activation),
            Dense(n_atom_basis, 1, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=self.activation),
            Dense(n_filters, n_filters),
        )

        max_z: int = 100
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

    def _compute_distances(self, inputs: Inputs):
        _atomic_numbers = torch.clone(inputs.Z)
        atomic_numbers = list(_atomic_numbers.detach().cpu().numpy())
        positions = list(inputs.R.detach().cpu().numpy())
        ase_atoms = Atoms(numbers=atomic_numbers, positions=positions)
        idx_i, idx_j, idx_S, r_ij = neighbor_list(
            "ijSD", ase_atoms, 5.0, self_interaction=False
        )
        r_ij = torch.from_numpy(r_ij)
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = cosine_cutoff(d_ij, self.cutoff)
        return f_ij, r_ij, idx_i, idx_j, rcut_ij

    def _interaction_block(self, inputs: Inputs, f_ij, r_ij, idx_i, idx_j, rcut_ij):
        # compute atom and pair features
        x = self.embedding(inputs.Z)
        idx_i = torch.from_numpy(idx_i).to(self.device, torch.int32)
        for i in range(self.n_interactions):
            x = self.in2f(x)

            Wij = self.filter_network(f_ij)
            Wij = Wij * rcut_ij[:, None]
            Wij = Wij.to(dtype=self.dtype)
            # continuous-filter convolution
            x_j = x[idx_j]
            x_ij = x_j * Wij
            x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
            v = self.f2out(x)
            x = x + v

        return x

    def calculate_energies_and_forces(self, inputs: Inputs) -> torch.Tensor:
        f_ij, r_ij, idx_i, idx_j, rcut_ij = self._compute_distances(inputs)
        x = self._interaction_block(inputs, f_ij, r_ij, idx_i, idx_j, rcut_ij)
        return x

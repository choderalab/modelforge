import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from openff.units import unit

from .models import CoreNetwork

if TYPE_CHECKING:
    from .models import PairListOutputs
    from modelforge.potential.utils import NNPInput

from modelforge.potential.utils import NeuralNetworkData
from icecream import ic


@dataclass
class SpookyNetNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass to structure the inputs specifically for SpookyNet-based neural network potentials, including the necessary
    geometric and chemical information, along with the radial symmetry function expansion (`f_ij`) and the cosine cutoff
    (`f_cutoff`) to accurately represent atomistic systems for energy predictions.

    Attributes
    ----------
    pair_indices : torch.Tensor
        A 2D tensor of shape [2, num_pairs], indicating the indices of atom pairs within a molecule or system.
    d_ij : torch.Tensor
        A 1D tensor containing the distances between each pair of atoms identified in `pair_indices`. Shape: [num_pairs, 1].
    r_ij : torch.Tensor
        A 2D tensor of shape [num_pairs, 3], representing the displacement vectors between each pair of atoms.
    number_of_atoms : int
        A integer indicating the number of atoms in the batch.
    positions : torch.Tensor
        A 2D tensor of shape [num_atoms, 3], representing the XYZ coordinates of each atom within the system.
    atomic_numbers : torch.Tensor
        A 1D tensor containing atomic numbers for each atom, used to identify the type of each atom in the system(s).
    atomic_subsystem_indices : torch.Tensor
        A 1D tensor mapping each atom to its respective subsystem or molecule, useful for systems involving multiple
        molecules or distinct subsystems.
    total_charge : torch.Tensor
        A tensor with the total charge of each system or molecule. Shape: [num_systems], where each entry corresponds
        to a distinct system or molecule.
    atomic_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived from atomic numbers.
        Shape: [num_atoms, embedding_dim], where `embedding_dim` is the dimensionality of the embedding vectors.
    charge_embedding : torch.Tensor
        A 2D tensor containing embeddings or features for each atom, derived from total charge.
        Shape: [num_atoms, embedding_dim], where `embedding_dim` is the dimensionality of the embedding vectors.
    f_ij : Optional[torch.Tensor]
        A tensor representing the radial symmetry function expansion of distances between atom pairs, capturing the
        local chemical environment. Shape: [num_pairs, number_of_atom_features], where `number_of_atom_features` is the dimensionality of
        the radial symmetry function expansion. This field will be populated after initialization.
    f_cutoff : Optional[torch.Tensor]
        A tensor representing the cosine cutoff function applied to the radial symmetry function expansion, ensuring
        that atom pair contributions diminish smoothly to zero at the cutoff radius. Shape: [num_pairs]. This field
        will be populated after initialization.

    Notes
    -----
    The `SpookyNetNeuralNetworkData` class is designed to encapsulate all necessary inputs for SpookyNet-based neural network
    potentials in a structured and type-safe manner, facilitating efficient and accurate processing of input data by
    the model. The inclusion of radial symmetry functions (`f_ij`) and cosine cutoff functions (`f_cutoff`) allows
    for a detailed and nuanced representation of the atomistic systems, crucial for the accurate prediction of system
    energies and properties.

    Examples
    --------
    >>> inputs = SpookyNetNeuralNetworkData(
    ...     pair_indices=torch.tensor([[0, 1], [0, 2], [1, 2]]),
    ...     d_ij=torch.tensor([1.0, 1.0, 1.0]),
    ...     r_ij=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ...     number_of_atoms=3,
    ...     positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ...     atomic_numbers=torch.tensor([1, 6, 8]),
    ...     atomic_subsystem_indices=torch.tensor([0, 0, 0]),
    ...     total_charge=torch.tensor([0.0]),
    ...     atomic_embedding=torch.randn(3, 5),  # Example atomic embeddings
    ...     f_ij=torch.randn(3, 4),  # Example radial symmetry function expansion
    ...     f_cutoff=torch.tensor([0.5, 0.5, 0.5])  # Example cosine cutoff function
    ... )
    """

    atomic_embedding: torch.Tensor
    charge_embedding: torch.Tensor
    f_ij: Optional[torch.Tensor] = field(default=None)
    f_cutoff: Optional[torch.Tensor] = field(default=None)


class SpookyNetCore(CoreNetwork):
    def __init__(
        self,
            max_Z,
            cutoff: unit.Quantity,
            number_of_atom_features,
            number_of_radial_basis_functions,
            number_of_interaction_modules,
            number_of_residual_blocks,
    ) -> None:
        """
        Initialize the SpookyNet class.

        Parameters
        ----------
        max_Z : int, default=87
            Maximum atomic number to be embedded.
        number_of_atom_features : int, default=64
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions:int, default=16
        number_of_interaction_modules : int, default=2
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        """
        from .utils import Dense, ShiftedSoftplus

        log.debug("Initializing SpookyNet model.")
        super().__init__()
        self.number_of_atom_features = number_of_atom_features
        self.number_of_radial_basis_functions = number_of_radial_basis_functions

        assert max_Z <= 87
        self.atomic_embedding_module = SpookyNetAtomicEmbedding(
            number_of_atom_features, max_Z
        )
        self.charge_embedding_module = ElectronicEmbedding(
            number_of_atom_features, number_of_residual_blocks
        )

        # initialize representation block
        self.spookynet_representation_module = SpookyNetRepresentation(
            cutoff, number_of_radial_basis_functions
        )

        ic(number_of_interaction_modules)
        # Intialize interaction blocks
        self.interaction_modules = nn.ModuleList(
            [
                SpookyNetInteractionModule(
                    number_of_atom_features=number_of_atom_features,
                    number_of_radial_basis_functions=number_of_radial_basis_functions,
                    num_residual_pre=number_of_residual_blocks,
                    num_residual_local_x=number_of_residual_blocks,
                    num_residual_local_s=number_of_residual_blocks,
                    num_residual_local_p=number_of_residual_blocks,
                    num_residual_local_d=number_of_residual_blocks,
                    num_residual_local=number_of_residual_blocks,
                    num_residual_nonlocal_q=number_of_residual_blocks,
                    num_residual_nonlocal_k=number_of_residual_blocks,
                    num_residual_nonlocal_v=number_of_residual_blocks,
                    num_residual_post=number_of_residual_blocks,
                    num_residual_output=number_of_residual_blocks,
                )
                for _ in range(number_of_interaction_modules)
            ]
        )

        # final output layer
        self.energy_and_charge_readout = nn.Sequential(
            Dense(
                number_of_atom_features,
                2,
                activation=None,
                bias=False,
            ),
        )

        # learnable shift and bias that is applied per-element to ech atomic energy
        self.atomic_shift = nn.Parameter(torch.zeros(max_Z, 2))

    def _model_specific_input_preparation(
        self, data: "NNPInput", pairlist_output: "PairListOutputs"
    ) -> SpookyNetNeuralNetworkData:
        number_of_atoms = data.atomic_numbers.shape[0]

        atomic_embedding = self.atomic_embedding_module(data.atomic_numbers)

        charge_embedding = self.charge_embedding_module(
            atomic_embedding, data.total_charge, data.atomic_subsystem_indices
        )

        nnp_input = SpookyNetNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
            atomic_embedding=atomic_embedding,
            charge_embedding=charge_embedding,
        )

        return nnp_input

    def compute_properties(
        self, data: SpookyNetNeuralNetworkData
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        data : NamedTuple

        Returns
        -------
        Dict[str, torch.Tensor]
            Calculated energies; shape (nr_systems,).
        """

        # Compute the representation for each atom (transform to radial basis set, multiply by cutoff)
        representation = self.spookynet_representation_module(data.d_ij, data.r_ij)
        x = data.atomic_embedding + data.charge_embedding

        f = x.new_zeros(x.size())  # initialize output features to zero
        # Iterate over interaction blocks to update features
        for interaction in self.interaction_modules:
            x, y = interaction(
                x=x,
                pair_indices=data.pair_indices,
                filters=representation["filters"],
                dir_ij=representation["dir_ij"],
                d_orbital_ij=representation["d_orbital_ij"],
            )
            f += y  # accumulate module output to features

        per_atom_energy_and_charge = self.energy_and_charge_readout(x)

        per_atom_energy_and_charge_shifted = self.atomic_shift[data.atomic_numbers] + per_atom_energy_and_charge

        E_i = per_atom_energy_and_charge_shifted[:, 0]  # shape(nr_of_atoms, 1)
        q_i = per_atom_energy_and_charge_shifted[:, 1]  # shape(nr_of_atoms, 1)

        output = {
            "per_atom_energy": E_i.contiguous(),  # reshape memory mapping for JAX/dlpack
            "q_i": q_i.contiguous(),
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
        }

        return output



from .models import InputPreparation, NNPInput, BaseNetwork


class SpookyNetAtomicEmbedding(nn.Module):

    def __init__(self, number_of_atom_features, max_Z):
        super().__init__()

        # fmt: off
        electron_config = np.array([
            #  Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p vs vp  vd  vf
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # n
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # H
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # He
            [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Li
            [4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # Be
            [5, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],  # B
            [6, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],  # C
            [7, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0],  # N
            [8, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],  # O
            [9, 2, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],  # F
            [10, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0],  # Ne
            [11, 2, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Na
            [12, 2, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # Mg
            [13, 2, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],  # Al
            [14, 2, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0],  # Si
            [15, 2, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0],  # P
            [16, 2, 2, 6, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0],  # S
            [17, 2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0],  # Cl
            [18, 2, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 0],  # Ar
            [19, 2, 2, 6, 2, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # K
            [20, 2, 2, 6, 2, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # Ca
            [21, 2, 2, 6, 2, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0],  # Sc
            [22, 2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0],  # Ti
            [23, 2, 2, 6, 2, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0],  # V
            [24, 2, 2, 6, 2, 6, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0],  # Cr
            [25, 2, 2, 6, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0],  # Mn
            [26, 2, 2, 6, 2, 6, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 6, 0],  # Fe
            [27, 2, 2, 6, 2, 6, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 7, 0],  # Co
            [28, 2, 2, 6, 2, 6, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 8, 0],  # Ni
            [29, 2, 2, 6, 2, 6, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0],  # Cu
            [30, 2, 2, 6, 2, 6, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 10, 0],  # Zn
            [31, 2, 2, 6, 2, 6, 2, 10, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 10, 0],  # Ga
            [32, 2, 2, 6, 2, 6, 2, 10, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 10, 0],  # Ge
            [33, 2, 2, 6, 2, 6, 2, 10, 3, 0, 0, 0, 0, 0, 0, 0, 2, 3, 10, 0],  # As
            [34, 2, 2, 6, 2, 6, 2, 10, 4, 0, 0, 0, 0, 0, 0, 0, 2, 4, 10, 0],  # Se
            [35, 2, 2, 6, 2, 6, 2, 10, 5, 0, 0, 0, 0, 0, 0, 0, 2, 5, 10, 0],  # Br
            [36, 2, 2, 6, 2, 6, 2, 10, 6, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 0],  # Kr
            [37, 2, 2, 6, 2, 6, 2, 10, 6, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Rb
            [38, 2, 2, 6, 2, 6, 2, 10, 6, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # Sr
            [39, 2, 2, 6, 2, 6, 2, 10, 6, 2, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0],  # Y
            [40, 2, 2, 6, 2, 6, 2, 10, 6, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0],  # Zr
            [41, 2, 2, 6, 2, 6, 2, 10, 6, 1, 4, 0, 0, 0, 0, 0, 1, 0, 4, 0],  # Nb
            [42, 2, 2, 6, 2, 6, 2, 10, 6, 1, 5, 0, 0, 0, 0, 0, 1, 0, 5, 0],  # Mo
            [43, 2, 2, 6, 2, 6, 2, 10, 6, 2, 5, 0, 0, 0, 0, 0, 2, 0, 5, 0],  # Tc
            [44, 2, 2, 6, 2, 6, 2, 10, 6, 1, 7, 0, 0, 0, 0, 0, 1, 0, 7, 0],  # Ru
            [45, 2, 2, 6, 2, 6, 2, 10, 6, 1, 8, 0, 0, 0, 0, 0, 1, 0, 8, 0],  # Rh
            [46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0],  # Pd
            [47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0, 0, 0, 0, 1, 0, 10, 0],  # Ag
            [48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0, 0, 0, 0, 2, 0, 10, 0],  # Cd
            [49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0, 0, 0, 0, 2, 1, 10, 0],  # In
            [50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0, 0, 0, 0, 2, 2, 10, 0],  # Sn
            [51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0, 0, 0, 0, 2, 3, 10, 0],  # Sb
            [52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0, 0, 0, 0, 2, 4, 10, 0],  # Te
            [53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0, 0, 0, 0, 2, 5, 10, 0],  # I
            [54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 0, 0, 0, 2, 6, 10, 0],  # Xe
            [55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 0, 0, 0, 1, 0, 0, 0],  # Cs
            [56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 0, 0, 2, 0, 0, 0],  # Ba
            [57, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 0, 1, 0, 2, 0, 1, 0],  # La
            [58, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 1, 1, 0, 2, 0, 1, 1],  # Ce
            [59, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 3, 0, 0, 2, 0, 0, 3],  # Pr
            [60, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 4, 0, 0, 2, 0, 0, 4],  # Nd
            [61, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 5, 0, 0, 2, 0, 0, 5],  # Pm
            [62, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 6, 0, 0, 2, 0, 0, 6],  # Sm
            [63, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 0, 0, 2, 0, 0, 7],  # Eu
            [64, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 7, 1, 0, 2, 0, 1, 7],  # Gd
            [65, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 9, 0, 0, 2, 0, 0, 9],  # Tb
            [66, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10, 0, 0, 2, 0, 0, 10],  # Dy
            [67, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11, 0, 0, 2, 0, 0, 11],  # Ho
            [68, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12, 0, 0, 2, 0, 0, 12],  # Er
            [69, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13, 0, 0, 2, 0, 0, 13],  # Tm
            [70, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 0, 0, 2, 0, 0, 14],  # Yb
            [71, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 1, 0, 2, 0, 1, 14],  # Lu
            [72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 2, 0, 2, 0, 2, 14],  # Hf
            [73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 3, 0, 2, 0, 3, 14],  # Ta
            [74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 4, 0, 2, 0, 4, 14],  # W
            [75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 5, 0, 2, 0, 5, 14],  # Re
            [76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 6, 0, 2, 0, 6, 14],  # Os
            [77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 7, 0, 2, 0, 7, 14],  # Ir
            [78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 9, 0, 1, 0, 9, 14],  # Pt
            [79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 1, 0, 10, 14],  # Au
            [80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14],  # Hg
            [81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 2, 1, 10, 14],  # Tl
            [82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 2, 2, 10, 14],  # Pb
            [83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 2, 3, 10, 14],  # Bi
            [84, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 2, 4, 10, 14],  # Po
            [85, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 2, 5, 10, 14],  # At
            [86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14]  # Rn
        ], dtype=np.float64)
        # fmt: on
        # normalize entries (between 0.0 and 1.0)
        self.register_buffer(
            "electron_config",
            torch.tensor(electron_config / np.max(electron_config, axis=0)),
        )
        self.element_embedding = nn.Embedding(max_Z, number_of_atom_features)
        self.register_parameter(
            "config_linear",
            nn.Parameter(
                torch.zeros((number_of_atom_features, self.electron_config.shape[1]))
            ),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.zeros_(self.element_embedding.weight)
        nn.init.zeros_(self.config_linear)

    def forward(self, atomic_numbers):
        return torch.einsum(
            "fe,ne->nf",
            self.config_linear,
            self.electron_config[atomic_numbers],
        ) + self.element_embedding(atomic_numbers)


class SpookyNet(BaseNetwork):
    def __init__(
        self,
        max_Z: int,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        number_of_interaction_modules: int,
        number_of_residual_blocks: int,
        cutoff: unit.Quantity,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the SpookyNet network.

        Unke, O.T., Chmiela, S., Gastegger, M. et al. SpookyNet: Learning force fields with electronic degrees of
        freedom and nonlocal effects. Nat Commun 12, 7273 (2021).

        Parameters
        ----------
        max_Z : int
            Maximum atomic number to be embedded.
        number_of_atom_features : int
            Dimension of the embedding vectors for atomic numbers.
        number_of_radial_basis_functions :int
        number_of_interaction_modules : int
        cutoff : openff.units.unit.Quantity
            The cutoff distance for interactions.
        """
        from modelforge.utils.units import _convert

        self.only_unique_pairs = False  # NOTE: need to be set before super().__init__

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            cutoff=_convert(cutoff),
        )
        from modelforge.utils.units import _convert

        self.core_module = SpookyNetCore(
            max_Z=max_Z,
            cutoff=_convert(cutoff),
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            number_of_interaction_modules=number_of_interaction_modules,
            number_of_residual_blocks=number_of_residual_blocks,
        )

    def _config_prior(self):
        log.info("Configuring SpookyNet model hyperparameter prior distribution")
        from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "cutoff": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior


class ElectronicEmbedding(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with the
    electrons.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_residual (int):
            TODO:
    """

    def __init__(
            self,
            num_features: int,
            num_residual: int,
    ) -> None:
        """Initializes the ElectronicEmbedding class."""
        super().__init__()
        self.linear_q = nn.Linear(num_features, num_features)
        # charges are duplicated to use separate weights for +/-
        self.linear_k = nn.Linear(2, num_features, bias=False)
        self.linear_v = nn.Linear(2, num_features, bias=False)
        self.resmlp = SpookyNetResidualMLP(
            num_features,
            num_residual,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.orthogonal_(self.linear_k.weight)
        nn.init.orthogonal_(self.linear_v.weight)
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

    def forward(
            self,
            x: torch.Tensor,
            E: torch.Tensor,
            atomic_subsystem_indices: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        ic(E.shape)
        q = self.linear_q(x)  # queries
        ic(q.shape)
        e = F.relu(torch.stack([E, -E], dim=-1))  # charges are duplicated to use separate weights for +/-
        ic(e.shape)
        k = self.linear_k(e / torch.clamp(e, min=1))[atomic_subsystem_indices]  # keys
        ic(k.shape)
        v = self.linear_v(e)[atomic_subsystem_indices]  # values
        ic(v.shape)
        dot = torch.einsum("nf,nf->n", k, q) / math.sqrt(k.shape[-1])  # scaled dot product
        a = nn.functional.softplus(dot)  # unnormalized attention weights
        a_normalized = a / (a.sum(-1) + eps)  # TODO: why is this needed? shouldn't softplus add up to 1?
        return self.resmlp(torch.einsum("n,nf->nf", a_normalized, v))


class SpookyNetRepresentation(nn.Module):

    def __init__(
        self,
        cutoff: unit = 5 * unit.angstrom,
        number_of_radial_basis_functions: int = 16,
    ):
        """
        Representation module for the PhysNet potential, handling the generation of
        the radial basis functions (RBFs) with a cutoff.

        Parameters
        ----------
        cutoff : openff.units.unit.Quantity, default=5*unit.angstrom
            The cutoff distance for interactions.
        number_of_radial_basis_functions : int, default=16
            Number of radial basis functions
        """

        super().__init__()

        # cutoff
        # radial symmetry function
        from .utils import ExponentialBernsteinRadialBasisFunction, CosineCutoff

        self.radial_symmetry_function_module = ExponentialBernsteinRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
        )

        self.cutoff_module = CosineCutoff(cutoff=cutoff)

    def forward(
        self, d_ij: torch.Tensor, r_ij: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the representation module.

        Parameters
        ----------
        d_ij : torch.Tensor
            pairwise distances between atoms, shape [num_pairs, 1].
        r_ij : torch.Tensor
            pairwise displacements between atoms, shape [num_pairs, 3].

        Returns
        -------
        torch.Tensor
            The radial basis function expansion applied to the input distances,
            shape (n_pairs, n_gaussians), after applying the cutoff function.
        """

        sqrt3 = math.sqrt(3)
        sqrt3half = 0.5 * sqrt3
        # short-range distances
        dir_ij = r_ij / d_ij
        d_orbital_ij = torch.stack(
            [
                sqrt3 * dir_ij[:, 0] * dir_ij[:, 1],  # xy
                sqrt3 * dir_ij[:, 0] * dir_ij[:, 2],  # xz
                sqrt3 * dir_ij[:, 1] * dir_ij[:, 2],  # yz
                0.5 * (3 * dir_ij[:, 2] * dir_ij[:, 2] - 1.0),  # z2
                sqrt3half
                * (dir_ij[:, 0] * dir_ij[:, 0] - dir_ij[:, 1] * dir_ij[:, 1]),  # x2-y2
            ],
            dim=-1,
        )
        f_ij = self.radial_symmetry_function_module(d_ij)
        f_ij_cutoff = self.cutoff_module(d_ij)
        filters = (
            f_ij.broadcast_to(
                len(d_ij),
                self.radial_symmetry_function_module.radial_basis_function.number_of_radial_basis_functions,
            )
        ) * f_ij_cutoff

        return {"filters": filters, "dir_ij": dir_ij, "d_orbital_ij": d_orbital_ij}


class Swish(nn.Module):
    """
    Swish activation function with learnable feature-wise parameters:
    f(x) = alpha*x * sigmoid(beta*x)
    sigmoid(x) = 1/(1 + exp(-x))
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the "linear component".
        initial_beta (float):
            Initial "temperature" of the "sigmoid component". The default value
            of 1.702 has the effect of initializing swish to an approximation
            of the Gaussian Error Linear Unit (GELU) activation function from
            Hendrycks, Dan, and Gimpel, Kevin. "Gaussian error linear units
            (GELUs)."
    """

    def __init__(
        self,
        number_of_atom_features: int,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.702,
    ) -> None:
        """Initializes the Swish class."""
        super(Swish, self).__init__()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter(
            "alpha", nn.Parameter(torch.Tensor(number_of_atom_features))
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.Tensor(number_of_atom_features))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters alpha and beta."""
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, number_of_atom_features]):
                Input features.

        Returns:
            y (FloatTensor [:, number_of_atom_features]):
                Activated features.
        """
        return self.alpha * F.silu(self.beta * x)


class SpookyNetResidual(nn.Module):
    """
    Pre-activation residual block inspired by He, Kaiming, et al. "Identity
    mappings in deep residual networks.".

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
    """

    def __init__(
        self,
        number_of_atom_features: int,
        bias: bool = True,
    ) -> None:
        """Initializes the Residual class."""
        super(SpookyNetResidual, self).__init__()
        # initialize attributes
        self.activation1 = Swish(number_of_atom_features)
        self.linear1 = nn.Linear(
            number_of_atom_features, number_of_atom_features, bias=bias
        )
        self.activation2 = Swish(number_of_atom_features)
        self.linear2 = nn.Linear(
            number_of_atom_features, number_of_atom_features, bias=bias
        )
        self.reset_parameters(bias)

    def reset_parameters(self, bias: bool = True) -> None:
        """Initialize parameters to compute an identity mapping."""
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input atomic features.
        N: Number of atoms.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Input feature representations of atoms.

        Returns:
            y (FloatTensor [N, number_of_atom_features]):
                Output feature representations of atoms.
        """
        y = self.activation1(x)
        y = self.linear1(y)
        y = self.activation2(y)
        y = self.linear2(y)
        return x + y


class SpookyNetResidualStack(nn.Module):
    """
    Stack of num_blocks pre-activation residual blocks evaluated in sequence.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_residual_blocks (int):
            Number of residual blocks to be stacked in sequence.
    """

    def __init__(
        self,
        number_of_atom_features: int,
        number_of_residual_blocks: int,
        bias: bool = True,
    ) -> None:
        """Initializes the ResidualStack class."""
        super(SpookyNetResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [
                SpookyNetResidual(number_of_atom_features, bias)
                for _ in range(number_of_residual_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies all residual blocks to input features in sequence.
        N: Number of inputs.
        number_of_atom_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Input feature representations.

        Returns:
            y (FloatTensor [N, number_of_atom_features]):
                Output feature representations.
        """
        for residual in self.stack:
            x = residual(x)
        return x


class SpookyNetResidualMLP(nn.Module):
    def __init__(
        self,
        number_of_atom_features: int,
        number_of_residual_blocks: int,
        bias: bool = True,
    ) -> None:
        super(SpookyNetResidualMLP, self).__init__()
        self.residual = SpookyNetResidualStack(
            number_of_atom_features, number_of_residual_blocks, bias=bias
        )
        self.activation = Swish(number_of_atom_features)
        self.linear = nn.Linear(
            number_of_atom_features, number_of_atom_features, bias=bias
        )
        self.reset_parameters(bias)

    def reset_parameters(self, bias: bool = True) -> None:
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.residual(x)))


class SpookyNetLocalInteraction(nn.Module):
    """
    Block for updating atomic features through local interactions with
    neighboring atoms (message-passing) described in Eq. 12.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_radial_basis_functions (int):
            Number of radial basis functions.
        num_residual_local_x (int):
            Number of residual blocks applied to atomic features in resmlp_c in Eq. 12.
        num_residual_local_s (int):
            Number of residual blocks applied to atomic features in resmlp_s in Eq. 12.
        num_residual_local_p (int):
            Number of residual blocks applied to atomic features in resmlp_p in Eq. 12.
        num_residual_local_d (int):
            Number of residual blocks applied to atomic features in resmlp_d in Eq. 12.
        num_residual_local (int):
            Number of residual blocks applied to atomic features in resmlp_l in Eq. 12.
    """

    def __init__(
        self,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
            num_residual_local_x: int,
            num_residual_local_s: int,
            num_residual_local_p: int,
            num_residual_local_d: int,
            num_residual_local: int,
    ) -> None:
        """Initializes the LocalInteraction class."""
        super(SpookyNetLocalInteraction, self).__init__()
        self.radial_s = nn.Linear(
            number_of_radial_basis_functions, number_of_atom_features, bias=False
        )
        self.radial_p = nn.Linear(
            number_of_radial_basis_functions, number_of_atom_features, bias=False
        )
        self.radial_d = nn.Linear(
            number_of_radial_basis_functions, number_of_atom_features, bias=False
        )
        self.resmlp_x = SpookyNetResidualMLP(number_of_atom_features, num_residual_local_x)
        self.resmlp_s = SpookyNetResidualMLP(number_of_atom_features, num_residual_local_s)
        self.resmlp_p = SpookyNetResidualMLP(number_of_atom_features, num_residual_local_p)
        self.resmlp_d = SpookyNetResidualMLP(number_of_atom_features, num_residual_local_d)
        self.projection_p = nn.Linear(
            number_of_atom_features, 2 * number_of_atom_features, bias=False
        )
        self.projection_d = nn.Linear(
            number_of_atom_features, 2 * number_of_atom_features, bias=False
        )
        self.resmlp_l = SpookyNetResidualMLP(number_of_atom_features, num_residual_local)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.orthogonal_(self.radial_s.weight)
        nn.init.orthogonal_(self.radial_p.weight)
        nn.init.orthogonal_(self.radial_d.weight)
        nn.init.orthogonal_(self.projection_p.weight)
        nn.init.orthogonal_(self.projection_d.weight)

    def forward(
        self,
        x_tilde: torch.Tensor,
        f_ij_after_cutoff: torch.Tensor,
        dir_ij: torch.Tensor,
        d_orbital_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.
        P: Number of atom pairs.

        x (FloatTensor [N, number_of_atom_features]):
            Atomic feature vectors.
        rbf (FloatTensor [N, number_of_radial_basis_functions]):
            Values of the radial basis functions for the pairwise distances.
        dir_ij (TODO:):
            TODO:
        d_orbital_ij (TODO):
            TODO:
        idx_i (LongTensor [P]):
            Index of atom i for all atomic pairs ij. Each pair must be
            specified as both ij and ji.
        idx_j (LongTensor [P]):
            Same as idx_i, but for atom j.
        """
        # interaction functions
        gs = self.radial_s(f_ij_after_cutoff)
        # p: num_pairs, f: number_of_atomic_features, r: number_of_radial_basis_functions
        gp = torch.einsum("pf,pr->prf", self.radial_p(f_ij_after_cutoff), dir_ij)
        gd = torch.einsum("pf,pr->prf", self.radial_d(f_ij_after_cutoff), d_orbital_ij)
        # atom featurizations
        xx = self.resmlp_x(x_tilde)
        xs = self.resmlp_s(x_tilde)
        xp = self.resmlp_p(x_tilde)
        xd = self.resmlp_d(x_tilde)
        # collect neighbors
        xs = xs[idx_j]  # L=0
        xp = xp[idx_j]  # L=1
        xd = xd[idx_j]  # L=2
        # sum over neighbors
        pp = x_tilde.new_zeros(x_tilde.shape[0], dir_ij.shape[-1], x_tilde.shape[-1])
        dd = x_tilde.new_zeros(
            x_tilde.shape[0], d_orbital_ij.shape[-1], x_tilde.shape[-1]
        )
        s = xx.index_add(0, idx_i, torch.einsum("pf,pf->pf", gs, xs))  # L=0
        # p: num_pairs, x: 3 (geometry axis), f: number_of_atom_features
        p = pp.index_add_(0, idx_i, torch.einsum("pxf,pf->pxf", gp, xp))  # L=1
        d = dd.index_add_(0, idx_i, torch.einsum("pxf,pf->pxf", gd, xd))  # L=2
        # project tensorial features to scalars
        pa, pb = torch.split(self.projection_p(p), p.shape[-1], dim=-1)
        da, db = torch.split(self.projection_d(d), d.shape[-1], dim=-1)
        # n: number_of_atoms_in_system, x: 3 (geometry axis), f: number_of_atom_features
        ic(pa.shape)
        ic(f_ij_after_cutoff.shape)
        return self.resmlp_l(
            s
            + torch.einsum("nxf,nxf->nf", pa, pb)
            + torch.einsum("nxf,nxf->nf", da, db)
        )


class SpookyNetAttention(nn.Module):
    """
    Efficient (linear scaling) approximation for attention described in
    Choromanski, K., et al. "Rethinking Attention with Performers.".

    Arguments:
        dim_qk (int):
            Dimension of query/key vectors.
        num_random_features (int):
            Number of random features for approximating attention matrix. If
            this is 0, the exact attention matrix is computed.
    """

    def __init__(self, dim_qk: int, num_random_features: int) -> None:
        """Initializes the Attention class."""
        super(SpookyNetAttention, self).__init__()
        self.num_random_features = num_random_features
        omega = self._omega(num_random_features, dim_qk)
        self.register_buffer("omega", torch.tensor(omega))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """For compatibility with other modules."""
        pass

    def _omega(self, nrows: int, ncols: int) -> np.ndarray:
        """Return a (nrows x ncols) random feature matrix."""
        nblocks = int(nrows / ncols)
        blocks = []
        for i in range(nblocks):
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q))
        missing_rows = nrows - nblocks * ncols
        if missing_rows > 0:
            block = np.random.normal(size=(ncols, ncols))
            q, _ = np.linalg.qr(block)
            blocks.append(np.transpose(q)[:missing_rows])
        norm = np.linalg.norm(  # renormalize rows so they still follow N(0,1)
            np.random.normal(size=(nrows, ncols)), axis=1, keepdims=True
        )
        return (norm * np.vstack(blocks)).T

    def _phi(
        self,
        X: torch.Tensor,
        is_query: bool,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """Normalize X and project into random feature space."""
        d = X.shape[-1]
        m = self.omega.shape[-1]
        U = torch.matmul(X / d**0.25, self.omega)
        h = torch.sum(X**2, dim=-1, keepdim=True) / (2 * d**0.5)  # OLD
        # determine maximum (is subtracted to prevent numerical overflow)
        if is_query:
            maximum, _ = torch.max(U, dim=-1, keepdim=True)
        else:
            maximum = torch.max(U)
        return (torch.exp(U - h - maximum) + eps) / math.sqrt(m)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute attention for the given query, key and value vectors.
        N: Number of input values.
        dim_qk: Dimension of query/key vectors.
        dim_v: Dimension of value vectors.

        Arguments:
            Q (FloatTensor [N, dim_qk]):
                Matrix of N query vectors.
            K (FloatTensor [N, dim_qk]):
                Matrix of N key vectors.
            V (FloatTensor [N, dim_v]):
                Matrix of N value vectors.
            eps (float):
                Small constant to prevent numerical instability.
        Returns:
            y (FloatTensor [N, dim_v]):
                Attention-weighted sum of value vectors.
        """
        Q = self._phi(Q, True)  # random projection of Q
        K = self._phi(K, False)  # random projection of K
        ic(Q.shape)
        ic(K.shape)
        ic(V.shape)
        ic(torch.sum(K, 0, keepdim=True).T)
        norm = torch.einsum("nf,f->n", Q, torch.sum(K, dim=0)) + eps
        ic(norm.shape)
        # n: number of atoms, F: dim_qk, f: value features
        rv = torch.einsum("nF,nF,nf->nf", Q, K, V) / norm.unsqueeze(-1)
        ic((K.T @ V).shape)
        ic((Q @ (K.T @ V)).shape)
        ic(rv.shape)
        return rv


class SpookyNetNonlocalInteraction(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with all
    atoms.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        num_residual_q (int):
            Number of residual blocks for queries.
        num_residual_k (int):
            Number of residual blocks for keys.
        num_residual_v (int):
            Number of residual blocks for values.
    """

    def __init__(
        self,
        number_of_atom_features: int,
        num_residual_q: int,
        num_residual_k: int,
        num_residual_v: int,
    ) -> None:
        """Initializes the NonlocalInteraction class."""
        super(SpookyNetNonlocalInteraction, self).__init__()
        self.resmlp_q = SpookyNetResidualMLP(number_of_atom_features, num_residual_q)
        self.resmlp_k = SpookyNetResidualMLP(number_of_atom_features, num_residual_k)
        self.resmlp_v = SpookyNetResidualMLP(number_of_atom_features, num_residual_v)
        self.attention = SpookyNetAttention(
            dim_qk=number_of_atom_features, num_random_features=number_of_atom_features
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """For compatibility with other modules."""
        pass

    def forward(
        self,
        x_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, number_of_atom_features]):
            Atomic feature vectors.
        """
        q = self.resmlp_q(x_tilde)  # queries
        k = self.resmlp_k(x_tilde)  # keys
        v = self.resmlp_v(x_tilde)  # values
        return self.attention(q, k, v)


class SpookyNetInteractionModule(nn.Module):
    """
    InteractionModule of SpookyNet, which computes a single iteration.

    Arguments:
        number_of_atom_features (int):
            Dimensions of feature space.
        number_of_radial_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features before
            interaction with neighbouring atoms.
        num_residual_local_x (int):
            Number of residual blocks applied to atomic features in resmlp_c in Eq. 12.
        num_residual_local_s (int):
            Number of residual blocks applied to atomic features in resmlp_s in Eq. 12.
        num_residual_local_p (int):
            Number of residual blocks applied to atomic features in resmlp_p in Eq. 12.
        num_residual_local_d (int):
            Number of residual blocks applied to atomic features in resmlp_d in Eq. 12.
        num_residual_local (int):
            Number of residual blocks applied to atomic features in resmlp_l in Eq. 12.
        num_residual_nonlocal_q (int):
            Number of residual blocks for queries in nonlocal interactions.
        num_residual_nonlocal_k (int):
            Number of residual blocks for keys in nonlocal interactions.
        num_residual_nonlocal_v (int):
            Number of residual blocks for values in nonlocal interactions.
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms.
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch.
    """

    def __init__(
        self,
        number_of_atom_features: int,
        number_of_radial_basis_functions: int,
        num_residual_pre: int,
        num_residual_local_x: int,
        num_residual_local_s: int,
        num_residual_local_p: int,
        num_residual_local_d: int,
        num_residual_local: int,
        num_residual_nonlocal_q: int,
        num_residual_nonlocal_k: int,
        num_residual_nonlocal_v: int,
        num_residual_post: int,
        num_residual_output: int,
    ) -> None:
        """Initializes the InteractionModule class."""
        super(SpookyNetInteractionModule, self).__init__()
        # initialize modules
        self.local_interaction = SpookyNetLocalInteraction(
            number_of_atom_features=number_of_atom_features,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            num_residual_local_x=num_residual_local_x,
            num_residual_local_s=num_residual_local_s,
            num_residual_local_p=num_residual_local_p,
            num_residual_local_d=num_residual_local_d,
            num_residual_local=num_residual_local,
        )
        self.nonlocal_interaction = SpookyNetNonlocalInteraction(
            number_of_atom_features=number_of_atom_features,
            num_residual_q=num_residual_nonlocal_q,
            num_residual_k=num_residual_nonlocal_k,
            num_residual_v=num_residual_nonlocal_v,
        )

        self.residual_pre = SpookyNetResidualStack(
            number_of_atom_features, num_residual_pre
        )
        self.residual_post = SpookyNetResidualStack(
            number_of_atom_features, num_residual_post
        )
        self.resmlp = SpookyNetResidualMLP(
            number_of_atom_features, num_residual_output
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """For compatibility with other modules."""
        pass

    def forward(
        self,
        x: torch.Tensor,
            pair_indices: torch.Tensor,  # shape [n_pairs, 2]
        filters: torch.Tensor,  # shape [n_pairs, 1, number_of_radial_basis_functions] TODO: why the 1?
        dir_ij: torch.Tensor,  # shape [n_pairs, 1]
        d_orbital_ij: torch.Tensor,  # shape [n_pairs, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate all modules in the block.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            x (FloatTensor [N, number_of_atom_features]):
                Latent atomic feature vectors.
            pair_indices :
                Indices of atom pairs within the maximum interaction radius.
            filters:
                TODO:
            dir_ij (FloatTensor [P, 3]):
                Unit vectors pointing from atom i to atom j for all atomic pairs.
            d_orbital_ij (FloatTensor [P]):
                Distances between atom i and atom j for all atomic pairs.
        Returns:
            x (FloatTensor [N, number_of_atom_features]):
                Updated latent atomic feature vectors.
            y (FloatTensor [N, number_of_atom_features]):
                Contribution to output atomic features (environment
                descriptors).
        """
        idx_i, idx_j = pair_indices[0], pair_indices[1]
        x_tilde = self.residual_pre(x)
        del x
        l = self.local_interaction(
            x_tilde=x_tilde,
            f_ij_after_cutoff=filters,
            dir_ij=dir_ij,
            d_orbital_ij=d_orbital_ij,
            idx_i=idx_i,
            idx_j=idx_j,
        )
        n = self.nonlocal_interaction(x_tilde)
        x_updated = self.residual_post(x_tilde + l + n)
        del x_tilde
        return x_updated, self.resmlp(x_updated)

from torch._tensor import Tensor
import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP
from .utils import Dense
import torch
import torch.nn.functional as F
from openff.units import unit


class PaiNN(BaseNNP):
    """PaiNN - polarizable interaction neural network

    References:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        max_Z: int = 100,
        embedding_dimensions: int = 64,
        nr_interaction_blocks: int = 2,
        cutoff: unit.Quantity = 5 * unit.angstrom,
        number_of_gaussians_basis_functions: int = 16,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):

        log.debug("Initializing PaiNN model.")
        self.nr_interaction_blocks = nr_interaction_blocks
        self.nr_atom_basis = nr_atom_basis = embedding_dimensions
        self.only_unique_pairs = False  # NOTE: for pairlist
        self.shared_filters = shared_filters
        super().__init__(cutoff=cutoff)

        # embedding
        from modelforge.potential.utils import Embedding

        self.embedding_module = Embedding(max_Z, embedding_dimensions)

        # initialize the energy readout
        from .utils import EnergyReadout

        self.readout_module = EnergyReadout(embedding_dimensions)

        # initialize representation block
        self.representation_module = PaiNNRepresentation(
            cutoff,
            number_of_gaussians_basis_functions,
            nr_interaction_blocks,
            nr_atom_basis,
            shared_filters,
        )

        # initialize the interaction and mixing networks
        self.interaction_modules = nn.ModuleList(
            PaiNNInteraction(nr_atom_basis, activation=F.silu)
            for _ in range(nr_interaction_blocks)
        )
        self.mixing_modules = nn.ModuleList(
            PaiNNMixing(nr_atom_basis, activation=F.silu, epsilon=epsilon)
            for _ in range(nr_interaction_blocks)
        )

    def _readout(self, input: Dict[str, Tensor]):
        return self.readout_module(input)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding

        inputs["atomic_embedding"] = self.embedding_module(inputs["atomic_numbers"])
        return inputs

    def _forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        input : Dict[str, torch.Tensor]
            Dictionary containing pairlist information.
        atomic_embedding : torch.Tensor
            Tensor containing atomic number embeddings.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector representations.
        """

        # initialize filters, q and mu
        transformed_input = self.representation_module(inputs)

        filter_list = transformed_input["filters"]
        q = transformed_input["q"]
        mu = transformed_input["mu"]
        dir_ij = transformed_input["dir_ij"]

        for i, (interaction_mod, mixing_mod) in enumerate(
            zip(self.interaction_modules, self.mixing_modules)
        ):
            q, mu = interaction_mod(
                q,
                mu,
                filter_list[i],
                dir_ij,
                inputs["pair_indices"],
            )
            q, mu = mixing_mod(q, mu)

        # Use squeeze to remove dimensions of size 1
        q = q.squeeze(dim=1)

        return {
            "scalar_representation": q,
            "vector_representation": mu,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


from openff.units import unit


class PaiNNRepresentation(nn.Module):
    """PaiNN representation module"""

    def __init__(
        self,
        cutoff: unit = 5 * unit.angstrom,
        number_of_gaussians: int = 16,
        nr_interaction_blocks: int = 3,
        nr_atom_basis: int = 8,
        shared_filters: bool = False,
    ):
        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(cutoff)

        # radial symmetry function
        from .utils import RadialSymmetryFunction

        self.radial_symmetry_function_module = RadialSymmetryFunction(
            number_of_gaussians=number_of_gaussians,
            radial_cutoff=cutoff,
            ani_style=False,
            dtype=torch.float32,
        )

        # initialize the filter network
        if shared_filters:
            filter_net = Dense(
                number_of_gaussians,
                3 * nr_atom_basis,
            )

        else:
            filter_net = Dense(
                number_of_gaussians,
                nr_interaction_blocks * nr_atom_basis * 3,
                activation=None,
            )

        self.filter_net = filter_net

        self.shared_filters = shared_filters
        self.nr_interaction_blocks = nr_interaction_blocks
        self.nr_atom_basis = nr_atom_basis

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Transforms the input data for the PAInn potential model.

        Parameters
        ----------
        inputs (Dict[str, torch.Tensor]): A dictionary containing the input tensors.
            - "d_ij" (torch.Tensor): Pairwise distances between atoms. Shape: (n_pairs, 1, distance).
            - "r_ij" (torch.Tensor): Displacement vector between atoms. Shape: (n_pairs, 1, 3).
            - "atomic_embedding" (torch.Tensor): Embeddings of atomic numbers. Shape: (n_atoms, embedding_dim).

        Returns:
        ----------
        Dict[str, torch.Tensor]:
            A dictionary containing the transformed input tensors.
            - "mu" (torch.Tensor)
                Zero-initialized tensor for atom features. Shape: (n_atoms, 3, nr_atom_basis).
            - "dir_ij" (torch.Tensor)
                Direction vectors between atoms. Shape: (n_pairs, 1, distance).
            - "q" (torch.Tensor): Reshaped atomic number embeddings. Shape: (n_atoms, 1, embedding_dim).
        """

        # compute pairwise distances
        d_ij = inputs["d_ij"]
        r_ij = inputs["r_ij"]
        dir_ij = r_ij / d_ij

        f_ij = self.radial_symmetry_function_module(d_ij)

        fcut = self.cutoff_module(d_ij)

        filters = self.filter_net(f_ij) * fcut[..., None]

        if self.shared_filters:
            filter_list = [filters] * self.nr_interaction_blocks
        else:
            filter_list = torch.split(filters, 3 * self.nr_atom_basis, dim=-1)

        # generate q and mu
        atomic_embedding = inputs["atomic_embedding"]
        qs = atomic_embedding.shape

        q = atomic_embedding[:, None]
        qs = q.shape
        mu = torch.zeros(
            (qs[0], 3, qs[2]), device=q.device
        )  # total_number_of_atoms_in_the_batch, 3, nr_atom_basis

        return {"filters": filter_list, "dir_ij": dir_ij, "q": q, "mu": mu}


class PaiNNInteraction(nn.Module):
    """
    PaiNN Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

    """

    def __init__(self, nr_atom_basis: int, activation: Callable):
        """
        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation : Callable
            Activation function to use.

        Attributes
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        interatomic_net : nn.Sequential
            Neural network for interatomic interactions.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # Initialize the intra-atomic neural network
        self.interatomic_net = nn.Sequential(
            Dense(nr_atom_basis, nr_atom_basis, activation=activation),
            Dense(nr_atom_basis, 3 * nr_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,  # shape [nr_of_atoms_in_batch, nr_atom_basis]
        mu: torch.Tensor,  # shape [n_mols, n_interactions, nr_atom_basis]
        Wij: torch.Tensor,  # shape [n_interactions]
        dir_ij: torch.Tensor,
        pairlist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction output.

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values of shape [nr_of_atoms_in_systems, nr_atom_basis].
        mu : torch.Tensor
            Vector input values of shape [n_mols, n_interactions, nr_atom_basis].
        Wij : torch.Tensor
            Filter of shape [n_interactions].
        dir_ij : torch.Tensor
            Directional vector between atoms i and j.
        pairlist : torch.Tensor, shape (2, n_pairs)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        # inter-atomic
        idx_i, idx_j = pairlist[0], pairlist[1]

        x = self.interatomic_net(q)
        nr_of_atoms_in_all_systems, _, _ = q.shape  # [nr_systems,n_atoms,96]
        x = x.reshape(nr_of_atoms_in_all_systems, 1, 3 * self.nr_atom_basis)

        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.nr_atom_basis, dim=-1)

        ##############
        # Expand the dimensions of idx_i to match that of dq
        expanded_idx_i = idx_i.view(-1, 1, 1).expand_as(dq)

        # Create a zero tensor with appropriate shape
        zeros = torch.zeros(
            nr_of_atoms_in_all_systems,
            1,
            self.nr_atom_basis,
            dtype=dq.dtype,
            device=dq.device,
        )

        dq = zeros.scatter_add(0, expanded_idx_i, dq)
        ##########
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        zeros = torch.zeros(
            nr_of_atoms_in_all_systems,
            3,
            self.nr_atom_basis,
            dtype=dmu.dtype,
            device=dmu.device,
        )
        expanded_idx_i_dmu = idx_i.view(-1, 1, 1).expand_as(dmu)
        dmu = zeros.scatter_add(0, expanded_idx_i_dmu, dmu)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, nr_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation : Callable
            Activation function to use.
        epsilon : float, optional
            Stability constant added in norm to prevent numerical instabilities. Default is 1e-8.

        Attributes
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        intra_atomic_net : nn.Sequential
            Neural network for intra-atomic interactions.
        mu_channel_mix : nn.Sequential
            Neural network for mixing mu channels.
        epsilon : float
            Stability constant for numerical stability.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # initialize the intra-atomic neural network
        self.intra_atomic_net = nn.Sequential(
            Dense(2 * nr_atom_basis, nr_atom_basis, activation=activation),
            Dense(nr_atom_basis, 3 * nr_atom_basis, activation=None),
        )
        # initialize the mu channel mixing network
        self.mu_channel_mix = Dense(nr_atom_basis, 2 * nr_atom_basis, bias=False)
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """
        compute intratomic mixing

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values.
        mu : torch.Tensor
            Vector input values.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.nr_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intra_atomic_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.nr_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu

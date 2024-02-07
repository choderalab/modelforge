from torch._tensor import Tensor
import torch.nn as nn
from loguru import logger as log
from typing import Dict, Type, Callable, Optional, Tuple

from .models import BaseNNP, LightningModuleMixin
from .utils import Dense
import torch
import torch.nn.functional as F


class SAKE(BaseNNP):
    """SAKE - spatial attention kinetic networks with E(n) equivariance.

    References:
        https://openreview.net/pdf?id=3DIpIf3wQMC

    """

    def __init__(
        self,
        embedding_module: nn.Module,
        nr_interaction_blocks: int,
        radial_basis_module: nn.Module,
        cutoff_module: nn.Module,
        activation: Optional[Callable] = F.silu,
    ):
        """
        Parameters
            ----------
            embedding_module : torch.Module, contains atomic species embedding.
                Embedding dimensions also define self.nr_atom_basis.
            nr_interaction_blocks : int
                Number of interaction blocks.
            radial_basis_module : torch.Module
                radial basis functions.
            cutoff_module : torch.Module
                Cutoff function for the radial basis.
            activation : Callable, optional
                Activation function to use.
            epsilon : float, optional
                Stability constant to prevent numerical instabilities (default is 1e-8).
        """
        from .utils import EnergyReadout

        log.debug("Initializing SAKE model.")
        super().__init__(cutoff=cutoff_module.cutoff)
        self.nr_interaction_blocks = nr_interaction_blocks
        self.cutoff_module = cutoff_module
        self.radial_basis_module = radial_basis_module

        # initialize the energy readout
        self.nr_atom_basis = embedding_module.embedding_dim
        self.readout_module = EnergyReadout(self.nr_atom_basis)

        log.debug(
            f"Passed parameters to constructor: {self.nr_atom_basis=}, {nr_interaction_blocks=}, {cutoff_module=}"
        )
        log.debug(f"Initialized embedding: {embedding_module=}")

        # initialize the filter network
        self.filter_net = nn.Sequential(
            nn.Linear(
                self.radial_basis_module.n_rbf,
                self.nr_interaction_blocks * 3 * self.nr_atom_basis,
            ),
            nn.Identity(),
        )

        # initialize the interaction networks
        self.interaction_modules = nn.ModuleList(
            SAKEInteraction(self.nr_atom_basis, activation=activation)
            for _ in range(self.nr_interaction_blocks)
        )

        # save the embedding
        self.embedding_module = embedding_module

    def _readout(self, input: Dict[str, Tensor]):
        return self.readout_module(input)

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        inputs = self._prepare_inputs(inputs)
        return self._model_specific_input_preparation(inputs)

    def _model_specific_input_preparation(self, inputs: Dict[str, torch.Tensor]):
        # Perform atomic embedding
        from modelforge.potential.utils import embed_atom_features

        atomic_embedding = embed_atom_features(
            inputs["atomic_numbers"], self.embedding_module
        )
        inputs["atomic_embedding"] = atomic_embedding
        return inputs

    def _generate_representation(self, inputs: Dict[str, torch.Tensor]):
        """
        Transforms the input data for the SAKE potential model.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing the input tensors.
                - "d_ij" (torch.Tensor): Pairwise distances between atoms. Shape: (n_pairs, 1, distance).
                - "r_ij" (torch.Tensor): Displacement vector between atoms. Shape: (n_pairs, 1, 3).
                - "atomic_embedding" (torch.Tensor): Embeddings of atomic numbers. Shape: (n_atoms, embedding_dim).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the transformed input tensors.
                - "mu" (torch.Tensor): Zero-initialized tensor for atom features. Shape: (n_atoms, 3, nr_atom_basis).
                - "dir_ij" (torch.Tensor): Direction vectors between atoms. Shape: (n_pairs, 1, distance).
                - "q" (torch.Tensor): Reshaped atomic number embeddings. Shape: (n_atoms, 1, embedding_dim).
        """
        from modelforge.potential.utils import _distance_to_radial_basis

        # compute pairwise distances
        d_ij = inputs["d_ij"]
        r_ij = inputs["r_ij"]
        dir_ij = r_ij / d_ij
        f_ij, _ = _distance_to_radial_basis(d_ij, self.radial_basis_module)

        fcut = self.cutoff_module(d_ij)

        filters = self.filter_net(f_ij) * fcut[..., None]
        self.filter_list = torch.split(filters, 3 * self.nr_atom_basis, dim=-1)

        # generate q and mu
        atomic_embedding = inputs["atomic_embedding"]

        q = atomic_embedding[:, None]
        qs = q.shape
        mu = torch.zeros(
            (qs[0], 3, qs[2]), device=q.device
        )  # total_number_of_atoms_in_the_batch, 3, nr_atom_basis
        return {"mu": mu, "dir_ij": dir_ij, "q": q}

    def _forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        inputs: Dict[str, torch.Tensor]
            Dictionary containing pairlist information.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector representations.
        """

        # extract properties from pairlist
        transformed_input = self._generate_representation(inputs)
        q = transformed_input["q"]
        mu = transformed_input["mu"]
        for i, interaction_mod in enumerate(self.interaction_modules):
            q, mu = interaction_mod(
                q,
                mu,
                self.filter_list[i],
                transformed_input["dir_ij"],
                inputs["pair_indices"],
            )

        # Use squeeze to remove dimensions of size 1
        q = q.squeeze(dim=1)

        return {
            "scalar_representation": q,
            "vector_representation": mu,
            "atomic_subsystem_indices": inputs["atomic_subsystem_indices"],
        }


class SAKEInteraction(nn.Module):
    """
    SAKE Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

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
        intra_atomic_net : nn.Sequential
            Neural network for intra-atomic interactions.
        """
        super().__init__()
        self.nr_atom_basis = nr_atom_basis

        # Initialize the intra-atomic neural network
        self.intra_atomic_net = nn.Sequential(
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

        x = self.intra_atomic_net(q)
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


class LightningSAKE(SAKE, LightningModuleMixin):
    def __init__(
        self,
        embedding: nn.Module,
        nr_interaction_blocks: int,
        radial_basis: nn.Module,
        cutoff: nn.Module,
        activation: Optional[Callable] = F.silu,
        loss: Type[nn.Module] = nn.MSELoss(),
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
    ) -> None:
        """PyTorch Lightning version of the SAKE model."""

        super().__init__(
            embedding_module=embedding,
            nr_interaction_blocks=nr_interaction_blocks,
            radial_basis_module=radial_basis,
            cutoff_module=cutoff,
            activation=activation,
        )
        self.loss_function = loss
        self.optimizer = optimizer
        self.learning_rate = lr

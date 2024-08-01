from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from loguru import logger as log
from openff.units import unit
from .models import NNPInput, BaseNetwork, CoreNetwork, PairListOutputs

from .utils import Dense
from dataclasses import dataclass

from modelforge.potential.utils import NeuralNetworkData


@dataclass
class PaiNNNeuralNetworkData(NeuralNetworkData):
    """
    A dataclass designed to structure the inputs for PaiNN neural network potentials, ensuring
    an efficient and structured representation of atomic systems for energy computation and
    property prediction within the PaiNN framework.
    """

    atomic_embedding: torch.Tensor


class PaiNNCore(CoreNetwork):
    """PaiNN - polarizable interaction neural network

    References:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        featurization_config: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: unit.Quantity,
        number_of_interaction_modules: int,
        shared_interactions: bool,
        shared_filters: bool,
        activation_name: str,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the PaiNNCore network.

        Parameters
        ----------
        featurization_config : Dict[str, Union[List[str], int]]
            Configuration for atomic featurization.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        maximum_interaction_radius : unit.Quantity
            Maximum interaction radius.
        number_of_interaction_modules : int
            Number of interaction modules.
        shared_interactions : bool
            Whether to share interactions across modules.
        shared_filters : bool
            Whether to share filters across modules.
        activation_name : str
            Activation function to use.
        epsilon : float, optional
            Stability constant added in norm to prevent numerical instabilities. Default is 1e-8.
        """
        from modelforge.potential.utils import FeaturizeInput

        log.debug("Initializing the PaiNN architecture.")
        super().__init__(activation_name)

        self.number_of_interaction_modules = number_of_interaction_modules

        # featurize the atomic input

        self.featurize_input = FeaturizeInput(featurization_config)
        number_of_per_atom_features = int(
            featurization_config["number_of_per_atom_features"]
        )
        # initialize representation block
        self.representation_module = PaiNNRepresentation(
            maximum_interaction_radius,
            number_of_radial_basis_functions,
            number_of_interaction_modules,
            number_of_per_atom_features,
            shared_filters,
        )

        # initialize the interaction and mixing networks
        if shared_interactions:
            self.interaction_modules = nn.ModuleList(
                [
                    PaiNNInteraction(
                        number_of_per_atom_features,
                        activation_function=self.activation_function_class(),
                    )
                ]
                * number_of_interaction_modules
            )
        else:
            self.interaction_modules = nn.ModuleList(
                [
                    PaiNNInteraction(
                        number_of_per_atom_features,
                        activation_function=self.activation_function_class(),
                    )
                    for _ in range(number_of_interaction_modules)
                ]
            )

        self.mixing_modules = nn.ModuleList(
            [
                PaiNNMixing(
                    number_of_per_atom_features,
                    activation_function=self.activation_function_class(),
                    epsilon=epsilon,
                )
                for _ in range(number_of_interaction_modules)
            ]
        )

        # reduce per-atom features to per atom scalar
        self.energy_layer = nn.Sequential(
            Dense(
                number_of_per_atom_features,
                number_of_per_atom_features,
                activation_function=self.activation_function_class(),
            ),
            Dense(
                number_of_per_atom_features,
                1,
            ),
        )

    def _model_specific_input_preparation(
        self, data: NNPInput, pairlist_output: PairListOutputs
    ) -> PaiNNNeuralNetworkData:
        """
        Prepare the model-specific input for the PaiNN network.

        Parameters
        ----------
        data : NNPInput
            The input data.
        pairlist_output : PairListOutputs
            The pairlist output.

        Returns
        -------
        PaiNNNeuralNetworkData
            The prepared model-specific input.
        """
        # Perform atomic embedding

        number_of_atoms = data.atomic_numbers.shape[0]

        nnp_input = PaiNNNeuralNetworkData(
            pair_indices=pairlist_output.pair_indices,
            d_ij=pairlist_output.d_ij,
            r_ij=pairlist_output.r_ij,
            number_of_atoms=number_of_atoms,
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            atomic_subsystem_indices=data.atomic_subsystem_indices,
            total_charge=data.total_charge,
            atomic_embedding=self.featurize_input(data),
        )  # add per-atom properties and embedding

        return nnp_input

    def compute_properties(
        self,
        data: PaiNNNeuralNetworkData,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        data : PaiNNNeuralNetworkData
            The input data.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing scalar and vector representations.
        """
        # initialize filters, q and mu
        transformed_input = self.representation_module(data)

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
                data.pair_indices,
            )
            q, mu = mixing_mod(q, mu)

        # Use squeeze to remove dimensions of size 1
        q = q.squeeze(dim=1)
        E_i = self.energy_layer(q).squeeze(1)

        return {
            "per_atom_energy": E_i,
            "mu": mu,
            "q": q,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
        }


class PaiNNRepresentation(nn.Module):
    """PaiNN representation module"""

    def __init__(
        self,
        maximum_interaction_radius: unit.Quantity,
        number_of_radial_basis_functions: int,
        nr_interaction_blocks: int,
        nr_atom_basis: int,
        shared_filters: bool,
    ):
        """
        Initialize the PaiNNRepresentation module.

        Parameters
        ----------
        maximum_interaction_radius : unit.Quantity
            Maximum interaction radius.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        nr_interaction_blocks : int
            Number of interaction blocks.
        nr_atom_basis : int
            Number of features to describe atomic environments.
        shared_filters : bool
            Whether to share filters across blocks.
        """
        from .utils import SchnetRadialBasisFunction

        super().__init__()

        # cutoff
        from modelforge.potential import CosineCutoff

        self.cutoff_module = CosineCutoff(maximum_interaction_radius)

        # radial symmetry function

        self.radial_symmetry_function_module = SchnetRadialBasisFunction(
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            max_distance=maximum_interaction_radius,
            dtype=torch.float32,
        )

        # initialize the filter network
        if shared_filters:
            filter_net = Dense(
                in_features=number_of_radial_basis_functions,
                out_features=3 * nr_atom_basis,
            )

        else:
            filter_net = Dense(
                in_features=number_of_radial_basis_functions,
                out_features=nr_interaction_blocks * nr_atom_basis * 3,
            )

        self.filter_net = filter_net

        self.shared_filters = shared_filters
        self.nr_interaction_blocks = nr_interaction_blocks
        self.nr_atom_basis = nr_atom_basis

    def forward(self, data: PaiNNNeuralNetworkData) -> Dict[str, torch.Tensor]:
        """
        Transforms the input data for the PaiNN potential model.

        Parameters
        ----------
        data : PaiNNNeuralNetworkData
            The input data.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the transformed input tensors.
        """
        # compute normalized pairwise distances
        d_ij = data.d_ij
        dir_ij = data.r_ij / d_ij

        # featurize pairwise distances using RBF
        f_ij = self.radial_symmetry_function_module(d_ij)
        # calculate the smoothing values
        fcut = self.cutoff_module(d_ij)
        # pass the featurized distances through the filter network and apply cutoff based on distances
        filters = self.filter_net(f_ij) * fcut

        # depending on whether we share filters or not
        # filters have different shape at dim=1 (dim=0 is always the number of atom pairs)
        # if we share filters, we copy the filters and use the same filters for all blocks
        if self.shared_filters:
            filter_list = [filters] * self.nr_interaction_blocks
        # otherwise we index into subset of the calculated filters and provide each block with its own set of filters
        else:
            filter_list = torch.split(filters, 3 * self.nr_atom_basis, dim=-1)

        # generate q and mu
        q = data.atomic_embedding.unsqueeze(1)  # nr_of_atoms, 1, nr_atom_basis
        q_shape = q.shape
        mu = torch.zeros(
            (q_shape[0], 3, q_shape[2]), device=q.device, dtype=q.dtype
        )  # nr_of_atoms, 3, nr_atom_basis

        return {"filters": filter_list, "dir_ij": dir_ij, "q": q, "mu": mu}


class PaiNNInteraction(nn.Module):
    """
    PaiNN Interaction Block for Modeling Equivariant Interactions of Atomistic Systems.

    """

    def __init__(self, nr_atom_basis: int, activation_function: torch.nn.Module):
        """
        Initialize the PaiNNInteraction module.

        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation_function : torch.nn.Module
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
            Dense(
                nr_atom_basis, nr_atom_basis, activation_function=activation_function
            ),
            Dense(nr_atom_basis, 3 * nr_atom_basis),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        W_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        pairlist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interaction output.

        Parameters
        ----------
        q : torch.Tensor
            Scalar input values of shape [nr_of_atoms, 1, nr_atom_basis].
        mu : torch.Tensor
            Vector input values of shape [nr_of_atoms, 3, nr_atom_basis].
        W_ij : torch.Tensor
            Filter of shape [nr_of_pairs, 1, n_interactions].
        dir_ij : torch.Tensor
            Directional vector between atoms i and j.
        pairlist : torch.Tensor, shape (2, n_pairs)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated scalar and vector representations (q, mu).
        """
        # perform the scalar operations (same as in SchNet)
        idx_i, idx_j = pairlist[0], pairlist[1]

        x_per_atom = self.interatomic_net(q)  # per atom

        x_j = x_per_atom[idx_j]  # per pair
        x_per_pair = W_ij.unsqueeze(1) * x_j  # per_pair

        # split the output into dq, dmuR, dmumu to exchange information between the scalar and vector outputs
        dq_per_pair, dmuR, dmumu = torch.split(x_per_pair, self.nr_atom_basis, dim=-1)

        # for scalar output only dq is used
        # scatter the dq to the atoms (reducton from pairs to atoms)
        dq_per_atom = torch.zeros_like(q)  # Shape: (nr_of_pairs, 1, nr_atom_basis)
        # Expand idx_i to match the shape of dq for scatter_add operation
        expanded_idx_i = idx_i.unsqueeze(-1).expand(-1, dq_per_pair.size(2))

        dq_per_atom.scatter_add_(0, expanded_idx_i.unsqueeze(1), dq_per_pair)

        q = q + dq_per_atom

        # ----------------- vector output -----------------
        # for vector output dmuR and dmumu are used
        # dmuR: (nr_of_pairs, 1, nr_atom_basis)
        # dir_ij: (nr_of_pairs, 3)
        # dmumu: (nr_of_pairs, 1, nr_atom_basis)
        # muj: (nr_of_pairs, 1, nr_atom_basis)
        # idx_i: (nr_of_pairs)
        # mu: (nr_of_atoms, 3, nr_atom_basis)

        muj = mu[idx_j]  # shape (nr_of_pairs, 1, nr_atom_basis)

        dmu_per_pair = (
            dmuR * dir_ij.unsqueeze(-1) + dmumu * muj
        )  # shape (nr_of_pairs, 3, nr_atom_basis)

        # Create a tensor to store the result, matching the size of `mu`
        dmu_per_atom = torch.zeros_like(mu)  # Shape: (nr_of_atoms, 3, nr_atom_basis)

        # Expand idx_i to match the shape of dmu for scatter_add operation
        expanded_idx_i = (
            idx_i.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, dmu_per_atom.size(1), dmu_per_atom.size(2))
        )

        # Perform scatter_add_ operation
        dmu_per_atom.scatter_add_(0, expanded_idx_i, dmu_per_pair)

        mu = mu + dmu_per_atom

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(
        self,
        nr_atom_basis: int,
        activation_function: torch.nn.Module,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the PaiNNMixing module.

        Parameters
        ----------
        nr_atom_basis : int
            Number of features to describe atomic environments.
        activation_function : torch.nn.Module
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
            Dense(
                2 * nr_atom_basis,
                nr_atom_basis,
                activation_function=activation_function,
            ),
            Dense(nr_atom_basis, 3 * nr_atom_basis, activation_function=None),
        )
        # initialize the mu channel mixing network
        self.mu_channel_mix = Dense(nr_atom_basis, 2 * nr_atom_basis, bias=False)
        self.epsilon = epsilon

    def forward(
        self, q: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intratomic mixing.

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


from .models import ComputeInteractingAtomPairs, NNPInput, BaseNetwork


class PaiNN(BaseNetwork):
    def __init__(
        self,
        featurization: Dict[str, Union[List[str], int]],
        number_of_radial_basis_functions: int,
        maximum_interaction_radius: Union[unit.Quantity, str],
        number_of_interaction_modules: int,
        activation_function: str,
        shared_interactions: bool,
        shared_filters: bool,
        postprocessing_parameter: Dict[str, Dict[str, bool]],
        dataset_statistic: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-8,
    ) -> None:
        """
        Initialize the PaiNN network.

        Parameters
        ----------
        featurization : Dict[str, Union[List[str], int]]
            Configuration for atomic featurization.
        number_of_radial_basis_functions : int
            Number of radial basis functions.
        maximum_interaction_radius : Union[unit.Quantity, str]
            Maximum interaction radius.
        number_of_interaction_modules : int
            Number of interaction modules.
        activation_function : str
            Activation function to use.
        shared_interactions : bool
            Whether to share interactions across modules.
        shared_filters : bool
            Whether to share filters across modules.
            epsilon=epsilon,
        )
        """

        from modelforge.utils.units import _convert_str_to_unit

        self.only_unique_pairs = False  # NOTE: for pairlist

        super().__init__(
            dataset_statistic=dataset_statistic,
            postprocessing_parameter=postprocessing_parameter,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
        )

        self.core_module = PaiNNCore(
            featurization_config=featurization,
            number_of_radial_basis_functions=number_of_radial_basis_functions,
            maximum_interaction_radius=_convert_str_to_unit(maximum_interaction_radius),
            number_of_interaction_modules=number_of_interaction_modules,
            shared_interactions=shared_interactions,
            shared_filters=shared_filters,
            activation_name=activation_function,
            epsilon=epsilon,
        )

    def _config_prior(self):
        log.info("Configuring PaiNN model hyperparameter prior distribution")
        from modelforge.utils.io import import_

        tune = import_("ray").tune
        # from ray import tune

        from modelforge.potential.utils import shared_config_prior

        prior = {
            "number_of_per_atom_features": tune.randint(2, 256),
            "number_of_interaction_modules": tune.randint(1, 5),
            "maximum_interaction_radius": tune.uniform(5, 10),
            "number_of_radial_basis_functions": tune.randint(8, 32),
            "shared_filters": tune.choice([True, False]),
            "shared_interactions": tune.choice([True, False]),
        }
        prior.update(shared_config_prior())
        return prior

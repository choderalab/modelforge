"""
MultipoleNet: an equivariant neural network potential that predicts per-atom
charge and spin multipoles (monopole, dipole, quadrupole) as a latent
representation, then reads out energy from the multipole latent representation.
"""

from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
from loguru import logger as log

from modelforge.utils.prop import NNPInput
from modelforge.potential.neighbors import PairlistData

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.math import soft_one_hot_linspace


class MultipoleNetCore(nn.Module):
    def __init__(
        self,
    ) -> None:
        """
        Core MultipoleNet architecture for predicting equivariant per-atom
        multipoles and reading out energy and partial charges.
        Multipole properties (q, mu, 2S) can be readout directly from the
        latent multipole space.
        """
        super().__init__()

        log.debug("Initializing the MultipoleNet architecture.")

    def compute_properties(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:

        return {
            "per_atom_charge_multipole": None,
            "per_atom_spin_multipole": None,
            "per_atom_scalar_representation": None,
            "atomic_subsystem_indices": data.atomic_subsystem_indices,
            "atomic_numbers": data.atomic_numbers,
            "per_atom_charge": None,
        }

    @staticmethod
    def calculate_per_system_dipole_moment(
        charge_multipole: torch.Tensor,
        positions: torch.Tensor,
        atomic_subsystem_indices: torch.Tensor,
        per_system_dipole_origin: torch.Tensor = None,
    ):
        partial_charge = charge_multipole[:, 0]
        dipole = charge_multipole[:, 1:4]
        number_of_systems = positions.shape[0]

        # default origin is (0, 0, 0)
        if per_system_dipole_origin is not None:
            positions = positions - per_system_dipole_origin

        per_atom_dipole_moment = partial_charge.unsqueeze(-1) * positions + dipole
        per_system_dipole_moment = torch.zeros(
            number_of_systems,
            3,
            dtype=per_atom_dipole_moment.dtype,
            device=per_atom_dipole_moment.device
        )
        per_system_dipole_moment = per_system_dipole_moment.index_add_(
            0,
            atomic_subsystem_indices,
            per_atom_dipole_moment,
        )

        return per_system_dipole_moment  # Shape: (number_of_systems, 3)


    @staticmethod
    def _multipole_invariants(multipole: torch.Tensor) -> torch.Tensor:
        """
        Reduce an (A, 9) multipole to (A, 3) rotation invariants:
        [monopole, |dipole|, ||quadrupole||].

        Parameters
        ----------
        multipole : torch.Tensor, shape [n_atoms, 9]
            Per-atom multipoles [monopole | dipole(3) | quadrupole(5)].

        Returns
        -------
        torch.Tensor, shape [n_atoms, 3]
            Per-atom rotation invariants.
        """
        monopole = multipole[:, 0:1]
        dipole_norm = multipole[:, 1:4].norm(dim=1, keepdim=True)
        quadrupole_norm = multipole[:, 4:9].norm(dim=1, keepdim=True)
        return torch.cat([monopole, dipole_norm, quadrupole_norm], dim=1)

    def forward(
        self, data: NNPInput, pairlist_output: PairlistData
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MultipoleNet model.

        Parameters
        ----------
        data : NNPInput
            Input data including atomic numbers, positions, and relevant fields.
        pairlist_output : PairlistData
            Pair indices, distances, and displacement vectors.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of calculated properties. Contains the equivariant
            latent (per_atom_charge_multipole, per_atom_spin_multipole),
            the readout outputs (e.g. per_atom_energy, per_atom_charge).
        """
        # Compute the equivariant latent and its invariants.
        results = self.compute_properties(data, pairlist_output)
        atom_invariants = results["per_atom_scalar_representation"]

        # Scalar readout: invariants -> per-atom energy / charge.
        readout = self.readout_module(atom_invariants)
        results.update(readout)

        # Calculate dipole moments from the latent charge multipole
        results["per_system_dipole_moment"] = self.calculate_per_system_dipole_moment(
            results["per_atom_charge_multipole"],
            data.positions,
            results["atomic_subsystem_indices"],
        )

        return results


class MultipoleInteractionModule(nn.Module):
    def __init__(
        self,
        irreps_in: Optional[o3._irreps.Irreps, str],
        irreps_out: Optional[o3._irreps.Irreps, str],
        irreps_spherical_harmonics: Optional[o3._irreps.Irreps, str],
        number_of_radial_basis_functions: int,
        number_of_radial_basis_module_dimensions: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        maximum_interaction_radius: float,
    ):
        """
        Equivariant message passing with a gated nonlinearity on the invariant features.

        Parameters
        ----------

        """
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_spherical_harmonics)
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        self.number_of_radial_basis_module_dimensions = number_of_radial_basis_module_dimensions
        self.activation_function = activation_function
        self.maximum_interaction_radius = maximum_interaction_radius

        # split irreps_out into scalar vs higher-l for the Gate
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])

        # one scalar gate per gated channel
        n_gated = irreps_gated.num_irreps
        irreps_gates = o3.Irreps(f"{n_gated}x0e") if n_gated > 0 else o3.Irreps("")

        self.gate = Gate(
            irreps_scalars, [self.activation_function] if len(irreps_scalars) > 0 else [],
            irreps_gates, [torch.sigmoid] if len(irreps_gates) > 0 else [],  # bounded activation, fixed
            irreps_gated,
        )

        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_spherical_harmonics,
            self.gate.irreps_in,
            shared_weights=False,
        )

        self.radial_basis_module = FullyConnectedNet(
            [
                number_of_radial_basis_functions,
                self.number_of_radial_basis_module_dimensions,
                self.tensor_product.weight_numel,
            ],
            self.activation_function,
        )

        self.self_interaction = o3.Linear(self.irreps_in, self.gate.irreps_in)  # match tensor product for aggregation

    def forward(
        self,
        x,
        pair_indices,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        atomic_numbers,
    ):
        # angular
        spherical_harmonics_vector = o3.spherical_harmonics(
            self.irreps_sh,
            d_ij,
            normalize=True,
            normalization='component',
        )

        # radial
        atomic_number_scale = (
                atomic_numbers[pair_indices[0]].float() * atomic_numbers[pair_indices[1]].float()
        ).sqrt()
        r_ij_scaled = r_ij / atomic_number_scale
        radial_basis_function_vector = soft_one_hot_linspace(
            r_ij_scaled,
            0.0,
            self.maximum_interaction_radius / 6,  # the atomic number of C
            self.number_of_radial_basis_functions,
            basis='gaussian',
            cutoff=True,
        )

        # message-passing
        w = self.radial_basis_module(radial_basis_function_vector)
        message = self.tensor_product(
            x[pair_indices[0]],
            spherical_harmonics_vector[pair_indices[0]],
            w,
        )
        agg = torch.zeros(
            d_ij.shape[0],
            message.shape[1],
            dtype=message.dtype,
            device=message.device,
        ).index_add_(
            0,
            pair_indices[1],
            message,
        )

        x_pre_gate = self.self_interaction(x) + agg
        return self.gate(x_pre_gate)

class MultipoleRepresentation(nn.Module):
    def __init__(
        self,
        number_of_hidden_layers: int,
        number_of_monopole_dimensions: int,
        number_of_dipole_dimensions: int,
        number_of_quadrupole_dimensions: int,
        maximum_l_of_spherical_harmonics: int,
        maximum_interaction_radius: float,
        number_of_radial_basis_functions: int,
        number_of_radial_basis_module_dimensions: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        """

        Parameters
        ----------
        number_of_hidden_layers
        number_of_monopole_dimensions
        number_of_dipole_dimensions
        number_of_quadrupole_dimensions
        maximum_l_of_spherical_harmonics
        maximum_interaction_radius
        activation_function
        """
        super().__init__()
        self.number_of_interaction_layers = number_of_hidden_layers
        self.number_of_monopole_dimensions = number_of_monopole_dimensions
        self.number_of_dipole_dimensions = number_of_dipole_dimensions
        self.number_of_quadrupole_dimensions = number_of_quadrupole_dimensions
        self.maximum_interaction_radius = maximum_interaction_radius
        self.number_of_radial_basis_functions = number_of_radial_basis_functions
        self.number_of_radial_basis_module_dimensions = number_of_radial_basis_module_dimensions
        self.activation_function = activation_function

        self.irreps_spherical_harmonics = o3.Irreps.spherical_harmonics(maximum_l_of_spherical_harmonics)
        self.irreps_hidden = o3.Irreps(
            f"{self.number_of_monopole_dimensions}x0e"
            f"+{self.number_of_dipole_dimensions}x1o"
            f"+{self.number_of_quadrupole_dimensions}x2e",
        )

        # (q, 2S) conditioning for monopole (0e) properties
        # 2S is the number of unpaired electrons
        self.conditioning = nn.Linear(2, self.number_of_monopole_dimensions)

        irreps_in = o3.Irreps(f"{self.number_of_monopole_dimensions}x0e")
        self.interaction_layers = nn.ModuleList([
            MultipoleInteractionModule(
                irreps_in if i == 0 else self.irreps_hidden,
                self.irreps_hidden,
                self.irreps_spherical_harmonics,
                self.number_of_radial_basis_functions,
                self.number_of_radial_basis_module_dimensions,
                self.activation_function,
                self.maximum_interaction_radius,
            )
            for i in range(self.number_of_interaction_layers)])

        irreps_out = o3.Irreps("1x0e + 1x1o + 1x2e")  # monopole, dipole, quadrupole
        self.head_charge = o3.Linear(self.irreps_hidden, irreps_out)
        self.head_spin = o3.Linear(self.irreps_hidden, irreps_out)

    def forward(
        self,
        data: NNPInput,
        pairlist_output: PairlistData,
    ):
        x = self.conditioning(
            torch.stack([
                (data.per_system_spin_state - 1).to(data.positions.dtype),
                data.per_system_total_charge.to(data.positions.dtype),
            ], dim=-1)[data.atomic_subsystem_indices]
        )

        for layer in self.interaction_layers:
            x = layer(
                x,
                pairlist_output.pair_indices,
                pairlist_output.d_ij,
                pairlist_output.r_ij,
                data.atomic_subsystem_indices,
            )

        charge_multipole = self.head_charge(x)
        spin_multipole = self.head_spin(x)

        # force the sum of partial charges and spin on the monopole dimensions
        charge_mult = self._project_monopole(
            charge_multipole,
            data.per_system_total_charge,
        )
        spin_mult = self._project_monopole(
            spin_multipole,
            data.per_system_spin_state - 1,
        )

        return charge_mult, spin_mult

    @staticmethod
    def _project_monopole(
        multipole: torch.Tensor,
        monopole_target: torch.Tensor,
    ):
        """"""
        number_of_atoms = multipole.shape[0]
        out = multipole.clone()
        out[:, 0] = multipole[:, 0] - multipole[:, 0].mean() + monopole_target / number_of_atoms
        return out


class MultipoleReadout(nn.Module):
    def __init__(
        self,
        number_of_hidden_features: int,
        activation_function: nn.Module,
    ):
        """
        Scalar readout module. Consumes the per-atom rotation invariants of the
        charge and spin multipoles and predicts the requested per-atom energy.
        Only energy gets its output neural network, while charges will be
        read out from the charge monopole directly.

        Parameters
        ----------
        number_of_hidden_features : int
            Dimension of the readout network.
        activation_function : nn.Module
            Activation function for the readout network.
        """
        super().__init__()

        # 3 invariants from the charge latent + 3 from the spin latent = 6.
        number_of_invariants_per_atom = 6

        self.energy_readout = nn.Sequential(
            nn.Linear(number_of_invariants_per_atom, number_of_hidden_features),
            activation_function,
            nn.Linear(number_of_hidden_features, number_of_hidden_features),
            activation_function,
            nn.Linear(number_of_hidden_features, 1),  # per-atom energy
        )

    def forward(self, atom_invariants: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the readout.

        Parameters
        ----------
        atom_invariants : torch.Tensor, shape [n_atoms, 6]
            Per-atom rotation invariants of the charge and spin multipoles.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary mapping each predicted property name to its per-atom
            output tensor.
        """
        results = {
            "per_atom_energy": self.energy_readout(atom_invariants),
            "per_atom_charge": atom_invariants[:, 0]
        }

        return results
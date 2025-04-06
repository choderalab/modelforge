"""
This module contains utility functions and classes for processing the output of the potential model.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, Union

import torch
from openff.units import unit

from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT


def load_atomic_self_energies(path: str) -> Dict[str, unit.Quantity]:
    """
    Load the atomic self energies from the dataset statistics toml.

    Parameters
    ----------
    path : str
        path to the dataset statistics toml file.

    Returns
    -------
    Dict[str, unit.Quantity]
        returns the atomic self energies from the dataset statistics toml with units attached.
    """

    import toml

    energy_statistic = toml.load(open(path, "r"))

    # attach units
    atomic_self_energies = {
        key: unit.Quantity(value)
        for key, value in energy_statistic["atomic_self_energies"].items()
    }

    return atomic_self_energies


def load_dataset_energy_statistics(path: str) -> Dict[str, unit.Quantity]:
    """
    Load the per-atom energy distribution statistics (mean and stddev of atomic energies) from the dataset statistics toml.

    Parameters
    ----------
    path : str
        path to the dataset statistics toml file.

    Returns
    -------
    Dict[str, unit.Quantity]
        returns the per-atom energy distribution statistics from the dataset statistics toml with units attached.
    """
    import toml

    energy_statistic = toml.load(open(path, "r"))
    # attach units
    training_dataset_statistics = {
        key: unit.Quantity(value)
        for key, value in energy_statistic["training_dataset_statistics"].items()
    }

    return training_dataset_statistics


class FromAtomToMoleculeReduction(torch.nn.Module):
    """
    Reducing per-atom property to per-molecule property.
    """

    def __init__(
        self,
        reduction_mode: str = "sum",
    ):
        """
        Initializes the per-atom property readout_operation module.

        Parameters
        ----------
        per_atom_property_name : str
            The name of the per-atom property that will be reduced.
        index_name : str
            The name of the index used to identify the molecules.
        output_name : str
            The name of the output property.
        reduction_mode : str, optional
            The reduction mode. Default is "sum".
        keep_per_atom_property : bool, optional
            Whether to keep the per-atom property. Default is False.
        """
        super().__init__()
        self.reduction_mode = reduction_mode

    def forward(
        self, indices: torch.Tensor, per_atom_property: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data dictionary containing the per-atom property and
            index.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output data dictionary containing the per-molecule property.
        """

        # Perform scatter add operation for atoms belonging to the same molecule
        nr_of_molecules = torch.unique(indices).unsqueeze(1)
        per_system_property = torch.zeros_like(
            nr_of_molecules,
            dtype=per_atom_property.dtype,
            device=per_atom_property.device,
        )

        return per_system_property.scatter_reduce(
            0,
            indices.long().unsqueeze(1),
            per_atom_property,
            reduce=self.reduction_mode,
        )


@dataclass
class AtomicSelfEnergies:
    """
    AtomicSelfEnergies stores a mapping of atomic elements to their self energies.

    Provides lookup by atomic number or symbol, iteration over the mapping,
    and utilities to convert between atomic number and symbol.

    Intended as a base class to be extended with specific element-energy values.
    """

    # We provide a dictionary with {str:float} of element name to atomic self-energy,
    # which can then be accessed by atomic index or element name
    energies: Dict[str, unit.Quantity] = field(default_factory=dict)
    # Example mapping, replace or extend as necessary
    atomic_number_to_element: Dict[int, str] = field(
        default_factory=lambda: _ATOMIC_NUMBER_TO_ELEMENT
    )
    _ase_tensor_for_indexing = None

    def __getitem__(self, key):
        from modelforge.utils.units import chem_context

        if isinstance(key, int):
            # Convert atomic number to element symbol
            element = self.atomic_number_to_element.get(key)
            if element is None:
                raise KeyError(f"Atomic number {key} not found.")
            if self.energies.get(element) is None:
                return None
            return self.energies.get(element).to(unit.kilojoule_per_mole, "chem").m
        elif isinstance(key, str):
            # Directly access by element symbol
            if key not in self.energies:
                raise KeyError(f"Element {key} not found.")
            if self.energies[key] is None:
                return None

            return self.energies[key].to(unit.kilojoule_per_mole, "chem").m
        else:
            raise TypeError(
                "Key must be an integer (atomic number) or string (element name)."
            )

    def __iter__(self) -> Iterator[Dict[str, float]]:
        """Iterate over the energies dictionary."""
        from modelforge.utils.units import chem_context

        for element, energy in self.energies.items():
            atomic_number = self.element_to_atomic_number(element)
            yield (atomic_number, energy.to(unit.kilojoule_per_mole, "chem").m)

    def __len__(self) -> int:
        """Return the number of element-energy pairs."""
        return len(self.energies)

    def element_to_atomic_number(self, element: str) -> int:
        """Return the atomic number for a given element symbol."""
        for atomic_number, elem_symbol in self.atomic_number_to_element.items():
            if elem_symbol == element:
                return atomic_number
        raise ValueError(f"Element symbol '{element}' not found in the mapping.")

    @property
    def atomic_number_to_energy(self) -> Dict[int, float]:
        """Return a dictionary mapping atomic numbers to their energies."""
        return {
            atomic_number: self[atomic_number]
            for atomic_number in self.atomic_number_to_element.keys()
            if self[atomic_number] is not None
        }

    @property
    def ase_tensor_for_indexing(self) -> torch.Tensor:
        if self._ase_tensor_for_indexing is None:
            max_z = max(self.atomic_number_to_element.keys()) + 1
            ase_tensor_for_indexing = torch.zeros(max_z)
            for idx in self.atomic_number_to_element:
                if self[idx]:
                    ase_tensor_for_indexing[idx] = self[idx]
                else:
                    ase_tensor_for_indexing[idx] = 0.0
            self._ase_tensor_for_indexing = ase_tensor_for_indexing

        return self._ase_tensor_for_indexing


class ScaleValues(torch.nn.Module):
    """
    Rescales values using the provided mean and standard deviation.
    """

    def __init__(
        self,
        mean: float,
        stddev: float,
    ) -> None:
        """
        Rescales values using the provided mean and standard deviation.
        Parameters
        ----------
        mean : float
            The mean value used for rescaling.
        stddev : float
            The standard deviation value used for rescaling.
        property : str
            The name of the property to be rescaled.
        output_name : str
            The name of the output property.

        """

        super().__init__()
        self.register_buffer("mean", torch.tensor([mean]))
        self.register_buffer("stddev", torch.tensor([stddev]))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Rescales values using the provided mean and standard deviation.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data dictionary containing the property to be rescaled.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output data dictionary containing the rescaled values.
        """
        return data * self.stddev + self.mean


def default_charge_conservation(
    per_atom_charge: torch.Tensor,
    per_system_total_charge: torch.Tensor,
    mol_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Adjusts partial atomic charges so that the sum of charges in each molecule
    matches the desired total charge.

    This method is based on equation 14 from the PhysNet paper.

    Parameters
    ----------
    partial_charges : torch.Tensor
        Tensor of partial charges for all atoms in all molecules.
    per_system_total_charge : torch.Tensor
        Tensor of desired total charges for each molecule.
    mol_indices : torch.Tensor
        Tensor of integers indicating which molecule each atom belongs to.

    Returns
    -------
    torch.Tensor
        Tensor of corrected partial charges.
    """
    # Calculate the sum of partial charges for each molecule
    predicted_per_system_total_charge = torch.zeros_like(
        per_system_total_charge, dtype=per_atom_charge.dtype
    ).scatter_add_(
        0,
        mol_indices.long().unsqueeze(1),
        per_atom_charge,
    )

    # Calculate the number of atoms in each molecule
    num_atoms_per_system = mol_indices.bincount(
        minlength=per_system_total_charge.size(0)
    )

    # Calculate the correction factor for each molecule
    correction_factors = (
        per_system_total_charge - predicted_per_system_total_charge
    ) / num_atoms_per_system.unsqueeze(1)

    # Apply the correction to each atom's charge
    per_atom_charge_corrected = per_atom_charge + correction_factors[mol_indices]

    return per_atom_charge_corrected


class ChargeConservation(torch.nn.Module):
    def __init__(self, method="default"):
        """
        Module to enforce charge conservation on partial atomic charges.

        Parameters
        ----------
        method : str, optional, default='default'
            The method to use for charge conservation. Currently, only 'default'
            is supported.

        Methods
        -------
        forward(data)
            Applies charge conservation to the partial charges in the provided
            data dictionary.
        """

        super().__init__()
        self.method = method
        if self.method == "default":
            self.correct_partial_charges = default_charge_conservation
        else:
            raise ValueError(f"Unknown charge conservation method: {self.method}")

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ):
        """
        Apply charge conservation to partial charges in the data dictionary.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing the following keys:
            - "per_atom_charge":
                Tensor of partial charges for all atoms in the batch.
            -  "per_system_total_charge":
                Tensor of desired total charges for each
            molecule.
            - "atomic_subsystem_indices":
                Tensor indicating which molecule each atom belongs to.

        Returns
        -------
        Dict[str, torch.Tensor]
            Updated data dictionary with the key "per_atom_charge_corrected"
            added, containing the corrected per-atom charges.
        """
        data["per_atom_charge_uncorrected"] = data["per_atom_charge"]
        data["per_atom_charge"] = self.correct_partial_charges(
            data["per_atom_charge"],
            data["per_system_total_charge"],
            data["atomic_subsystem_indices"],
        )
        return data


class PerAtomEnergy(torch.nn.Module):

    def __init__(
        self,
        per_atom_energy: Dict[str, bool],
        dataset_statistics: Dict[str, float],
    ):
        """
        Process per atom energies. Depending on what has been requested in the per_atom_energy dictionary, the per atom energies are normalized and/or reduced to per system energies.
        Parameters
        ----------
        per_atom_energy : Dict[str, bool]
            A dictionary containing the per atom energy processing options.
        dataset_statistics : Dict[str, float]

        """
        super().__init__()

        if per_atom_energy.get("normalize"):
            scale = ScaleValues(
                dataset_statistics["per_atom_energy_mean"],
                dataset_statistics["per_atom_energy_stddev"],
            )
        else:
            scale = ScaleValues(0.0, 1.0)

        self.scale = scale

        if per_atom_energy.get("from_atom_to_system_reduction"):
            reduction = FromAtomToMoleculeReduction()

        self.reduction = reduction

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        per_atom_property, indices = (
            data["per_atom_energy"],
            data["atomic_subsystem_indices"],
        )
        scaled_values = self.scale(per_atom_property)
        per_system_energy = self.reduction(indices, scaled_values)

        data["per_system_energy"] = per_system_energy
        data["per_atom_energy"] = data["per_atom_energy"].detach()

        return data


class PerAtomCharge(torch.nn.Module):

    def __init__(self, per_atom_charge: Dict[str, bool]):
        super().__init__()
        from torch import nn

        if per_atom_charge["conserve"] == True:
            self.conserve = ChargeConservation(per_atom_charge["conserve_strategy"])
        else:
            self.conserve = nn.Identity()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.conserve(data)


class CalculateAtomicSelfEnergy(torch.nn.Module):
    """
    Calculates the atomic self energy for each molecule.
    """

    def __init__(
        self, atomic_self_energies: Dict[str, Union[unit.Quantity, str]]
    ) -> None:
        """
        Calculates the atomic self energy for each molecule given the atomic self energies provided by the training dataset and the element information.


        Parameters
        ----------
        atomic_self_energies : Dict[str, Union[unit.Quantity, str]]
            A dictionary containing the atomic self energies for each atomic number.

        """

        super().__init__()

        # if values in atomic_self_energies are strings convert them to kJ/mol
        if isinstance(list(atomic_self_energies.values())[0], str):
            atomic_self_energies = {
                key: unit.Quantity(value) for key, value in atomic_self_energies.items()
            }
        self.atomic_self_energies = AtomicSelfEnergies(atomic_self_energies)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates the molecular self energy.

        Parameters
        ----------
        data : dict
            The input data for the model, including atomic numbers and subsystem indices.

        Returns
        -------
        torch.Tensor
            The tensor containing the molecular self energy for each molecule.
        """
        atomic_numbers = data["atomic_numbers"]
        atomic_subsystem_indices = data["atomic_subsystem_indices"]

        atomic_subsystem_indices = atomic_subsystem_indices.to(
            dtype=torch.long, device=atomic_numbers.device
        )

        # atomic_number_to_energy
        ase_tensor_for_indexing = self.atomic_self_energies.ase_tensor_for_indexing.to(
            device=atomic_numbers.device
        )

        # use the atomic numbers to generate a tensor that
        # contains the atomic self energy for each atomic number
        ase_tensor = ase_tensor_for_indexing[atomic_numbers]

        data["ase_tensor"] = ase_tensor
        return data


class CoulombPotential(torch.nn.Module):
    def __init__(self, cutoff: float):
        """
        Computes the long-range electrostatic energy for a molecular system
        based on predicted partial charges and pairwise distances between atoms.

        The implementation follows the methodology described in the PhysNet
        paper, using a cutoff function to handle long-range interactions.

        Parameters
        ----------
        cutoff : float
            The cutoff distance beyond which the interactions are not
            considered in nanometer.

        Attributes
        ----------
        strategy : str
            The strategy for computing long-range interactions.
        cutoff_function : nn.Module
            The cutoff function applied to the pairwise distances.
        """
        super().__init__()
        from .representation import PhysNetAttenuationFunction

        self.cutoff_function = PhysNetAttenuationFunction(cutoff)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the long-range electrostatic energy.

        This function calculates the long-range electrostatic energy by considering
        pairwise Coulomb interactions between atoms, applying a cutoff function to
        handle long-range interactions.

        this is based off the implementation in PhysNet:
        PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments, and Partial Charges
        Oliver T. Unke and Markus Meuwly
        Journal of Chemical Theory and Computation 2019 15 (6), 3678-3693
        DOI: 10.1021/acs.jctc.9b00181

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data containing the following keys:
            - 'per_atom_charge': Tensor of shape (N,) with partial charges for each atom.
            - 'atomic_subsystem_indices': Tensor indicating the molecule each atom belongs to.
            - 'pairwise_properties': Object containing pairwise distances and indices.

        Returns
        -------
        Dict[str, torch.Tensor]
            The input data dictionary with an additional key 'long_range_electrostatic_energy'
            containing the computed long-range electrostatic energy.
        """
        # todo:  need to add back in the multiple-cutoff neighborlist since we typically
        # want to do a longer range cutoff for electrostatics
        # this will ultimately just take an extra key for accessing the appropriate pair indices
        idx_i, idx_j = data["pair_indices"]

        # only unique paris
        unique_pairs_mask = idx_i < idx_j
        idx_i = idx_i[unique_pairs_mask]
        idx_j = idx_j[unique_pairs_mask]

        # since we will be working with pairs not individual atoms
        # we need to map the atomic_subsystem_indices onto the pairs
        # since pairs only exist between atoms on the same molecule
        # we can just use one of the pair indices tensors
        system_indices = data["atomic_subsystem_indices"]
        system_indices_of_pair = system_indices[idx_i]

        # mask pairwise properties
        pairwise_distances = data["d_ij"][unique_pairs_mask]
        # The per-atom charge tensor is expected to be of shape (N, 1) where N is the number of atoms
        # we will squeeze it to make it easier to work with
        per_atom_charge = data["per_atom_charge"].squeeze()

        # Initialize the long-range electrostatic energy
        # the shape should match the number of unique systems
        num_unique_systems = torch.unique(system_indices)

        electrostatic_energy = torch.zeros_like(
            num_unique_systems,
            dtype=per_atom_charge.dtype,
            device=per_atom_charge.device,
        )

        # Apply the cutoff function to pairwise distances
        phi_2r = self.cutoff_function(2 * pairwise_distances)

        chi_r = phi_2r * (1 / torch.sqrt(pairwise_distances**2 + 1)) + (
            1 - phi_2r
        ) * (1 / pairwise_distances)

        # Compute the Coulomb interaction term
        coulomb_interactions = (
            per_atom_charge[idx_i] * per_atom_charge[idx_j]
        ) * chi_r.squeeze()

        # sum up the energy for pairs in the same molecule
        data["electrostatic_energy"] = (
            electrostatic_energy.scatter_add_(
                0, system_indices_of_pair.long(), coulomb_interactions
            )
            * 138.96
        ).unsqueeze(
            1
        )  # in kj/mol nm

        return data


class ZBLPotential(torch.nn.Module):
    """
    Computes the Ziegler-Biersack-Littmark (ZBL) potential for the pairs of atoms based off of
    implementation by Peter Eastman:

    https://github.com/openmm/nutmeg/blob/main/source/zbl_tensornet.py
    """

    def __init__(self):
        """
        Initializes the ZBL potential module.

        Parameters
        ----------

        """
        super().__init__()

        # note, I think which source of radii should be toggleable
        # since for some systems we will need more than what is provided from this source
        # This maps atomic numbers to covalent radii.  The values are from https://doi.org/10.1063/1.1725697.
        self.radii = {
            1: 0.025,
            3: 0.145,
            4: 0.105,
            5: 0.085,
            6: 0.07,
            7: 0.065,
            8: 0.06,
            9: 0.05,
            11: 0.18,
            12: 0.15,
            13: 0.125,
            14: 0.11,
            15: 0.1,
            16: 0.1,
            17: 0.1,
            19: 0.22,
            20: 0.18,
            21: 0.16,
            22: 0.14,
            23: 0.135,
            24: 0.14,
            25: 0.14,
            26: 0.14,
            27: 0.135,
            28: 0.135,
            29: 0.135,
            30: 0.135,
            31: 0.13,
            32: 0.125,
            33: 0.115,
            34: 0.115,
            35: 0.115,
            37: 0.235,
            38: 0.2,
            39: 0.18,
            40: 0.155,
            41: 0.145,
            42: 0.145,
            43: 0.135,
            44: 0.13,
            45: 0.135,
            46: 0.14,
            47: 0.16,
            48: 0.155,
            49: 0.155,
            50: 0.145,
            51: 0.145,
            52: 0.14,
            53: 0.14,
        }
        # radius_map = torch.tensor(
        #     [radii[n] for n in model_config.atomic_number_map], dtype=torch.float32
        # )
        # self.register_buffer("radius_map", radius_map)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # first let us extract all the releveant data from the data dictionary
        idx_i, idx_j = data["pair_indices"]

        # let us ensure we only consider  unique pairs
        # this avoids having to know if the neighbor list used unique or non unique pairs
        unique_pairs_mask = idx_i < idx_j
        idx_i = idx_i[unique_pairs_mask]
        idx_j = idx_j[unique_pairs_mask]

        # mask pairwise properties
        pairwise_distances = data["d_ij"][unique_pairs_mask]

        atomic_numbers = data["atomic_numbers"]

        atomic_number_i = atomic_numbers[idx_i]
        atomic_number_j = atomic_numbers[idx_j]

        # since we will be working with pairs not individual atoms
        # we need to map the atomic_subsystem_indices onto the pairs
        # since pairs only exist between atoms on the same molecule
        # we can just use one of the pair indices tensors
        system_indices = data["atomic_subsystem_indices"]
        system_indices_of_pair = system_indices[idx_i]

        # generate the radius_i and radius_j tensors based on the atomic numbers in the pairs

        radius_i = torch.tensor(
            [self.radii[int(n)] for n in atomic_number_i],
            dtype=torch.float32,
            device=pairwise_distances.device,
        )
        radius_j = torch.tensor(
            [self.radii[int(n)] for n in atomic_number_j],
            dtype=torch.float32,
            device=pairwise_distances.device,
        )

        # Compute the ZBL potential. 5.29e-2 is the Bohr radius in nm.  All other numbers are magic constants from the ZBL potential.
        # E^{ZBL}_{ij} & = \frac{1}{4\pi\epsilon_0} \frac{Z_i Z_j \,e^2}{r_{ij}} \phi(r_{ij}/a)+ S(r_{ij})
        # a = \frac{0.46850}{Z_{i}^{0.23} + Z_{j}^{0.23}}
        # phi(x) & =  0.18175e^{-3.19980x} + 0.50986e^{-0.94229x} + 0.28022e^{-0.40290x} + 0.02817e^{-0.20162x}
        a = 0.8854 * 5.29177210903e-2 / (atomic_number_i**0.23 + atomic_number_j**0.23)
        d = pairwise_distances / a

        f = (
            0.1818 * torch.exp(-3.2 * d)
            + 0.5099 * torch.exp(-0.9423 * d)
            + 0.2802 * torch.exp(-0.4029 * d)
            + 0.02817 * torch.exp(-0.2016 * d)
        )

        phi_ji = torch.where(
            pairwise_distances < radius_i + radius_j,
            0.5
            * (torch.cos(torch.pi * pairwise_distances / (radius_i + radius_j)) + 1),
            torch.zeros_like(pairwise_distances),
        )

        f *= phi_ji
        # Compute the energy.  The prefactor is 1/(4*pi*eps0) in kJ*nm/mol.
        energy = (
            f * 138.9354576 * atomic_number_i * atomic_number_j / pairwise_distances
        )
        # figure out the number of unique systems so we can initialize the zbl_energy
        num_unique_systems = torch.unique(system_indices)
        zbl_energy = torch.zeros_like(num_unique_systems, dtype=energy.dtype).reshape(
            -1, 1
        )

        # we need to scatter the energy to the correct molecules using the system_indices_of_pair
        data["zbl_energy"] = (
            torch.zeros_like(zbl_energy)
            .scatter_add_(0, system_indices_of_pair.long().unsqueeze(1), energy)
            .detach()
        )

        return data

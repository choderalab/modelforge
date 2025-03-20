from modelforge.curate.units import GlobalUnitSystem, chem_context
from modelforge.curate.properties import (
    PropertyBaseModel,
    PropertyClassification,
    PropertyType,
)

from openff.units import unit

import numpy as np
import copy
from typing import Union, List, Type, Optional

from typing_extensions import Self

from loguru import logger as log


class Record:
    def __init__(self, name: str, append_property: bool = False):

        assert isinstance(name, str)
        self.name = name
        self.per_atom = {}
        self.per_system = {}
        self.meta_data = {}
        self.atomic_numbers = None
        self._n_atoms = -1
        self._n_configs = -1
        self.append_property = append_property

    def __repr__(self):

        output_string = f"name: {self.name}\n"
        if self.n_atoms == -1:
            output_string += f"* n_atoms: cannot be determined, see warnings log\n"
        else:
            output_string += f"* n_atoms: {self.n_atoms}\n"
        if self.n_configs == -1:
            output_string += f"* n_configs: cannot be determined, see warnings log\n"
        else:
            output_string += f"* n_configs: {self.n_configs}\n"
        output_string += "* atomic_numbers:\n"
        output_string += f" -  {self.atomic_numbers}\n"
        output_string += f"* per-atom properties: ({list(self.per_atom.keys())}):\n"
        for key, value in self.per_atom.items():
            output_string += f" -  {value}\n"
        output_string += f"* per-system properties: ({list(self.per_system.keys())}):\n"
        for key, value in self.per_system.items():
            output_string += f" -  {value}\n"
        output_string += f"* meta_data: ({list(self.meta_data.keys())})\n"
        for key, value in self.meta_data.items():
            output_string += f" -  {value}\n"
        return output_string

    @property
    def n_atoms(self):
        """
        Get the number of atoms in the record

        Returns
        -------
            int: number of atoms in the record
        """
        # the validate function will set self._n_atoms to -1 if the number of atoms cannot be determined
        # or if the number of atoms is inconsistent between properties
        # otherwise will set it to the value from the atomic_numbers
        self._validate_n_atoms()
        return self._n_atoms

    @property
    def n_configs(self):
        """
        Get the number of configurations in the record

        Returns
        -------
            int: number of configurations in the record
        """

        # the validate function will set self._n_configs to -1 if the number of configurations cannot be determined
        # or if the number of configurations is inconsistent between properties.
        # will set it to the value from the properties otherwise
        self._validate_n_configs()
        return self._n_configs

    def slice_record(self, min: int = 0, max: int = -1) -> Self:
        """
        Slice the record to only include a subset of configs

        Slicing occurs on all per_atom and per_system properties

        Parameters
        ----------
        min: int
            Starting index for slicing.
        max: int
            ending index for slicing.

        Returns
        -------
            Record: Slice record.
        """
        new_record = copy.deepcopy(self)
        for key, value in self.per_atom.items():
            new_record.per_atom[key].value = value.value[min:max]
        for key, value in self.per_system.items():
            new_record.per_system[key].value = value.value[min:max]

        return new_record

    def contains_atomic_numbers(self, atomic_numbers_of_interest: np.ndarray) -> bool:
        """
        Check if the atomic numbers in the record are a subset of the input.

        Parameters
        ----------
        atomic_numbers_of_interest: np.ndarray
            Array of atomic numbers to check against.

        Returns
        -------
            bool: True if the atomic numbers are contained within the input, False otherwise.
        """

        if self.atomic_numbers is None:
            log.warning(
                f"No atomic numbers set for record {self.name}. Cannot compare."
            )
            raise ValueError(
                f"No atomic numbers set for record {self.name}. Cannot compare."
            )
        status = set(self.atomic_numbers.value.flatten()).issubset(
            atomic_numbers_of_interest
        )

        return status

    def remove_high_force_configs(
        self, max_force: unit.Quantity, force_key: str = "forces"
    ):
        """
        Remove configurations with forces greater than the max_force

        Parameters
        ----------
        max_force, unit.Quantity
            Maximum force to allow in the record.
        force_key: str, optional, default="forces"
            Name of the property to use for filtering.

        Returns
        -------
        record: Record
            Copy of the Record with configurations removed.
        """

        if force_key not in self.per_atom.keys():
            log.warning(f"Force key {force_key} not found in record {self.name}.")

            raise ValueError(f"Force key {force_key} not found in record {self.name}.")

        if self.per_atom[force_key].property_type != "force":
            log.warning(f"Property {force_key} is not a force property.")

            raise ValueError(f"Property {force_key} is not a force property.")
        assert isinstance(max_force, unit.Quantity)

        # get the indices of the configurations that have forces less than the max_force
        indices_to_include = []
        for i in range(self.n_configs):
            force_magnitude = (
                np.abs(self.per_atom[force_key].value[i])
                * self.per_atom[force_key].units
            )
            if np.max(force_magnitude) <= max_force:
                indices_to_include.append(i)

        return self.remove_configs(indices_to_include)

    def remove_configs(self, indices_to_include: List[int]):
        """
        Remove configurations not in the indices_to_include list

        Parameters
        ----------
        indices_to_include: List[int]
            List of indices to keep in the record.

        Returns
        -------
        Record: Copy of the record with configurations removed.

        """

        new_record = copy.deepcopy(self)
        for key, value in self.per_atom.items():
            new_record.per_atom[key].value = value.value[indices_to_include]
        for key, value in self.per_system.items():
            new_record.per_system[key].value = value.value[indices_to_include]

        return new_record

    def to_dict(self):
        """
        Convert the record to a dictionary

        Returns
        -------
            dict: dictionary representation of the record
        """
        return {
            "name": self.name,
            "n_atoms": self.n_atoms,
            "n_configs": self.n_configs,
            "atomic_numbers": self.atomic_numbers,
            "per_atom": self.per_atom,
            "per_system": self.per_system,
            "meta_data": self.meta_data,
        }

    def _validate_n_atoms(self):
        """
        Validate the number of atoms in the record by checking that all per_atom properties have the same number of atoms as the atomic numbers.

        Returns
        -------
            bool: True if the number of atoms is defined and consistent, False otherwise.
        """
        self._n_atoms = -1
        if self.atomic_numbers is not None:
            for key, value in self.per_atom.items():
                if value.n_atoms != self.atomic_numbers.n_atoms:
                    log.warning(
                        f"Number of atoms for property {key} in record {self.name} does not match the number of atoms in the atomic numbers."
                    )
                    return False
        else:
            log.warning(
                f"No atomic numbers set for record {self.name}. Cannot validate number of atoms."
            )
            return False
        self._n_atoms = self.atomic_numbers.n_atoms
        return True

    def _validate_n_configs(self):
        """
        Validate the number of configurations in the record by checking that all properties have the same number of configurations.

        Returns
        -------
            bool: True if the number of configurations is defined and consistent, False otherwise.
        """
        n_configs = []
        for key, value in self.per_atom.items():
            n_configs.append(value.n_configs)
        for key, value in self.per_system.items():
            n_configs.append(value.n_configs)
        if len(n_configs) != 0:
            if all([n == n_configs[0] for n in n_configs]):
                self._n_configs = n_configs[0]
                return True
            else:
                self._n_configs = -1
                log.warning(
                    f"Number of configurations for properties in record {self.name} are not consistent."
                )
                for key, value in self.per_atom.items():
                    log.warning(f" - {key} : {value.n_configs}")
                for key, value in self.per_system.items():
                    log.warning(f" - {key} : {value.n_configs}")
                return False
        else:
            log.warning(
                f"No properties found in record {self.name}. Cannot determine the number of configurations."
            )
            self._n_configs = -1
            return False

    def validate(self):
        """
        Validate the record to ensure that the number of atoms and configurations are consistent across all properties.

        Returns
        -------
            True if the record validated, False otherwise.
        """
        if self._validate_n_atoms() and self._validate_n_configs():
            return True
        return False

    def add_properties(self, properties: List[Type[PropertyBaseModel]]):
        """
        Add a list of properties to the record.

        Parameters
        ----------
        properties: List[Type[PropertyBaseModel]]
            List of properties to add to the record.
        Returns
        -------

        """
        for property in properties:
            self.add_property(property)

    def add_property(self, property: Type[PropertyBaseModel]):
        """
        Add a property to the record.

        Parameters
        ----------
        property: Type[PropertyBaseModel]
            Property to add to the record.
        Returns
        -------

        """
        if property.classification == PropertyClassification.atomic_numbers:
            # we will not allow atomic numbers to be set twice
            if self.atomic_numbers is not None:
                raise ValueError(f"Atomic numbers already set for record {self.name}")

            self.atomic_numbers = property.model_copy(deep=True)

            # Note, the number of atoms will always be set by the atomic_numbers property.
            # We will later validate that per_atom properties are consistent with this value later
            # since we are not enforcing that atomic_numbers need to be set before any other property

        elif property.classification == PropertyClassification.meta_data:
            if property.name in self.meta_data.keys():
                log.warning(
                    f"Metadata with name {property.name} already exists in the record {self.name}."
                )
                raise ValueError(
                    f"Metadata with name {property.name} already exists in the record {self.name}"
                )

            elif property.name in self.per_atom.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_atom property."
                )
            elif property.name in self.per_system.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_system property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them, not the MetaData class."
                )
            self.meta_data[property.name] = property.model_copy(deep=True)

        elif property.classification == PropertyClassification.per_atom:
            if property.name in self.per_system.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_system property."
                )
            elif property.name in self.meta_data.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a meta_data property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
                )
            elif property.name in self.per_atom.keys():
                if self.append_property == False:
                    error_msg = f"Property with name {property.name} already exists in the record {self.name}."
                    error_msg += (
                        f"Set append_property=True to append to the existing property."
                    )
                    raise ValueError(error_msg)
                # if the property already exists, we will use vstack to add it to the existing array
                # after first checking that the dimensions are consistent
                # note we do not check shape[0], as that corresponds to the number of configurations
                assert (
                    self.per_atom[property.name].value.shape[1]
                    == property.value.shape[1]
                ), f"{self.name}: n_atoms of {property.name} does not: {property.value.shape[1]} != {self.per_atom[property.name].value.shape[1]}."
                assert (
                    self.per_atom[property.name].value.shape[2]
                    == property.value.shape[2]
                )
                # In order to append to the array, everything needs to have the same units
                # We will use the units of the first property that was added

                temp_array = property.value
                if property.units != self.per_atom[property.name].units:
                    temp_array = (
                        unit.Quantity(property.value, property.units)
                        .to(
                            self.per_atom[property.name].units,
                            "chem",
                        )
                        .magnitude
                    )
                self.per_atom[property.name].value = np.vstack(
                    (
                        self.per_atom[property.name].value,
                        temp_array,
                    )
                )

            else:
                self.per_atom[property.name] = property.model_copy(deep=True)
        elif property.classification == PropertyClassification.per_system:
            if property.name in self.per_atom.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a per_atom property."
                )
            elif property.name in self.meta_data.keys():
                raise ValueError(
                    f"Property with name {property.name} already exists in the record {self.name}, but as a meta_data property."
                )
            elif property.name == "atomic_numbers":
                raise ValueError(
                    f"The name atomic_numbers is reserved. Use AtomicNumbers to define them."
                )
            elif property.name in self.per_system.keys():
                if self.append_property == False:
                    error_msg = f"Property with name {property.name} already exists in the record {self.name}."
                    error_msg += (
                        f"Set append_property=True to append to the existing property."
                    )
                    raise ValueError(error_msg)

                assert (
                    self.per_system[property.name].value.shape[1]
                    == property.value.shape[1]
                )
                temp_array = property.value
                if property.units != self.per_system[property.name].units:
                    temp_array = (
                        unit.Quantity(property.value, property.units)
                        .to(
                            self.per_system[property.name].units,
                            "chem",
                        )
                        .magnitude
                    )

                self.per_system[property.name].value = np.vstack(
                    (
                        self.per_system[property.name].value,
                        temp_array,
                    )
                )
            else:
                self.per_system[property.name] = property.model_copy(deep=True)

    def keys(self):
        """Return the property keys in the record"""
        if self.atomic_numbers is not None:
            return (
                list(self.per_atom.keys())
                + list(self.per_system.keys())
                + list(self.meta_data.keys())
                + ["atomic_numbers"]
            )
        else:
            return (
                list(self.per_atom.keys())
                + list(self.per_system.keys())
                + list(self.meta_data.keys())
            )

    def to_rdkit(self, position_key: str = "positions", infer_bonds: bool = True):
        """
        Convert the record to an RDKit molecule

        Parameters
        ----------
        position_key: str, optional, default="positions"
            Name of the property to use for the positions of the atoms
        infer_bonds: bool, optional, default=True
            If True, infer bonds in the molecule based on the first configuration
        Returns
        -------
        RWMol RDKit Molecule
        """

        from rdkit import Chem
        from rdkit.Geometry import Point3D

        mol = Chem.RWMol()
        from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

        if self.atomic_numbers is None:
            log.warning(
                f"No atomic numbers set for record {self.name}. Cannot convert to RDKit molecule."
            )
            raise ValueError(
                f"No atomic numbers set for record {self.name}. Cannot convert to RDKit molecule."
            )
        if position_key not in self.per_atom.keys():
            log.warning(
                f"Position key {position_key} not found in record {self.name}. Cannot convert to RDKit molecule."
            )
            raise ValueError(
                f"Position key {position_key} not found in record {self.name}. Cannot convert to RDKit molecule."
            )

        for i in range(self.n_atoms):
            atom = Chem.Atom(_ATOMIC_NUMBER_TO_ELEMENT[self.atomic_numbers.value[i][0]])
            mol.AddAtom(atom)

        for ii in range(self.n_configs):

            conf = Chem.Conformer(self.n_atoms)
            conf.SetId(ii)
            pos = (
                self.per_atom[position_key].value[ii]
                * self.per_atom[position_key].units
            )

            for j in range(pos.shape[0]):
                conf.SetAtomPosition(
                    j,
                    Point3D(
                        pos[j][0].to(unit.angstrom).m,
                        pos[j][1].to(unit.angstrom).m,
                        pos[j][2].to(unit.angstrom).m,
                    ),
                )
            mol.AddConformer(conf)
            if ii == 0 and infer_bonds:
                from rdkit.Chem import rdDetermineBonds

                rdDetermineBonds.DetermineConnectivity(mol)
        return mol

    def remove_property(self, property_key: str):
        """Remove a property from the record

        Parameters
        ----------
        property_key: str
            Key of the property to remove

        Returns
        -------

        """
        if property_key == "atomic_numbers":
            self.atomic_numbers = None
        elif property_key in self.per_atom.keys():
            self.per_atom.pop(property_key)
        elif property_key in self.per_system.keys():
            self.per_system.pop(property_key)
        elif property_key in self.meta_data.keys():
            self.meta_data.pop(property_key)
        else:
            log.warning(
                f"Property with key {property_key} not found in record {self.name}"
            )
            raise ValueError(
                f"Property with key {property_key} not found in record {self.name}"
            )

    def get_property_value(
        self, property_key: str
    ) -> Union[np.ndarray, unit.Quantity, str, float, int, List]:
        """
        Function to return a copy of the value field for a given property.

        Units will be added if the property is not unitless.


        Parameters
        ----------
        property_key: str
            Key of the property to return

        Returns
        -------
        Union[np.ndarray, unit.Quantity, str, float, int, List]
            Value of the property, with units attached if appropriate
        """

        if property_key == "atomic_numbers":
            return copy.deepcopy(self.atomic_numbers.value)
        elif property_key in self.per_atom.keys():
            if self.per_atom[property_key].units != unit.dimensionless:
                return unit.Quantity(
                    self.per_atom[property_key].value, self.per_atom[property_key].units
                )
            else:
                return copy.deepcopy(self.per_atom[property_key].value)
        elif property_key in self.per_system.keys():
            if self.per_system[property_key].units != unit.dimensionless:
                return unit.Quantity(
                    self.per_system[property_key].value,
                    self.per_system[property_key].units,
                )
            else:
                return copy.deepcopy(self.per_system[property_key].value)
        elif property_key in self.meta_data.keys():
            if self.meta_data[property_key].units != unit.dimensionless:
                return unit.Quantity(
                    self.meta_data[property_key].value,
                    self.meta_data[property_key].units,
                )
            else:
                return copy.deepcopy(self.meta_data[property_key].value)
        else:
            log.warning(
                f"Property with key {property_key} not found in record {self.name}"
            )
            raise ValueError(
                f"Property with key {property_key} not found in record {self.name}"
            )

    def get_property(self, property_key: str) -> PropertyBaseModel:
        """
        Function to return a copy of the property with the given key

        Parameters
        ----------
        property_key, str
            Key of the property to return

        Returns
        -------
        property: PropertyBaseModel

        """

        if property_key == "atomic_numbers":
            return copy.deepcopy(self.atomic_numbers)
        elif property_key in self.per_atom.keys():
            return copy.deepcopy(self.per_atom[property_key])
        elif property_key in self.per_system.keys():
            return copy.deepcopy(self.per_system[property_key])
        elif property_key in self.meta_data.keys():
            return copy.deepcopy(self.meta_data[property_key])
        else:
            log.warning(
                f"Property with key {property_key} not found in record {self.name}"
            )
            raise ValueError(
                f"Property with key {property_key} not found in record {self.name}"
            )


def calculate_max_bond_length_change(
    record: Record, bonds: List[List[int]], positions_key: str = "positions"
) -> list:
    """
    Calculate the maximum bond length change for each configuration, given a list of bonds
    Note, the first configuration is used as the reference.

    Parameters
    ----------
    record: Record
        The record to calculate the bond length changes for.
    bonds: List[List[int]]
        A list of bonds to calculate the bond length changes for. This is a list contain a pair of indices for each bond
    positions_key: str, optional, default="positions"
        The key in the per_atom dictionary to use for positions

    Returns
    -------
        List[unit.Quantity]
            A list of the maximum bond length changes for each configuration (length= n_configs).
            Note, since the first configuration is used as the reference, the first entry is 0.
    """

    max_changes = []

    initial_distances = []
    for bond in bonds:
        d1 = record.per_atom[positions_key].value[0][bond[0]]
        d2 = record.per_atom[positions_key].value[0][bond[1]]
        initial_distance = np.linalg.norm(d1 - d2)
        initial_distances.append(initial_distance)

    max_changes.append(0.0 * record.per_atom[positions_key].units)

    for i in range(1, record.n_configs):
        changes_temp = [0]

        for b, bond in enumerate(bonds):

            d1 = record.per_atom[positions_key].value[i][bond[0]]
            d2 = record.per_atom[positions_key].value[i][bond[1]]
            distance = np.linalg.norm(d1 - d2)
            changes_temp.append(np.abs(distance - initial_distances[b]))
        max_changes.append(np.max(changes_temp) * record.per_atom[positions_key].units)
    return max_changes


def infer_bonds(
    record: Record, position_key: str = "positions", config_id: int = 0
) -> List[List[int]]:
    """
    Infer bonds from a record using RDKit

    Parameters
    ----------
    record: Record
        The record to infer bonds from
    position_key: str, optional, default="positions"
        The key in the per_atom dictionary to use for positions
    config_id: int, optional, default=0
        The configuration to use for inferring bonds

    Returns
    -------
    List[List[int]]
        A list of contain a pair of indices for each bond
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

    mol = Chem.RWMol()
    atomic_numbers = record.atomic_numbers.value.reshape(-1)
    for i in range(atomic_numbers.shape[0]):
        atom = Chem.Atom(_ATOMIC_NUMBER_TO_ELEMENT[atomic_numbers[i]])
        mol.AddAtom(atom)

    conf = Chem.Conformer()
    positions = (
        record.per_atom[position_key].value[config_id]
        * record.per_atom[position_key].units
    )

    # convert to angstroms for RDKIT
    positions = positions.to(unit.angstrom).magnitude
    for i in range(positions.shape[0]):
        conf.SetAtomPosition(
            i,
            Point3D(
                positions[i][0],
                positions[i][1],
                positions[i][2],
            ),
        )
    mol.AddConformer(conf)
    from rdkit.Chem import rdDetermineBonds

    rdDetermineBonds.DetermineConnectivity(mol)
    bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]

    return bonds

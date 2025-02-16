from abc import ABC, abstractmethod

from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from openff.units import unit
from modelforge.curate.properties import (
    AtomicNumbers,
    PartialCharges,
    Positions,
    DipoleMomentPerSystem,
    DipoleMomentScalarPerSystem,
)


class DatasetCuration(ABC):
    """
    Abstract base class with routines to fetch and process a dataset into a curated hdf5 file.
    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_dir: Optional[str] = "./",
        local_cache_dir: Optional[str] = "./datasets_cache",
        version_select: str = "latest",
    ):
        """
        Sets input and output parameters.

        Parameters
        ----------
        hdf5_file_name: str, required
            Name of the hdf5 file that will be generated.
        output_file_dir: str, optional, default='./'
            Location to write the output hdf5 file.
        local_cache_dir: str, optional, default='./qm9_datafiles'
            Location to save downloaded dataset.
        version_select: str, optional, default='latest'
            Version of the dataset to use as defined in the associated yaml file.

        """
        import os

        self.hdf5_file_name = hdf5_file_name
        self.output_file_dir = output_file_dir
        # make sure we can handle a path with a ~ in it
        self.local_cache_dir = os.path.expanduser(local_cache_dir)
        self.version_select = version_select
        os.makedirs(self.local_cache_dir, exist_ok=True)

        # initialize parameter information
        self._init_dataset_parameters()

    @abstractmethod
    def _init_dataset_parameters(self):
        """
        Initialize the dataset parameters.
        """
        pass

    def total_records(self):
        """
        Returns the total number of records in the dataset.

        Returns
        -------
        int
            total number of records in the dataset
        """
        return self.dataset.total_records()

    def total_configs(self):
        """
        Returns the total number of configurations in the dataset.

        Returns
        -------
        int
            total number of conformers in the dataset
        """
        return self.dataset.total_configs()

    def _calc_center_of_mass(
        self, atomic_numbers: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute the center of mass of a system

        parameters
        ----------
        atomic_numbers: np.ndarray, required
            atomic numbers of the atoms in the system
        positions: np.ndarray, required
            positions of the atoms in the system

        Returns
        -------
        np.ndarray
            center of mass of the system
        """

        from openff.units.elements import MASSES
        from openff.units import unit
        import numpy as np

        atomic_masses = np.array(
            [
                MASSES[atomic_number].m
                for atomic_number in atomic_numbers.reshape(-1).tolist()
            ]
        )

        center_of_mass = np.einsum(
            "i,ij->j",
            atomic_masses,
            positions / np.sum(atomic_masses),
        )
        return center_of_mass

    def compute_dipole_moment(
        self,
        atomic_numbers: AtomicNumbers,
        partial_charges: PartialCharges,
        positions: Positions,
        dipole_moment_scalar: Optional[DipoleMomentScalarPerSystem] = None,
    ) -> DipoleMomentPerSystem:
        """
        Compute the per-system  dipole moment from the atomic numbers, partial charges, and positions,
        rescaling to give the same magnitude to match the
        magnitude of the dipole moment (i.e., scalar) if provided.

        Parameters
        ----------
        atomic_numbers: AtomicNumbers
            atomic_numbers of the atoms in the system
        partial_charges: PartialCharges
            partial charges of the atoms in the system
        positions: Positions
            positions of the atoms in the system
        dipole_moment_scalar: Optional[DipoleMomentScalarPerSystem], optional, default=None
            scalar dipole moment to rescale the computed dipole moment

        Returns
        -------
            DipoleMomentPerSystem
                computed per system dipole moment
        """
        # from openff.units.elements import MASSES
        from openff.units import unit
        import numpy as np

        # atomic_masses = np.array(
        #     [
        #         MASSES[atomic_number].m
        #         for atomic_number in atomic_numbers.value.reshape(-1).tolist()
        #     ]
        # )

        dipole_moment_list = []
        # compute the center of mass
        for config in range(positions.value.shape[0]):
            # center_of_mass = np.einsum(
            #     "i,ij->j",
            #     atomic_masses,
            #     positions.value[config] / np.sum(atomic_masses),
            # )
            center_of_mass = self._calc_center_of_mass(
                atomic_numbers.value[config], positions.value[config]
            )
            pos = positions.value[config] - center_of_mass

            dm_temp = np.einsum(
                "i,ij->j", partial_charges.value[config].reshape(-1), pos
            ).reshape(1, 3)

            if dipole_moment_scalar is not None:
                if dipole_moment_scalar.value[config] == 0:
                    dm_temp = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
                else:
                    ratio = (
                        np.linalg.norm(dm_temp)
                        / (
                            dipole_moment_scalar.value[config]
                            * dipole_moment_scalar.units
                        )
                        .to(positions.units * unit.elementary_charge)
                        .m
                    )
                    dm_temp = dm_temp / ratio

            dipole_moment_list.append(dm_temp)

        return DipoleMomentPerSystem(
            value=np.array(dipole_moment_list).reshape(-1, 3),
            units=positions.units * unit.elementary_charge,
        )

    def write_hdf5_and_json_files(self, file_name: str, file_path: str):
        """
        Write the dataset to an hdf5 file and a json file.
        """
        # save out the dataset to hdf5
        checksum = self.dataset.to_hdf5(file_name=file_name, file_path=file_path)
        # chop off .hdf5 extension and add .json
        json_file_name = file_name.replace(".hdf5", ".json")

        # save a summary to json
        self.dataset.summary_to_json(
            file_path=file_path,
            file_name=json_file_name,
            hdf5_checksum=checksum,
            hdf5_file_name=file_name,
        )

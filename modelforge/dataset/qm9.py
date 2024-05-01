from typing import List

from .dataset import HDF5Dataset


class QM9Dataset(HDF5Dataset):
    """
    Data class for handling QM9 data.

    This class provides utilities for processing and interacting with QM9 data
    stored in HDF5 format.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "QM9".
    for_unit_testing : bool
        If set to True, a subset of the dataset is used for unit testing purposes; by default False.
    local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
    Examples
    --------
    >>> data = QM9Dataset()
    >>> data._download()
    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(
        "atomic_numbers", "geometry", "internal_energy_at_0K", "charges"
    )

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "internal_energy_at_0K",
        "internal_energy_at_298.15K",
        "enthalpy_at_298.15K",
        "free_energy_at_298.15K",
        "heat_capacity_at_298.15K",
        "zero_point_vibrational_energy",
        "electronic_spatial_extent",
        "lumo-homo_gap",
        "energy_of_homo",
        "energy_of_lumo",
        "rotational_constant_A",
        "rotational_constant_B",
        "rotational_constant_C",
        "dipole_moment",
        "isotropic_polarizability",
        "charges",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "QM9",
        for_unit_testing: bool = False,
        local_cache_dir: str = ".",
        overwrite: bool = False,
    ) -> None:
        """
        Initialize the QM9Data class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "QM9".
        for_unit_testing : bool, optional
            If set to True, a subset of the dataset is used for unit testing purposes; by default False.
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".

        Examples
        --------
        >>> data = QM9Dataset()  # Default dataset
        >>> test_data = QM9Dataset(for_unit_testing=True)  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "internal_energy_at_0K",
            "charges",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        super().__init__(
            f"{local_cache_dir}/{dataset_name}_cache.hdf5.gz",
            f"{local_cache_dir}/{dataset_name}_processed.npz",
            local_cache_dir=local_cache_dir,
        )
        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing
        self.test_url = (
            "https://github.com/wiederm/gm9/raw/main/qm9_dataset_n100.hdf5.gz"
        )
        self.full_url = "https://github.com/wiederm/gm9/raw/main/qm9.hdf5.gz"
        self._ase = {
            "H": -1313.4668615546,
            "C": -99366.70745535441,
            "N": -143309.9379722722,
            "O": -197082.0671774158,
            "F": -261811.54555874597,
        }

    @property
    def atomic_self_energies(self):
        from modelforge.potential.processing import AtomicSelfEnergies

        return AtomicSelfEnergies(energies=self._ase)

    @property
    def properties_of_interest(self) -> List[str]:
        """
        Getter for the properties of interest.
        The order of this list determines also the order provided in the __getitem__ call
        from the PytorchDataset.

        Returns
        -------
        List[str]
            List of properties of interest.

        """
        return self._properties_of_interest

    @property
    def available_properties(self) -> List[str]:
        """
        List of available properties in the dataset.

        Returns
        -------
        List[str]
            List of available properties in the dataset.

        Examples
        --------
        >>> data = QM9Dataset()
        >>> data.available_properties
        ['geometry', 'atomic_numbers', 'return_energy']
        """
        return self._available_properties

    @properties_of_interest.setter
    def properties_of_interest(self, properties_of_interest: List[str]) -> None:
        """
        Setter for the properties of interest.
        The order of this list determines also the order provided in the __getitem__ call
        from the PytorchDataset

        Parameters
        ----------
        properties_of_interest : List[str]
            List of properties of interest.

        Examples
        --------
        >>> data = QM9Dataset()
        >>> data.properties_of_interest = ["geometry", "atomic_numbers", "return_energy"]
        """
        if not set(properties_of_interest).issubset(self._available_properties):
            raise ValueError(
                f"Properties of interest must be a subset of {self._available_properties}"
            )
        self._properties_of_interest = properties_of_interest

    def _download(self) -> None:
        """
        Download the hdf5 file containing the data from Google Drive.

        Examples
        --------
        >>> data = QM9Dataset()
        >>> data.download()  # Downloads the dataset from Google Drive

        """
        from modelforge.dataset.utils import _download_from_url

        url = self.test_url if self.for_unit_testing else self.full_url
        _download_from_url(url, self.raw_data_file)

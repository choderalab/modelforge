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
        force_download: bool = False,
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
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.

        Examples
        --------
        >>> data = QM9Dataset()  # Default dataset
        >>> test_data = QM9Dataset(for_unit_testing=True)  # Testing subset
        """

        self.force_download = force_download

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "internal_energy_at_0K",
            "charges",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        super().__init__()
        # super().__init__(
        #     f"{local_cache_dir}/{dataset_name}_cache.hdf5.gz",
        #     f"{local_cache_dir}/{dataset_name}_processed.npz",
        #     local_cache_dir=local_cache_dir,
        # )
        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing
        self.local_cache_dir = local_cache_dir

        # self.test_url = (
        #     "https://github.com/wiederm/gm9/raw/main/qm9_dataset_n100.hdf5.gz"
        # )
        # to ensure we have the same checksum each time, we need to copy the permalink to the file
        # otherwise the checksum will change each time we download the file from the same link because of how github works
        # self.test_url = "https://github.com/wiederm/gm9/blob/264af75e41e6e296f400d9a1019f082b21d5bc36/qm9_dataset_n100.hdf5.gz"
        self.test_url = "https://www.dropbox.com/scl/fi/9jeselknatcw9xi0qp940/qm9_dataset_n100.hdf5.gz?rlkey=50of7gn2s12i65c6j06r73c97&dl=1"
        # self.test_url = "https://drive.google.com/file/d/1yE8krZo3MMI84unZH0_H01C4m5RkXJ8d/view?usp=sharing"
        # self.full_url = "https://github.com/wiederm/gm9/raw/main/qm9.hdf5.gz"

        # to ensure we have the same checksum each time, we need to copy the permalink to the file
        self.full_url = "https://www.dropbox.com/scl/fi/4wu7zlpuuixttp0u741rv/qm9_dataset.hdf5.gz?rlkey=nszkqt2t4kmghih5mt4ssppvo&dl=1"
        # self.full_url = "https://github.com/wiederm/gm9/blob/264af75e41e6e296f400d9a1019f082b21d5bc36/qm9.hdf5.gz"
        self._ase = {
            "H": -1313.4668615546,
            "C": -99366.70745535441,
            "N": -143309.9379722722,
            "O": -197082.0671774158,
            "F": -261811.54555874597,
        }
        from loguru import logger

        if self.for_unit_testing:

            self.url = self.test_url
            self.md5_raw_checksum = "af3afda5c3265c9c096935ab060f537a"
            self.raw_data_file = "qm9_dataset_n100.hdf5.gz"

            # define the name and checksum of the unzipped file
            self.unzipped_data_file = "qm9_dataset_n100.hdf5"
            self.md5_unzipped_checksum = "77df0e1df7a5ec5629be52181e82a7d7"

            self.processed_data_file = "qm9_dataset_n100_processed.npz"
            self.md5_processed_checksum = "9d671b54f7b9d454db9a3dd7f4ef2020"
            logger.info("Using test dataset")

        else:
            self.url = self.full_url
            self.md5_raw_checksum = "d172127848de114bd9cc47da2bc72566"
            self.raw_data_file = "qm9_dataset.hdf5.gz"

            self.unzipped_data_file = "qm9_dataset.hdf5"
            self.md5_unzipped_checksum = "0b22dc048f3361875889f832527438db"

            self.processed_data_file = "qm9_dataset_processed.npz"
            self.md5_processed_checksum = "62d98cf38bcf02966e1fa2d9e44b3fa0"

            logger.info("Using full dataset")

    @property
    def atomic_self_energies(self):
        from modelforge.potential.utils import AtomicSelfEnergies

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
        from modelforge.utils.remote import download_from_url

        download_from_url(
            url=self.url,
            md5_checksum=self.md5_raw_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.raw_data_file,
            force_download=self.force_download,
        )
        # from modelforge.dataset.utils import _download_from_url
        #
        # url = self.test_url if self.for_unit_testing else self.full_url
        # _download_from_url(url, self.raw_data_file)

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
        Z="atomic_numbers",
        R="geometry",
        E="internal_energy_at_0K",  # Q="charges"
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
        regenerate_cache=False,
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
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            previously downloaded files, if available; by default False.
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

        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing
        from openff.units import unit

        # atomic self energies
        self._ase = {
            "H": -1313.4668615546 * unit.kilojoule_per_mole,
            "C": -99366.70745535441 * unit.kilojoule_per_mole,
            "N": -143309.9379722722 * unit.kilojoule_per_mole,
            "O": -197082.0671774158 * unit.kilojoule_per_mole,
            "F": -261811.54555874597 * unit.kilojoule_per_mole,
        }
        from loguru import logger

        # We need to define the checksums for the various files that we will be dealing with to load up the data
        # There are 3 files types that need name/checksum defined, of extensions hdf5.gz, hdf5, and npz.

        # note, need to change the end of the url to dl=1 instead of dl=0 (the default when you grab the share list), to ensure the same checksum each time we download
        self.test_url = "https://www.dropbox.com/scl/fi/oe2tooxwrkget75zwrfey/qm9_dataset_ntc_1000.hdf5.gz?rlkey=6hfb8ge0pqf4tly15rmdsthmw&st=tusk38vt&dl=1"
        self.full_url = "https://www.dropbox.com/scl/fi/4wu7zlpuuixttp0u741rv/qm9_dataset.hdf5.gz?rlkey=nszkqt2t4kmghih5mt4ssppvo&dl=1"

        if self.for_unit_testing:
            url = self.test_url
            gz_data_file = {
                "name": "qm9_dataset_nc_1000.hdf5.gz",
                "md5": "dc8ada0d808d02c699daf2000aff1fe9",
                "length": 1697917,
            }
            hdf5_data_file = {
                "name": "qm9_dataset_nc_1000.hdf5",
                "md5": "305a0602860f181fafa75f7c7e3e6de4",
            }
            processed_data_file = {
                "name": "qm9_dataset_nc_1000_processed.npz",
                # checksum of otherwise identical npz files are different if using 3.11 vs 3.9/10
                # we will therefore skip checking these files
                "md5": None,
            }

            logger.info("Using test dataset")

        else:
            url = self.full_url
            gz_data_file = {
                "name": "qm9_dataset.hdf5.gz",
                "md5": "d172127848de114bd9cc47da2bc72566",
                "length": 267228348,
            }

            hdf5_data_file = {
                "name": "qm9_dataset.hdf5",
                "md5": "0b22dc048f3361875889f832527438db",
            }

            processed_data_file = {
                "name": "qm9_dataset_processed.npz",
                "md5": None,
            }

            logger.info("Using full dataset")

        # to ensure that that we are consistent in our naming, we need to set all the names and checksums in the HDF5Dataset class constructor
        super().__init__(
            url=url,
            gz_data_file=gz_data_file,
            hdf5_data_file=hdf5_data_file,
            processed_data_file=processed_data_file,
            local_cache_dir=local_cache_dir,
            force_download=force_download,
            regenerate_cache=regenerate_cache,
        )

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
        Download the hdf5 file containing the data from Dropbox.

        Examples
        --------
        >>> data = QM9Dataset()
        >>> data.download()  # Downloads the dataset from Google Drive

        """
        # Right now this function needs to be defined for each dataset.
        # once all datasets are moved to zenodo, we should only need a single function defined in the base class
        from modelforge.utils.remote import download_from_url

        download_from_url(
            url=self.url,
            md5_checksum=self.gz_data_file["md5"],
            output_path=self.local_cache_dir,
            output_filename=self.gz_data_file["name"],
            force_download=self.force_download,
        )
        # from modelforge.dataset.utils import _download_from_url
        #
        # url = self.test_url if self.for_unit_testing else self.full_url
        # _download_from_url(url, self.raw_data_file)

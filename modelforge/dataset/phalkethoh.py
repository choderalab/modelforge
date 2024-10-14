"""
Data class for handling OpenFF Sandbox CHO PhAlkEthOH v1.0 dataset.
"""

from typing import List

from .dataset import HDF5Dataset


class PhAlkEthOHDataset(HDF5Dataset):
    """
    Data class for handling OpenFF Sandbox CHO PhAlkEthOH v1.0 dataset.

    This class provides utilities for processing and interacting with PhAlkEthOH dataset
    stored in HDF5 format.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "PhAlkEthOH".
    version_select : str
        Select the version of the dataset to use, default will provide the "latest".
        "latest_test" will select the testing subset of 1000 conformers.
        A version name can  be specified that corresponds to an entry in the associated yaml file, e.g., "full_dataset_v0".
    local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
    force_download: bool, optional
        If set to True, we will download the dataset even if it already exists; by default False.
    regenerate_cache: bool, optional
        If set to True, we will regenerate the npz cache file even if it already exists, using
        previously downloaded files, if available; by default False.

    Examples
    --------
    >>> data = PhAlkEthOHDataset()
    >>> data._download()
    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(
        atomic_numbers="atomic_numbers",
        positions="geometry",
        E="dft_total_energy",
        F="dft_total_force",
    )

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "dft_total_energy",
        "dft_total_force",
        "scf_dipole",
        "total_charge",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "PhAlkEthOH",
        version_select: str = "latest",
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache=False,
    ) -> None:
        """
        Initialize the PhAlkEthOHData class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "PhAlkEthOH".
        version_select : str,optional
            Select the version of the dataset to use, default will provide the "latest".
            "latest_test" will select the testing subset of 1000 conformers.
            A version name can  be specified that corresponds to an entry in the associated yaml file, e.g., "full_dataset_v0".
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            previously downloaded files, if available; by default False.
        Examples
        --------
        >>> data = PhAlkEthOHDataset()  # Default dataset
        >>> test_data = PhAlkEthOHDataset(version_select="latest_test")  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "total_charge",
            "scf_dipole",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest

        self.dataset_name = dataset_name
        self.version_select = version_select
        from openff.units import unit

        from loguru import logger

        from importlib import resources
        from modelforge.dataset import yaml_files
        import yaml

        # Reference energies, in hartrees, computed with Psi4 1.5 wB97M-D3BJ / def2-TZVPPD.
        # copied from spice 1.1.4

        self._ase = {
            "C": -37.87264507233593 * unit.hartree,
            "H": -0.498760510048753 * unit.hartree,
            "O": -75.11317840410095 * unit.hartree,
        }

        yaml_file = resources.files(yaml_files) / "PhAlkEthOH.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        # make sure we have the correct yaml file
        assert data_inputs["dataset"] == "PhAlkEthOH"

        if self.version_select == "latest":
            # in the yaml file, the entry latest will define the name of the version to use
            dataset_version = data_inputs["latest"]
            logger.info(f"Using the latest dataset: {dataset_version}")
        elif self.version_select == "latest_test":
            dataset_version = data_inputs["latest_test"]
            logger.info(f"Using the latest test dataset: {dataset_version}")
        else:
            dataset_version = self.version_select
            logger.info(f"Using dataset version {dataset_version}")

        url = data_inputs[dataset_version]["url"]

        # fetch the dictionaries that defined the size, md5 checksums (if provided) and filenames of the data files
        gz_data_file = data_inputs[dataset_version]["gz_data_file"]
        hdf5_data_file = data_inputs[dataset_version]["hdf5_data_file"]
        processed_data_file = data_inputs[dataset_version]["processed_data_file"]

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
        >>> data =  PhAlkEthOHDataset()
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
        >>> data =  PhAlkEthOHDataset()
        >>> data.properties_of_interest = ["geometry", "atomic_numbers", "dft_total_energy"]
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
        >>> data =  PhAlkEthOHDataset()
        >>> data.download()  # Downloads the dataset

        """
        # Right now this function needs to be defined for each dataset.
        # once all datasets are moved to zenodo, we should only need a single function defined in the base class
        from modelforge.utils.remote import download_from_url

        download_from_url(
            url=self.url,
            md5_checksum=self.gz_data_file["md5"],
            output_path=self.local_cache_dir,
            output_filename=self.gz_data_file["name"],
            length=self.gz_data_file["length"],
            force_download=self.force_download,
        )

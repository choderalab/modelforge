"""
Data class for handling QM9 data.
"""

from typing import List

from .dataset import HDF5Dataset


class tmQMDataset(HDF5Dataset):
    """
    Data class for handling tmQM dataset.

    This class provides utilities for processing and interacting with tmAM data stored in HDF5 format.

    The tmQM dataset contains the geometries and properties of 86,665 mononuclear complexes extracted from the
    Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large
    variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    Original Citation:

    David Balcells and Bastian Bjerkem Skjelstad,
    tmQM Dataset—Quantum Geometries and Properties of 86k Transition Metal Complexes
    Journal of Chemical Information and Modeling 2020 60 (12), 6135-6146
    DOI: 10.1021/acs.jcim.0c01041

    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "QM9".
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
    >>> data = tmQMDataset()
    >>> data._download()
    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(
        atomic_numbers="atomic_numbers",
        positions="geometry",
        E="total_energy",
        dipole_moment="dipole_moment_computed",
        total_charge="total_charge",
    )

    # for simplicity, commenting out those properties that are cannot be used in our current implementation
    _available_properties = [
        "geometry",
        "atomic_numbers",
        "total_charge",
        # "partial_charges",
        # "metal_center_charge",
        "dipole_moment_computed",
        "total_energy",
        "electronic_energy",
        "dispersion_energy",
        "energy_of_homo",
        "energy_of_lumo",
        "homo_lumo_gap",
        # "dipole_moment_magnitude",
        # "polarizability",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    _available_properties_association = {
        "geometry": "positions",
        "atomic_numbers": "atomic_numbers",
        "total_charge": "total_charge",
        "dipole_moment_computed": "dipole_moment",
        "total_energy": "E",
        "electronic_energy": "E",
        "dispersion_energy": "E",
        "energy_of_homo": "E",
        "energy_of_lumo": "E",
        "homo_lumo_gap": "E",
    }

    def __init__(
        self,
        dataset_name: str = "tmQM",
        version_select: str = "latest",
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache=False,
    ) -> None:
        """
        Initialize the tmQMData class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "QM9".
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
        >>> data = QM9Dataset()  # Default dataset
        >>> test_data = QM9Dataset(version_select="latest_test"))  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "total_energy",
            "dipole_moment_computed",
            "total_charge",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest

        self.dataset_name = dataset_name
        self.version_select = version_select
        from openff.units import unit

        from loguru import logger

        from importlib import resources
        from modelforge.dataset import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "tmqm.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        # make sure we have the correct yaml file
        assert data_inputs["dataset"] == "tmqm"

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
        >>> data = tmQMDataset()
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
        >>> data = tmQMDataset()
        >>> data.properties_of_interest = ["geometry", "atomic_numbers", "total_energy"]
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
        >>> data = tmQMDataset()
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

"""
Data class for handling Fe_II data.
"""

from typing import List

from .dataset import HDF5Dataset


class FeIIDataset(HDF5Dataset):
    """
    Data class for handling Fe (II) dataset.

    This class provides utilities for processing and interacting with Fe_II data stored in HDF5 format.
    This dataset contains 384 unique systems with a total of 28,834 configurations
    (note, the original publication states 383 unique systems).

    The full Fe(II) dataset includes 28834 total configurations Fe(II) organometallic complexes.
    Specifically, this includes 15568 HS geometries and 13266 LS geometries.
    These complexes originate from the Cambridge Structural Database (CSD) as curated by Nandy, et al.
    (Journal of Physical Chemistry Letters (2023), 14 (25), 10.1021/acs.jpclett.3c01214),
    and were filtered into “computation-ready” complexes, (those where both oxidation states and charges are
    already specified without hydrogen atoms missing in the structures), following the procedure outlined by
    Arunachalam, et al. (Journal of Chemical Physics (2022), 157 (18), 10.1063/5.0125700)


    Citation to the original dataset:

        Modeling Fe(II) Complexes Using Neural Networks
        Hongni Jin and Kenneth M. Merz Jr.
        Journal of Chemical Theory and Computation 2024 20 (6), 2551-2558
        DOI: 10.1021/acs.jctc.4c00063

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
    >>> data = FeIIDataset()
    >>> data._download()
    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(
        atomic_numbers="atomic_numbers",
        positions="positions",
        E="energies",
        F="forces",
        total_charge="total_charge",
        S="spin_multiplicities",
    )

    # for simplicity, commenting out those properties that are cannot be used in our current implementation
    _available_properties = [
        "positions",
        "atomic_numbers",
        "total_charge",
        "forces",
        "energies",
        "spin_multiplicities",
    ]

    _available_properties_association = {
        "positions": "positions",
        "atomic_numbers": "atomic_numbers",
        "total_charge": "total_charge",
        "energies": "E",
        "forces": "F",
        "spin_multiplicities": "S",
    }

    def __init__(
        self,
        dataset_name: str = "fe_II",
        version_select: str = "latest",
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache=False,
        element_filter: List[tuple] = None,
    ) -> None:
        """
        Initialize the tmQMData class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "Fe_II".
        version_select : str,optional
            Select the version of the dataset to use, default will provide the "latest".
            "latest_test" will select the testing subset of 1000 conformers.
             A version name can  be specified that corresponds to an entry in the associated yaml file,
             e.g., "full_dataset_v0".
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            previously downloaded files, if available; by default False.
        Examples
        --------
        >>> data = FeIIDataset()  # Default dataset
        >>> test_data = FeIIDataset(version_select="latest_test"))  # Testing subset
        """

        _default_properties_of_interest = [
            "positions",
            "atomic_numbers",
            "energies",
            "forces",
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

        yaml_file = resources.files(yaml_files) / "fe_II.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        # make sure we have the correct yaml file
        assert data_inputs["dataset"] == "fe_II"

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
            element_filter=element_filter,
        )

        # values from regression
        self._ase = {
            "H": -257.8658772400123 * unit.kilojoule_per_mole,
            "C": -897.1371901363243 * unit.kilojoule_per_mole,
            "N": -683.3438581909822 * unit.kilojoule_per_mole,
            "O": -707.3905177027947 * unit.kilojoule_per_mole,
            "P": -445.4451443983543 * unit.kilojoule_per_mole,
            "S": -367.7922055565044 * unit.kilojoule_per_mole,
            "Cl": -227.0568137730898 * unit.kilojoule_per_mole,
            "Fe": 224.48679425562852 * unit.kilojoule_per_mole,
        }

    @property
    def atomic_self_energies(self):
        from modelforge.potential.processing import AtomicSelfEnergies

        # return AtomicSelfEnergies()
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

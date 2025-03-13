"""
Data class for handling QM9 data.
"""

from typing import List

from .dataset import HDF5Dataset


class tmQMXTBDataset(HDF5Dataset):
    """
    Data class for handling tmQM-xtb dataset.

    This class provides utilities for processing and interacting with tmQM-xtb data stored in HDF5 format.
    The tmQM-xtb dataset, uses the tmQM dataset as a reference point, peforming GFN2-xTB-based MD simulations to
    provide additional configurations.

    The originalal tmQM dataset contains the geometries and properties of mononuclear complexes extracted from the
    Cambridge Structural Database, including Werner, bioinorganic, and organometallic complexes based on a large
    variety of organic ligands and 30 transition metals (the 3d, 4d, and 5d from groups 3 to 12).
    All complexes are closed-shell, with a formal charge in the range {+1, 0, −1}e

    :
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
    >>> data = tmQMXTBDataset()
    >>> data._download()
    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(
        atomic_numbers="atomic_numbers",
        positions="geometry",
        E="total_energy",
        F="forces",
        dipole_moment="dipole_moment_per_system",
        total_charge="total_charge",
    )

    # for simplicity, commenting out those properties that are cannot be used in our current implementation
    _available_properties = [
        "positions",
        "atomic_numbers",
        "total_charge",
        "forces",
        "dipole_moment_per_system",
        "energies",
        "partial_charges",
        # "spin_multiplicities",
    ]

    _available_properties_association = {
        "positions": "positions",
        "atomic_numbers": "atomic_numbers",
        "total_charge": "total_charge",
        "dipole_moment_per_system": "dipole_moment",
        "energies": "E",
        "forces": "F",
        "partial_charges": "total_charge",  # note this isn't interchangeable with partial charge but has the same units
    }

    def __init__(
        self,
        dataset_name: str = "tmQM-xtb",
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

        yaml_file = resources.files(yaml_files) / "tmqm_xtb.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        # make sure we have the correct yaml file
        assert data_inputs["dataset"] == "tmqm_xtb"

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
            "H": -1588.690123425219 * unit.kilojoule_per_mole,
            "B": -65302.33351128112 * unit.kilojoule_per_mole,
            "C": -100005.01654855655 * unit.kilojoule_per_mole,
            "N": -143654.56892638578 * unit.kilojoule_per_mole,
            "O": -197361.76171021158 * unit.kilojoule_per_mole,
            "F": -261926.20424903592 * unit.kilojoule_per_mole,
            "Si": -760035.7764038445 * unit.kilojoule_per_mole,
            "P": -896075.6280215026 * unit.kilojoule_per_mole,
            "S": -1045229.0663264447 * unit.kilojoule_per_mole,
            "Cl": -1208038.6914349555 * unit.kilojoule_per_mole,
            "Sc": -1997181.018901612 * unit.kilojoule_per_mole,
            "Ti": -2230278.4864245243 * unit.kilojoule_per_mole,
            "V": -2478389.354471244 * unit.kilojoule_per_mole,
            "Cr": -2741967.2994972193 * unit.kilojoule_per_mole,
            "Mn": -3021546.098466564 * unit.kilojoule_per_mole,
            "Fe": -3317395.5973328506 * unit.kilojoule_per_mole,
            "Co": -3629935.0938135427 * unit.kilojoule_per_mole,
            "Ni": -3959571.3270608196 * unit.kilojoule_per_mole,
            "Cu": -4306402.576897981 * unit.kilojoule_per_mole,
            "Zn": -4671113.922983311 * unit.kilojoule_per_mole,
            "As": -5869526.931994888 * unit.kilojoule_per_mole,
            "Se": -6304454.897949699 * unit.kilojoule_per_mole,
            "Br": -6757541.6132786125 * unit.kilojoule_per_mole,
            "Y": -100773.37555590154 * unit.kilojoule_per_mole,
            "Zr": -123709.71011983423 * unit.kilojoule_per_mole,
            "Nb": -149762.5718722473 * unit.kilojoule_per_mole,
            "Mo": -179149.19860244964 * unit.kilojoule_per_mole,
            "Tc": -212135.93903845942 * unit.kilojoule_per_mole,
            "Ru": -248990.05884762504 * unit.kilojoule_per_mole,
            "Rh": -290061.85478664236 * unit.kilojoule_per_mole,
            "Pd": -335541.5978772224 * unit.kilojoule_per_mole,
            "Ag": -385322.4473000328 * unit.kilojoule_per_mole,
            "Cd": -440036.922094555 * unit.kilojoule_per_mole,
            "I": -781538.4859057926 * unit.kilojoule_per_mole,
            "La": -82991.16536291114 * unit.kilojoule_per_mole,
            "Hf": -126278.27589562583 * unit.kilojoule_per_mole,
            "Ta": -149779.8577882084 * unit.kilojoule_per_mole,
            "W": -176248.83619057274 * unit.kilojoule_per_mole,
            "Re": -205660.78711122595 * unit.kilojoule_per_mole,
            "Os": -237930.364964837 * unit.kilojoule_per_mole,
            "Ir": -273916.6051998535 * unit.kilojoule_per_mole,
            "Pt": -313193.3625088413 * unit.kilojoule_per_mole,
            "Au": -356039.1883931112 * unit.kilojoule_per_mole,
            "Hg": -402551.5785347049 * unit.kilojoule_per_mole,
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

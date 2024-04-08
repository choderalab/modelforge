from typing import List

from .dataset import HDF5Dataset


class ModelDataset(HDF5Dataset):
    """
    Data class for handling the model data generated for the AlkEthOH dataset.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "ANI2x".
    for_unit_testing : bool
        If set to True, a subset of the dataset is used for unit testing purposes; by default False.
    local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
    Examples
    --------

    """

    from modelforge.utils import PropertyNames

    _property_names = PropertyNames(Z="atomic_numbers", R="geometry", E="energy")

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "energy",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "ModelDataset",
        # for_unit_testing: bool = False,
        data_combination: str = "PURE_MM",
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache: bool = False,
    ) -> None:
        """
        Initialize the ANI2xDataset class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "ANI2x".
        data_combination : str, optional
            The type of data combination to use, by default "PURE_MM"
            Options, PURE_MM_low_temp_correction, PURE_MM, PURE_ML
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            the data from the hdf5 file; by default False.
        Examples
        --------
        >>> data = ModelDataset()  # Default dataset
        >>> test_data = ModelDataset()
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "energy",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        dataset_name = f"{dataset_name}_{data_combination}"

        self.data_combination = data_combination
        from openff.units import unit

        # these come from the ANI-2x paper generated via linear fittingh of the data
        # https://github.com/isayev/ASE_ANI/blob/master/ani_models/ani-2x_8x/sae_linfit.dat
        self._ase = {
            "H": -0.5978583943827134 * unit.hartree,
            "C": -38.08933878049795 * unit.hartree,
            "N": -54.711968298621066 * unit.hartree,
            "O": -75.19106774742086 * unit.hartree,
            "S": -398.1577125334925 * unit.hartree,
            "F": -99.80348506781634 * unit.hartree,
            "Cl": -460.1681939421027 * unit.hartree,
        }
        from loguru import logger

        # We need to define the checksums for the various files that we will be dealing with to load up the data
        # There are 3 files types that need name/checksum defined, of extensions hdf5.gz, hdf5, and npz.

        # note, need to change the end of the url to dl=1 instead of dl=0 (the default when you grab the share list), to ensure the same checksum each time we download
        self.PURE_MM_url = "https://www.dropbox.com/scl/fi/0642s2ilwwmu4cyb36ttm/PURE_MM.hdf5?rlkey=n2hzkoqtchrdxtfzbybxeu2lm&dl=1"
        self.PURE_ML_url = "https://www.dropbox.com/scl/fi/lx2ets7ghw7abjbhyhvtd/PURE_ML.hdf5?rlkey=s7t0dtlab2rmt7j9bd9utx1lx&dl=1"
        self.PURE_MM_low_temp_correction_url = "https://www.dropbox.com/scl/fi/2rhc1m2ta420wgtgde9qz/MM_low_e_correction.hdf5?rlkey=bpt3t6uqo8194vl2l133nm001&dl=1"

        if self.data_combination == "PURE_MM":
            url = self.PURE_MM_url
            gz_data_file = {
                "name": "ani2x_dataset_n100.hdf5.gz",
                "md5": "093fa23aeb8f8813abd1ec08e9ff83ad",
            }
            hdf5_data_file = {
                "name": "ani2x_dataset_n100.hdf5",
                "md5": "4f54caf79e4c946dc3d6d53722d2b966",
            }
            processed_data_file = {
                "name": "ani2x_dataset_n100_processed.npz",
                "md5": "c1481fe9a6b15fb07b961d15411c0ddd",
            }

            logger.info("Using test dataset")

        else:
            url = self.full_url
            gz_data_file = {
                "name": "ani2x_dataset.hdf5.gz",
                "md5": "8daf9a7d8bbf9bcb1e9cea13b4df9270",
            }

            hdf5_data_file = {
                "name": "ani2x_dataset.hdf5",
                "md5": "86bb855cb8df54e082506088e949518e",
            }

            processed_data_file = {
                "name": "ani2x_dataset_processed.npz",
                "md5": "268438d8e1660728ba892bc7c3cd4339",
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

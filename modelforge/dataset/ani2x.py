from typing import List

from .dataset import HDF5Dataset


class ANI2xDataset(HDF5Dataset):
    """
    Data class for handling ANI2x data.

    This class provides utilities for processing the ANI2x dataset stored in the modelforge HDF5 format.

    The ANI-2x data set includes properties for small organic molecules that contain
    H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for nearly 200,000 molecules.
    This will fetch data generated with the wB97X/631Gd level of theory used in the original ANI-2x paper,
    calculated using Gaussian 09. See ani2x_curation.py for more details on the dataset curation.

    Citation: Devereux, C, Zubatyuk, R., Smith, J. et al.
                "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens."
                Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202.
                https://doi.org/10.1021/acs.jctc.0c00121

    DOI for dataset: 10.5281/zenodo.10108941

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

    _property_names = PropertyNames(
        Z="atomic_numbers", R="geometry", E="energies", F="forces"
    )

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "energies",
        "forces",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "ANI2x",
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
            Name of the dataset, by default "ANI2x".
        for_unit_testing : bool, optional
            If set to True, a subset of the dataset is used for unit testing purposes; by default False.
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.

        Examples
        --------
        >>> data = ANI2xDataset()  # Default dataset
        >>> test_data = ANI2xDataset(for_unit_testing=True)  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "energies",
            "forces",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing

        # self._ase = {
        #     "H": -1313.4668615546,
        #     "C": -99366.70745535441,
        #     "N": -143309.9379722722,
        #     "O": -197082.0671774158,
        #     "F": -261811.54555874597,
        # }
        from loguru import logger

        # We need to define the checksums for the various files that we will be dealing with to load up the data
        # There are 3 files types that need name/checksum defined, of extensions hdf5.gz, hdf5, and npz.

        # note, need to change the end of the url to dl=1 instead of dl=0 (the default when you grab the share list), to ensure the same checksum each time we download
        self.test_url = "https://www.dropbox.com/scl/fi/okv311e9yvh94owbiypcm/ani2x_dataset_n100.hdf5.gz?rlkey=pz7gnlncabtzr3b82lblr3yas&dl=1"
        self.full_url = "https://www.dropbox.com/scl/fi/egg04dmtho7l1ghqiwn1z/ani2x_dataset.hdf5.gz?rlkey=wq5qjyph5q2k0bn6vza735n19&dl=1"

        if self.for_unit_testing:
            url = self.test_url
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
        )

    @property
    def atomic_self_energies(self):
        from modelforge.potential.utils import AtomicSelfEnergies

        return AtomicSelfEnergies(element_energies=self._ase)

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

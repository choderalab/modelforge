from typing import List

from .dataset import HDF5Dataset


class ANI1xDataset(HDF5Dataset):
    """
    Data class for handling ANI1x dataset.

    This dataset includes ~5 million density function theory calculations
    for small organic molecules containing H, C, N, and O.
    A subset of ~500k are computed with accurate coupled cluster methods.

    References:

    ANI-1x dataset:
    Smith, J. S.; Nebgen, B.; Lubbers, N.; Isayev, O.; Roitberg, A. E.
    Less Is More: Sampling Chemical Space with Active Learning.
    J. Chem. Phys. 2018, 148 (24), 241733.
    https://doi.org/10.1063/1.5023802
    https://arxiv.org/abs/1801.09319

    ANI-1ccx dataset:
    Smith, J. S.; Nebgen, B. T.; Zubatyuk, R.; Lubbers, N.; Devereux, C.; Barros, K.; Tretiak, S.; Isayev, O.; Roitberg, A. E.
    Approaching Coupled Cluster Accuracy with a General-Purpose Neural Network Potential through Transfer Learning. N
    at. Commun. 2019, 10 (1), 2903.
    https://doi.org/10.1038/s41467-019-10827-4

    wB97x/def2-TZVPP data:
    Zubatyuk, R.; Smith, J. S.; Leszczynski, J.; Isayev, O.
    Accurate and Transferable Multitask Prediction of Chemical Properties with an Atoms-in-Molecules Neural Network.
    Sci. Adv. 2019, 5 (8), eaav6490.
    https://doi.org/10.1126/sciadv.aav6490


    Dataset DOI:
    https://doi.org/10.6084/m9.figshare.c.4712477.v1

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
        Z="atomic_numbers",
        R="geometry",
        E="wb97x_dz.energy",
        F="wb97x_dz.forces",
    )

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "wb97x_dz.energy",
        "wb97x_dz.forces",
        "wb97x_dz.cm5_charges",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "ANI1x",
        for_unit_testing: bool = False,
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache: bool = False,
    ) -> None:
        """
        Initialize the ANI2xDataset class.

        Parameters
        ----------
        data_name : str, optional
            Name of the dataset, by default "ANI1x".
        for_unit_testing : bool, optional
            If set to True, a subset of the dataset is used for unit testing purposes; by default False.
        local_cache_dir: str, optional
            Path to the local cache directory, by default ".".
        force_download: bool, optional
            If set to True, we will download the dataset even if it already exists; by default False.
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            the data from the hdf5 file; by default False.
        Examples
        --------
        >>> data = ANI1xDataset()  # Default dataset
        >>> test_data = ANI2xDataset(for_unit_testing=True)  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "wb97x_dz.energy",
            "wb97x_dz.forces",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing

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
        self.test_url = "https://www.dropbox.com/scl/fi/rqjc6pcv9jjzoq08hc5ao/ani1x_dataset_n100.hdf5.gz?rlkey=kgg0xvq9aac5sp3or9oh61igj&dl=1"
        self.full_url = "https://www.dropbox.com/scl/fi/d98h9kt4pl40qeapqzu00/ani1x_dataset.hdf5.gz?rlkey=7q1o8hh9qzbxehsobjurcksit&dl=1"

        if self.for_unit_testing:
            url = self.test_url
            gz_data_file = {
                "name": "ani1x_dataset_n100.hdf5.gz",
                "md5": "51e2491e3c5b7b5a432e2012892cfcbb",
                "length": 85445473,
            }
            hdf5_data_file = {
                "name": "ani1x_dataset_n100.hdf5",
                "md5": "f3c934b79f035ecc3addf88c027f5e46",
            }
            processed_data_file = {
                "name": "ani1x_dataset_n100_processed.npz",
                "md5": None,
            }

            logger.info("Using test dataset")

        else:
            url = self.full_url
            gz_data_file = {
                "name": "ani1x_dataset.hdf5.gz",
                "md5": "408cdcf9768ac96a8ae8ade9f078c51b",
                "length": 4510287721,
            }

            hdf5_data_file = {
                "name": "ani1x_dataset.hdf5",
                "md5": "361b7c4b9a4dfeece70f0fe6a893e76a",
            }

            processed_data_file = {"name": "ani1x_dataset_processed.npz", "md5": None}

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
            length=self.gz_data_file["length"],
            force_download=self.force_download,
        )

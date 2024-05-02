from typing import List

from .dataset import HDF5Dataset


class SPICE2Dataset(HDF5Dataset):
    """
    Data class for handling SPICE 2 dataset.

    The SPICE dataset contains conformations for a diverse set of small molecules,
    dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
    uncharged molecules, and a wide range of covalent and non-covalent interactions.
    It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
    using Psi4 along with other useful quantities such as multipole moments and bond orders.

    This includes the following collections from qcarchive. Collections included in SPICE 1.1.4 are annotated with
    along with the version used in SPICE 1.1.4; while the underlying molecules are typically the same in a given collection,
    newer versions may have had some calculations redone, e.g., rerun calculations that failed or rerun with
    a newer version Psi4

      - 'SPICE Solvated Amino Acids Single Points Dataset v1.1'     * (SPICE 1.1.4 at v1.1)
      - 'SPICE Dipeptides Single Points Dataset v1.3'               * (SPICE 1.1.4 at v1.2)
      - 'SPICE DES Monomers Single Points Dataset v1.1'             * (SPICE 1.1.4 at v1.1)
      - 'SPICE DES370K Single Points Dataset v1.0'                  * (SPICE 1.1.4 at v1.0)
      - 'SPICE DES370K Single Points Dataset Supplement v1.1'       * (SPICE 1.1.4 at v1.0)
      - 'SPICE PubChem Set 1 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 2 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 3 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 4 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 5 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 6 Single Points Dataset v1.3'            * (SPICE 1.1.4 at v1.2)
      - 'SPICE PubChem Set 7 Single Points Dataset v1.0'
      - 'SPICE PubChem Set 8 Single Points Dataset v1.0'
      - 'SPICE PubChem Set 9 Single Points Dataset v1.0'
      - 'SPICE PubChem Set 10 Single Points Dataset v1.0'
      - 'SPICE Ion Pairs Single Points Dataset v1.2'                * (SPICE 1.1.4 at v1.1)
      - 'SPICE PubChem Boron Silicon v1.0'
      - 'SPICE Solvated PubChem Set 1 v1.0'
      - 'SPICE Water Clusters v1.0'
      - 'SPICE Amino Acid Ligand v1.0


    SPICE 2 zenodo release:
    https://zenodo.org/records/10835749

    Reference to original SPICE publication:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6


    Attributes
    ----------
    dataset_name : str
        Name of the dataset, default is "SPICE2".
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
        E="dft_total_energy",
        F="dft_total_force",
        Q="mbis_charges",
    )

    _available_properties = [
        "geometry",
        "atomic_numbers",
        "dft_total_energy",
        "dft_total_force",
        "mbis_charges",
        "formation_energy",
        "scf_dipole",
        "total_charge",
        "reference_energy",
    ]  # All properties within the datafile, aside from SMILES/inchi.

    def __init__(
        self,
        dataset_name: str = "SPICE2",
        for_unit_testing: bool = False,
        local_cache_dir: str = ".",
        force_download: bool = False,
        regenerate_cache: bool = False,
    ) -> None:
        """
        Initialize the SPICE2Dataset class.

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
        regenerate_cache: bool, optional
            If set to True, we will regenerate the npz cache file even if it already exists, using
            the data from the hdf5 file; by default False.
        Examples
        --------
        >>> data = SPICE2Dataset()  # Default dataset
        >>> test_data = SPICE2Dataset(for_unit_testing=True)  # Testing subset
        """

        _default_properties_of_interest = [
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "mbis_charges",
        ]  # NOTE: Default values

        self._properties_of_interest = _default_properties_of_interest
        if for_unit_testing:
            dataset_name = f"{dataset_name}_subset"

        self.dataset_name = dataset_name
        self.for_unit_testing = for_unit_testing

        from openff.units import unit

        # SPICE provides reference values that depend upon charge, as charged molecules are included in the dataset.
        # The reference_energy (i.e., sum of the value of isolated atoms with appropriate charge considerations)
        # are included in the dataset, along with the formation_energy, which is the difference between
        # the dft_total_energy and the reference_energy.

        # To be able to use the dataset for training in a consistent way with the ANI datasets, we will only consider
        # the ase values for the uncharged isolated atoms, if available. Ions will use the values Ca 2+, K 1+, Li 1+, Mg 2+, Na 1+.
        # See spice_2_from_qcarchive_curation.py for more details.

        # We will need to address this further later to see how we best want to handle this; the ASE are just meant to bring everything
        # roughly to the same scale, and values do not vary substantially by charge state.

        # Reference energies, in hartrees, computed with Psi4 1.5 wB97M-D3BJ/def2-TZVPPD.

        self._ase = {
            "B": -24.671520535482145 * unit.hartree,
            "Br": -2574.1167240829964 * unit.hartree,
            "C": -37.87264507233593 * unit.hartree,
            "Ca": -676.9528465198214 * unit.hartree,  # 2+
            "Cl": -460.1988762285739 * unit.hartree,
            "F": -99.78611622985483 * unit.hartree,
            "H": -0.498760510048753 * unit.hartree,
            "I": -297.76228914445625 * unit.hartree,
            "K": -599.8025677513111 * unit.hartree,  # 1+
            "Li": -7.285254714046546 * unit.hartree,  # 1+
            "Mg": -199.2688420040449 * unit.hartree,  # 2+
            "N": -54.62327513368922 * unit.hartree,
            "Na": -162.11366478783253 * unit.hartree,  # 1+
            "O": -75.11317840410095 * unit.hartree,
            "P": -341.3059197024934 * unit.hartree,
            "S": -398.1599636677874 * unit.hartree,
            "Si": -289.4131352299586 * unit.hartree,
        }
        from loguru import logger

        # We need to define the checksums for the various files that we will be dealing with to load up the data
        # There are 3 files types that need name/checksum defined, of extensions hdf5.gz, hdf5, and npz.

        # note, need to change the end of the url to dl=1 instead of dl=0 (the default when you grab the share list), to ensure the same checksum each time we download
        self.test_url = "https://www.dropbox.com/scl/fi/1jawffjrh17r796g76udi/spice_2_dataset_ntc_1000.hdf5.gz?rlkey=r0crabvyg7xdgapv2qk3hk6t9&st=0ro9na0c&dl=1"
        self.full_url = "https://www.dropbox.com/scl/fi/udoc3jj7wa7du8jgqiat0/spice_2_dataset.hdf5.gz?rlkey=csgwqa237m002n54jnld5pfgy&dl=1"

        if self.for_unit_testing:
            url = self.test_url
            gz_data_file = {
                "name": "SPICE2_dataset_nc_1000.hdf5.gz",
                "md5": "04063f08a7ec93abfc661c22b12ceeb0",
                "length": 26751220,  # the number of bytes to be able to display the download progress bar correctly
            }
            hdf5_data_file = {
                "name": "SPICE2_dataset_nc_1000.hdf5",
                "md5": "0a2554d0dba4f289dd93670686e4842e",
            }
            # npz file checksums may vary with different versions of python/numpy
            processed_data_file = {
                "name": "SPICE2_dataset_nc_1000_processed.npz",
                "md5": None,
            }

            logger.info("Using test dataset")

        else:
            url = self.full_url
            gz_data_file = {
                "name": "SPICE2_dataset.hdf5.gz",
                "md5": "244a559a6062bbec5c9cb49af036ff7d",
                "length": 26313472231,
            }

            hdf5_data_file = {
                "name": "SPICE2_dataset.hdf5",
                "md5": "9659a0f18050b9e7b122c0046b705480",
            }

            processed_data_file = {
                "name": "SPICE2_dataset_processed.npz",
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

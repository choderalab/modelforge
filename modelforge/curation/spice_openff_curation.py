from modelforge.utils.units import *

from modelforge.curation.curation_baseclass import *
from retry import retry
from tqdm import tqdm


class SPICE_pubchem_1_2_openff_curation(dataset_curation):
    """
    Routines to fetch and process the spice 1.1.4 dataset into a curated hdf5 file.

    Small-molecule/Protein Interaction Chemical Energies (SPICE).
    The SPICE dataset contains 1.1 million conformations for a diverse set of small molecules,
    dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
    uncharged molecules, and a wide range of covalent and non-covalent interactions.
    It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
    using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.

    Reference:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

    Dataset DOI:
    https://doi.org/10.5281/zenodo.8222043

    Parameters
    ----------
    hdf5_file_name, str, required
        name of the hdf5 file generated for the SPICE dataset
    output_file_dir: str, optional, default='./'
        Path to write the output hdf5 files.
    local_cache_dir: str, optional, default='./spice_dataset'
        Location to save downloaded dataset.
    convert_units: bool, optional, default=True
        Convert from [angstrom, hartree] (i.e., source units)
        to [nanometer, kJ/mol]

    Examples
    --------
    >>> spice_openff_data = SPICE_pubchem_1_2_openff_curation(hdf5_file_name='spice_pubchem_12_openff_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/spice12_openff_dataset')
    >>> spice_openff_data.process()

    """

    def _init_dataset_parameters(self):
        self.qcarchive_server = "ml.qcarchive.molssi.org"

        self.qm_parameters = {
            "conformations": {
                "u_in": unit.bohr,
                "u_out": unit.nanometer,
            },
            "formation_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dft_total_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dft_total_gradient": {
                "u_in": unit.hartree / unit.bohr,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "mbis_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "mbis_dipoles": {
                "u_in": unit.elementary_charge * unit.bohr,
                "u_out": unit.elementary_charge * unit.nanometer,
            },
            "mbis_quadrupoles": {
                "u_in": unit.elementary_charge * unit.bohr**2,
                "u_out": unit.elementary_charge * unit.nanometer**2,
            },
            "mbis_octupoles": {
                "u_in": unit.elementary_charge * unit.bohr**3,
                "u_out": unit.elementary_charge * unit.nanometer**3,
            },
            "scf_dipole": {
                "u_in": unit.elementary_charge * unit.bohr,
                "u_out": unit.elementary_charge * unit.nanometer,
            },
            "scf_quadrupole": {
                "u_in": unit.elementary_charge * unit.bohr**2,
                "u_out": unit.elementary_charge * unit.nanometer**2,
            },
            "mayer_indices": {
                "u_in": None,
                "u_out": None,
            },
            "wiberg_lowdin_indices": {
                "u_in": None,
                "u_out": None,
            },
        }

    def _init_record_entries_series(self):
        # For data efficiency, information for different conformers will be grouped together
        # To make it clear to the dataset loader which pieces of information are common to all
        # conformers, or which pieces encode the series, we will label each value.
        # The keys in this dictionary correspond to the label of the entries in each record.
        # The value indicates if the entry contains series data (True) or a single common entry (False).
        # If the entry has a value of True, the "series" attribute in hdf5 file will be set to True; False, if False.
        # This information will be used by the code to read in the datafile to know how to parse underlying records.

        self._record_entries_series = {
            "name": False,
            "atomic_numbers": False,
            "n_configs": False,
            "smiles": False,
            "subset": False,
            "geometry": True,
            "dft_total_energy": True,
            "dft_total_gradient": True,
            "formation_energy": True,
            "mayer_indices": True,
            "mbis_charges": True,
            "mbis_dipoles": True,
            "mbis_octupoles": True,
            "mbis_quadrupoles": True,
            "scf_dipole": True,
            "scf_quadrupole": True,
            "wiberg_lowdin_indices": True,
        }

    @retry(delay=1, jitter=1, backoff=2, tries=50, logger=logger, max_delay=10)
    def _fetch_singlepoint_from_qcarchive(
        self,
        dataset_type: str,
        dataset_name: str,
        specification_name: str,
        local_database_name: str,
        local_path_dir: str,
        pbar: tqdm,
        force_download: bool,
        unit_testing_max_records: Optional[int] = None,
    ):
        from sqlitedict import SqliteDict
        from loguru import logger

        ds = self.client.get_dataset(
            dataset_type=dataset_type, dataset_name=dataset_name
        )

        entry_names = ds.entry_names
        with SqliteDict(
            f"{local_path_dir}/{local_database_name}",
            tablename=specification_name,
            autocommit=True,
        ) as spice_db:
            # defining the db_keys as a set is faster for
            # searching to see if a key exists
            db_keys = set(spice_db.keys())
            to_fetch = []
            if force_download:
                for name in entry_names[0:unit_testing_max_records]:
                    to_fetch.append(name)
            else:
                for name in entry_names[0:unit_testing_max_records]:
                    if name not in db_keys:
                        to_fetch.append(name)
            pbar.total = pbar.total + len(to_fetch)
            pbar.refresh()

            # We need a different routine to fetch entries vs records with a give specification
            if len(to_fetch) > 0:
                if specification_name == "entry":
                    logger.debug(
                        f"Fetching {len(to_fetch)} entries from dataset {dataset_name}."
                    )
                    for entry in ds.iterate_entries(
                        to_fetch, force_refetch=force_download
                    ):
                        spice_db[entry.dict()["name"]] = entry
                        pbar.update(1)

                else:
                    logger.debug(
                        f"Fetching {len(to_fetch)} records for specification {specification_name} from dataset {dataset_name}."
                    )
                    for record in ds.iterate_records(
                        to_fetch,
                        specification_names=[specification_name],
                        force_refetch=force_download,
                    ):
                        spice_db[record[0]] = record[2]
                        pbar.update(1)

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        unit_testing_max_records: Optional[int] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        name: str, required
            Name of the raw hdf5 file,
        unit_testing_max_records: int, optional, default=None
            If set to an integer ('n') the routine will only process the first 'n' records; useful for unit tests.

        Examples
        --------
        """
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"

        # From documentation: By default, objects inside group are iterated in alphanumeric order.
        # However, if group is created with track_order=True, the insertion order for the group is remembered (tracked)
        # in HDF5 file, and group contents are iterated in that order.
        # As such, we shouldn't need to do sort the objects to ensure reproducibility.
        # self.data = sorted(self.data, key=lambda x: x["name"])

    def process(
        self,
        force_download: bool = False,
        unit_testing_max_records: Optional[int] = None,
        n_threads=6,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        unit_testing_max_records: int, optional, default=None
            If set to an integer, 'n', the routine will only process the first 'n' records, useful for unit tests.

        Examples
        --------
        >>> spice_openff_data = SPICE_pubchem_1_2_openff_curation(hdf5_file_name='spice_pubchem_12_openff_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/spice12_openff_dataset')
        >>> spice_openff_data.process()

        """
        from qcportal import PortalClient
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dataset_type = "singlepoint"
        dataset_names = [
            "SPICE PubChem Set 1 Single Points Dataset v1.2",
            "SPICE PubChem Set 2 Single Points Dataset v1.2",
            "SPICE PubChem Set 3 Single Points Dataset v1.2",
            "SPICE PubChem Set 4 Single Points Dataset v1.2",
            "SPICE PubChem Set 5 Single Points Dataset v1.2",
            "SPICE PubChem Set 6 Single Points Dataset v1.2",
        ]
        specification_names = ["spec_2", "spec_6", "entry"]
        self.client = PortalClient(self.qcarchive_server)

        # for dataset_name in dataset_names:
        #     for specification_name in specification_names:
        #         self._fetch_singlepoint_from_qcarchive(
        #             dataset_type=dataset_type,
        #             dataset_name=dataset_name,
        #             specification_name=specification_name,
        #             local_database_name=local_database_name,
        #             local_path_dir=self.local_cache_dir,
        #             unit_testing_max_records=unit_testing_max_records,
        #         )
        threads = []
        completed = 0
        local_database_names = []

        with tqdm() as pbar:
            pbar.total = 0
            with ThreadPoolExecutor(max_workers=n_threads) as e:
                for i, dataset_name in enumerate(dataset_names):
                    local_database_name = f"spice_pubchem{i+1}_12.sqlite"
                    local_database_names.append(local_database_name)
                    for specification_name in specification_names:
                        threads.append(
                            e.submit(
                                self._fetch_singlepoint_from_qcarchive,
                                dataset_type=dataset_type,
                                dataset_name=dataset_name,
                                specification_name=specification_name,
                                local_database_name=local_database_name,
                                local_path_dir=self.local_cache_dir,
                                pbar=pbar,
                                force_download=force_download,
                                unit_testing_max_records=unit_testing_max_records,
                            )
                        )

        # self._clear_data()

        # process the rest of the dataset
        # self._process_downloaded(
        #    self.local_cache_dir, local_database_name, unit_testing_max_records
        # )

        # self._generate_hdf5()

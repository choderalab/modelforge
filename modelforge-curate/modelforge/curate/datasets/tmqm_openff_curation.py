from typing import List, Tuple, Dict, Optional

from modelforge.curate import Record, SourceDataset
from modelforge.curate.properties import (
    AtomicNumbers,
    TotalCharge,
    Energies,
    Positions,
    Forces,
    DipoleMomentPerSystem,
    SpinMultiplicitiesPerSystem,
    SpinMultiplicitiesPerAtom,
    QuadrupoleMomentPerSystem,
    MetaData,
    PartialCharges,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration
from modelforge.utils.io import import_, check_import

# retry = import_("retry").retry
from tqdm import tqdm
from openff.units import unit
from loguru import logger
import numpy as np


class tmQMOpenFFCuration(DatasetCuration):
    """
    Fetches the OpenFF tmQM dataset from the MOLSSI QCArchive and processes it into a local dataset.

    The tmQM openff dataset provides   The tmQM-xtb dataset performed MD-based sampling
    of the structures in the tmQM dataset using GFN2-XTB.  The tmQM openff dataset is a subset of the tmQM-xtb dataset
    focused on systems with


    Parameters
    ----------
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.

    Examples
    --------
    >>> tmqm_openff_data = tmQMOpenFFCuration(local_cache_dir='~/datasets/tmqm_openff_dataset')
    >>> tmqm_openff_data.process()

    """

    def _init_dataset_parameters(self):
        self.qcarchive_server = "https://api.qcarchive.molssi.org"

        self.molecule_names = {}

    # we will use the retry package to allow us to resume download if we lose connection to the server
    # @retry(delay=1, jitter=1, backoff=2, tries=50, logger=logger, max_delay=10)
    def _fetch_singlepoint_from_qcarchive(
        self,
        dataset_id: str,
        qcportal_view_filename: str,
        qcportal_view_path: str,
        specification_name: str,
        local_database_name: str,
        local_path_dir: str,
        force_download: bool,
        pbar: Optional[tqdm] = None,
    ):
        """
        Fetches a singlepoint dataset from the MOLSSI QCArchive and stores it in a local sqlite database.

        Parameters
        ----------
        dataset_id: str, required
            qcarchive dataset ID to fetch from the QCArchive
        qcportal_view_filename: str, required
            Name of the qcportal view to fetch from the QCArchive
        qcportal_view_path: str, required
            Path to the qcportal view to fetch from the QCArchive
        specification_name: str, required
            Name of the specification to fetch from the QCArchive
        local_database_name: str, required
            Name of the local sqlite database to store the dataset
        local_path_dir: str, required
            Path to the directory to store the local sqlite database
        force_download: bool, required
            If True, this will force the software to download the data again, even if present.
        pbar: Optional[tqdm], optional, default=None
            Progress bar to track the download process.

        pbar

        Returns
        -------

        """

        SqliteDict = import_("sqlitedict").SqliteDict
        # from sqlitedict import SqliteDict
        from loguru import logger

        PortalClient = import_("qcportal").PortalClient
        # from qcportal import PortalClient

        client = PortalClient(
            self.qcarchive_server, cache_dir=f"{local_path_dir}/qcarchive_cache1"
        )

        ds = client.get_dataset_by_id(dataset_id)
        qcportal_view_full_path = f"{qcportal_view_path}/{qcportal_view_filename}"
        logger.debug(f"Using qcportal view file: {qcportal_view_full_path}")
        ds.use_view_cache(qcportal_view_full_path)

        # ds = client.get_dataset(dataset_type=dataset_type, dataset_name=dataset_name)

        ds.fetch_entry_names()
        entry_names = ds.entry_names

        with SqliteDict(
            f"{local_path_dir}/{local_database_name}",
            tablename=specification_name,
            autocommit=True,
        ) as ph_db:
            # defining the db_keys as a set is faster for
            # searching to see if a key exists
            db_keys = set(ph_db.keys())
            to_fetch = []
            if force_download:
                for name in entry_names:
                    to_fetch.append(name)
            else:
                for name in entry_names:
                    if name not in db_keys:
                        to_fetch.append(name)
            if pbar is not None:
                pbar.total = pbar.total + len(to_fetch)
                pbar.refresh()
            logger.debug(f"Fetching {len(to_fetch)} entries from dataset {dataset_id}.")

            # We need a different routine to fetch entries vs records with a give specification
            if len(to_fetch) > 0:
                if specification_name == "entry":
                    logger.debug(
                        f"Fetching {len(to_fetch)} entries from dataset {dataset_id}."
                    )
                    for entry in ds.iterate_entries(
                        to_fetch, force_refetch=force_download
                    ):
                        ph_db[entry.dict()["name"]] = entry
                        if pbar is not None:
                            pbar.update(1)

                else:
                    logger.debug(
                        f"Fetching {len(to_fetch)} records for {specification_name} from dataset {dataset_id}."
                    )

                    for record in ds.iterate_records(
                        to_fetch,
                        specification_names=[specification_name],
                        force_refetch=force_download,
                    ):
                        # ph_db[record[0]] = [record[2].dict(), record[2].trajectory]

                        ph_db[record[0]] = record[2].dict()
                        if pbar is not None:
                            pbar.update(1)

    from functools import lru_cache

    def _process_downloaded(
        self,
        local_path_dir: str,
        filenames: List[str],
        dataset_ids: List[str],
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        filenames: List[str], required
            Names of the raw sqlite files to process,
        dataset_names: List[str], required
            List of names of the sqlite datasets to process.

        """
        from tqdm import tqdm
        import numpy as np

        SqliteDict = import_("sqlitedict").SqliteDict
        # from sqlitedict import SqliteDict
        from loguru import logger

        from modelforge.dataset.utils import _ATOMIC_ELEMENT_TO_NUMBER

        from numpy import newaxis

        dataset = SourceDataset(
            name=self.dataset_name,
            append_property=True,
            local_db_dir=self.local_cache_dir,
        )

        for filename, dataset_id in zip(filenames, dataset_ids):
            input_file_name = f"{local_path_dir}/{filename}"

            non_error_keys = []
            print("Reading from local database: ", input_file_name)
            # identify the set of molecules that do not have errors
            with SqliteDict(
                input_file_name, tablename="BP86/def2-TZVP", autocommit=False
            ) as db_default:
                db_keys = list(db_default.keys())

                for key in db_keys:
                    if db_default[key]["status"].value == "complete":
                        non_error_keys.append(key)
                print(len(non_error_keys))

                with SqliteDict(
                    input_file_name, tablename="entry", autocommit=False
                ) as ph_db:
                    logger.debug(f"Processing {filename}.")

                    # we will loop over the keys in the database
                    for key in tqdm(non_error_keys):

                        # format is "{molecule_name}-{conformer_id}-m{spin_multiplicity}"
                        mol_name = key.split("-")[0]

                        entry_val = ph_db[key].dict()
                        record_val = db_default[key]

                        ################################
                        # properties that apply to all records
                        # these properties only need to be added once
                        ################################
                        if not mol_name in dataset.records.keys():

                            atomic_numbers = []
                            for element in entry_val["molecule"]["symbols"]:
                                atomic_numbers.append(
                                    _ATOMIC_ELEMENT_TO_NUMBER[element]
                                )

                            atomic_numbers = AtomicNumbers(
                                value=np.array(atomic_numbers).reshape(-1, 1)
                            )
                            dataset.add_property(mol_name, atomic_numbers)

                            molecular_formula = MetaData(
                                name="molecular_formula",
                                value=entry_val["molecule"]["identifiers"][
                                    "molecular_formula"
                                ],
                            )
                            dataset.add_property(mol_name, molecular_formula)

                            canonical_isomeric_explicit_hydrogen_mapped_smiles = MetaData(
                                name="canonical_isomeric_explicit_hydrogen_mapped_smiles",
                                value=entry_val["molecule"]["extras"][
                                    "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                                ],
                            )
                            dataset.add_property(
                                mol_name,
                                canonical_isomeric_explicit_hydrogen_mapped_smiles,
                            )
                        ##############################
                        # per configuration properties
                        ##############################
                        # molecular_hash is unique for each conformer
                        molecule_hash = MetaData(
                            name="molecule_hash",
                            value=entry_val["molecule"]["identifiers"]["molecule_hash"],
                        )
                        dataset.add_property(mol_name, molecule_hash)

                        # add the identifier from qcarchive
                        identifier = MetaData(
                            name="id", value=entry_val["molecule"]["identifiers"]["id"]
                        )
                        dataset.add_property(mol_name, identifier)

                        # add the total charge
                        total_charge = TotalCharge(
                            name="total_charge",
                            value=np.array(
                                entry_val["molecule"]["molecular_charge"]
                            ).reshape(1, 1),
                            units=unit.elementary_charge,
                        )
                        dataset.add_property(mol_name, total_charge)

                        # add the multiplicity
                        system_multiplicity = SpinMultiplicitiesPerSystem(
                            name="per_system_spin_multiplicity",
                            value=np.array(
                                entry_val["molecule"]["molecular_multiplicity"]
                            ).reshape(1, 1),
                        )
                        dataset.add_property(mol_name, system_multiplicity)

                        # add the positions
                        positions = Positions(
                            name="positions",
                            value=np.array(entry_val["molecule"]["geometry"]).reshape(
                                1, -1, 3
                            ),
                            units=unit.bohr,
                        )
                        dataset.add_property(mol_name, positions)

                        # add the energies
                        energies = Energies(
                            name="dft_total_energy",
                            value=np.array(
                                record_val["properties"]["dft total energy"]
                            ).reshape(1, 1),
                            units=unit.hartree,
                        )
                        dataset.add_property(mol_name, energies)

                        # add the forces
                        forces = Forces(
                            name="dft_total_force",
                            value=-np.array(
                                record_val["properties"]["dft total gradient"]
                            ).reshape(1, -1, 3),
                            units=unit.hartree / unit.bohr,
                        )
                        dataset.add_property(mol_name, forces)

                        # per system dipole moment
                        dipole_moment = DipoleMomentPerSystem(
                            name="scf_dipole",
                            value=np.array(
                                record_val["properties"]["scf dipole moment"]
                            ).reshape(1, 3),
                            units=unit.elementary_charge * unit.bohr,
                        )
                        dataset.add_property(mol_name, dipole_moment)

                        # per system quadrupole moment
                        quadrupole_moment = QuadrupoleMomentPerSystem(
                            name="scf_quadrupole",
                            value=np.array(
                                record_val["properties"]["scf quadrupole moment"]
                            ).reshape(1, 3, 3),
                            units=unit.elementary_charge * unit.bohr**2,
                        )
                        dataset.add_property(mol_name, quadrupole_moment)

                        # add the mulliken partial charges
                        mulliken_charges = PartialCharges(
                            name="mulliken_partial_charges",
                            value=np.array(
                                record_val["properties"]["mulliken charges"]
                            ).reshape(1, -1),
                            units=unit.elementary_charge,
                        )
                        dataset.add_property(mol_name, mulliken_charges)

                        # add the lowdin charges
                        lowdin_charges = PartialCharges(
                            name="lowdin_partial_charges",
                            value=np.array(
                                record_val["properties"]["lowdin charges"]
                            ).reshape(1, -1),
                            units=unit.elementary_charge,
                        )
                        dataset.add_property(mol_name, lowdin_charges)

                        # extract the lowdin spin multiplicity
                        n_atoms = forces.n_atoms
                        spins = self._process_log_for_spin(
                            log=record_val["compute_history_"][0]["outputs_"]["stdout"][
                                "data_"
                            ],
                            n_atoms=n_atoms,
                        )
                        spin_multiplicity_per_atom = SpinMultiplicitiesPerAtom(
                            name="spin_multiplicity_per_atom",
                            value=spins,
                            units=unit.dimensionless,
                        )
                        dataset.add_property(mol_name, spin_multiplicity_per_atom)

        return dataset

    def _process_log_for_spin(self, log, n_atoms: int) -> np.array:
        """
        Processes the log file to extract the lowdin spin multiplicity.

        Parameters
        ----------
        log: bytes

        Returns
        -------
        np.array
            The spin multiplicity extracted from the log.
        """
        import zstandard

        # need to first decompress the log file
        dctx = zstandard.ZstdDecompressor()
        out = dctx.decompress(log)
        # decode the bytes to a string
        out2 = out.decode("utf-8", errors="ignore")
        # next split the string into lines and then find the line that contains the spin multiplicity title
        out3 = out2.split("\n")

        lowdin_start = 0
        for i, line in enumerate(out3):
            if "Lowdin Charges" in line:
                lowdin_start = i
        lowdin_data = out3[lowdin_start + 2 : lowdin_start + 2 + n_atoms]

        # now we need to extract the spin multiplicity from the lowdin data
        spins = []
        for line in lowdin_data:
            spins.append(line.split()[4])

        return np.array(spins, dtype=np.float64).reshape(1, n_atoms, 1)

    def process(
        self,
        qcportal_view_filename: str,
        qcportal_view_path: str,
        force_download: bool = False,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        qcportal_view_filename: str, required
            Name of the qcportal view file to use
        qcportal_view_path: str, required
            Path to the qcportal view file to use
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        n_threads, int, default=2
            Number of concurrent threads for retrieving data from QCArchive
        Examples
        --------
        >>> tmqm_openff_data = tmQMOpenFFCuration(local_cache_dir='~/datasets/phalkethoh_openff_dataset')
        >>> tmqm_openff_data.process()

        """
        # if max_records is not None and total_conformers is not None:
        #     raise ValueError(
        #         "max_records and total_conformers cannot be set at the same time."
        #     )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from importlib import resources
        from modelforge.curate.datasets import yaml_files
        import os

        import yaml

        # we need to expand the file path to the qcportal view file
        qcportal_view_path = os.path.expanduser(qcportal_view_path)
        yaml_file = resources.files(yaml_files) / "tmqm_openff_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "tmqm_openff"
        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Using latest version {self.version_select}.")

        dataset_ids = data_inputs[self.version_select]["collection_names"]

        specification_names = ["entry", "BP86/def2-TZVP"]

        threads = []
        local_database_names = []

        with tqdm() as pbar:
            pbar.total = 0
            for i, dataset_id in enumerate(dataset_ids):
                local_database_name = f"{dataset_id}.sqlite"
                local_database_names.append(local_database_name)
                for specification_name in specification_names:
                    self._fetch_singlepoint_from_qcarchive(
                        dataset_id=dataset_id,
                        qcportal_view_filename=qcportal_view_filename,
                        qcportal_view_path=qcportal_view_path,
                        specification_name=specification_name,
                        local_database_name=local_database_name,
                        local_path_dir=self.local_cache_dir,
                        force_download=force_download,
                        pbar=pbar,
                    )
        logger.debug(f"Data fetched.")
        logger.debug(f"Processing downloaded dataset.")

        self.dataset = self._process_downloaded(
            self.local_cache_dir,
            local_database_names,
            dataset_ids,
        )

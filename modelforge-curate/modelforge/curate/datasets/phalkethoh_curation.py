from typing import List, Tuple, Dict, Optional

from modelforge.curate import Record, SourceDataset
from modelforge.curate.properties import (
    AtomicNumbers,
    TotalCharge,
    Energies,
    Positions,
    Forces,
    DipoleMomentPerSystem,
    MetaData,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration
from modelforge.utils.io import import_, check_import

retry = import_("retry").retry
from tqdm import tqdm
from openff.units import unit
from loguru import logger


class PhAlkEthOHCuration(DatasetCuration):
    """
    Fetches the OpenFF PhAlkEthOH dataset from MOLSSI QCArchive and processes it into a curated hdf5 file.

    All QM datapoints retrieved were generated using B3LYP-D3BJ/DZVP level of theory.
    This is the default theory used for force field development by the Open Force Field Initiative.

    This includes the following collection from qcarchive:

    "OpenFF Sandbox CHO PhAlkEthOH v1.0"

    Link to associated github repository:
    https://github.com/openforcefield/qca-dataset-submission/tree/master/submissions/2020-09-18-OpenFF-Sandbox-CHO-PhAlkEthOH


    Parameters
    ----------
    hdf5_file_name, str, required
        name of the hdf5 file generated for the SPICE dataset
    output_file_dir: str, optional, default='./'
        Path to write the output hdf5 files.
    local_cache_dir: str, optional, default='./spice_dataset'
        Location to save downloaded dataset.

    Examples
    --------
    >>> PhAlkEthOH_openff_data = PhAlkEthOHOpenFFCuration(hdf5_file_name='PhAlkEthOH_openff_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/PhAlkEthOH_openff_dataset')
    >>> PhAlkEthOH_openff_data.process()

    """

    def _init_dataset_parameters(self):
        self.qcarchive_server = "https://api.qcarchive.molssi.org"

        self.molecule_names = {}

    # we will use the retry package to allow us to resume download if we lose connection to the server
    # @retry(delay=1, jitter=1, backoff=2, tries=50, logger=logger, max_delay=10)
    def _fetch_singlepoint_from_qcarchive(
        self,
        dataset_name: str,
        specification_name: str,
        local_database_name: str,
        local_path_dir: str,
        force_download: bool,
        max_records: Optional[int] = None,
        pbar: Optional[tqdm] = None,
    ):
        """
        Fetches a singlepoint dataset from the MOLSSI QCArchive and stores it in a local sqlite database.

        Parameters
        ----------
        dataset_name: str, required
            Name of the dataset to fetch from the QCArchive
        specification_name: str, required
            Name of the specification to fetch from the QCArchive
        local_database_name: str, required
            Name of the local sqlite database to store the dataset
        local_path_dir: str, required
            Path to the directory to store the local sqlite database
        force_download: bool, required
            If True, this will force the software to download the data again, even if present.
        max_records: Optional[int], optional, default=None
            If set to an integer, 'n', the routine will only process the first 'n' records, useful for unit tests.
            Note, conformers of the same molecule are saved in separate records, and thus the number of molecules
            that end up in the 'data' list after _process_downloaded is called  may be less than unit_testing_max_records.
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

        dataset_type = "optimization"
        client = PortalClient(
            self.qcarchive_server, cache_dir=f"{local_path_dir}/qcarchive_cache1"
        )

        ds = client.get_dataset(dataset_type=dataset_type, dataset_name=dataset_name)

        ds.fetch_entry_names()
        entry_names = ds.entry_names

        if max_records is None:
            max_records = len(entry_names)

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
                for name in entry_names[0:max_records]:
                    to_fetch.append(name)
            else:
                for name in entry_names[0:max_records]:
                    if name not in db_keys:
                        to_fetch.append(name)
            if pbar is not None:
                pbar.total = pbar.total + len(to_fetch)
                pbar.refresh()
            logger.debug(
                f"Fetching {len(to_fetch)} entries from dataset {dataset_name}."
            )

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
                        if pbar is not None:
                            pbar.update(1)

                else:
                    logger.debug(
                        f"Fetching {len(to_fetch)} records for {specification_name} from dataset {dataset_name}."
                    )

                    for record in ds.iterate_records(
                        to_fetch,
                        specification_names=[specification_name],
                        force_refetch=force_download,
                        include=["**"],
                    ):
                        # spice_db[record[0]] = [record[2].dict(), record[2].trajectory]

                        spice_db[record[0]] = [
                            record[2].dict()["status"].value,
                            [
                                [traj.dict(), traj.molecule.geometry]
                                for traj in record[2].trajectory
                            ],
                        ]
                        if pbar is not None:
                            pbar.update(1)

    def _calculate_total_charge(
        self, smiles: str
    ) -> Tuple[unit.Quantity, unit.Quantity]:
        """
        Calculate the  total charge as defined by the SMILES string.

        T
        Parameters
        ----------
        smiles: str, required
            SMILES string describing the molecule of interest.

        Returns
        -------
        unit.Quantity
            total charge of the molecule (in elementary charge).
        """

        Chem = import_("rdkit.Chem")
        # from rdkit import Chem

        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
        total_charge = sum(atom.GetFormalCharge() for atom in rdmol.GetAtoms())

        return int(total_charge) * unit.elementary_charge

    def _process_downloaded(
        self,
        local_path_dir: str,
        filenames: List[str],
        dataset_names: List[str],
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        max_force: Optional[unit.Quantity] = None,
        final_conformer_only: Optional[bool] = None,
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
        max_conformers_per_record: Optional[int], optional, default=None
            If set, this will limit the number of conformers per record to the specified number.
        total_conformers: Optional[int], optional, default=None
            If set, this will limit the total number of conformers to the specified number.
        max_force: Optional[float], optional, default=None
            If set, this will exclude any conformers with a force that exceeds this value.
        final_conformer_only: Optional[bool], optional, default=None
            If set to True, only the final conformer of each record will be processed. This should be the final
            energy minimized conformer.
        """
        from tqdm import tqdm
        import numpy as np

        SqliteDict = import_("sqlitedict").SqliteDict
        # from sqlitedict import SqliteDict
        from loguru import logger

        from modelforge.dataset.utils import _ATOMIC_ELEMENT_TO_NUMBER

        from numpy import newaxis

        dataset = SourceDataset("PhAlkEthOH_openff", append_property=True)

        for filename, dataset_name in zip(filenames, dataset_names):
            input_file_name = f"{local_path_dir}/{filename}"

            non_error_keys = []
            print("Reading from local database: ", input_file_name)
            # identify the set of molecules that do not have errors
            with SqliteDict(
                input_file_name, tablename="default", autocommit=False
            ) as db_default:
                db_keys = list(db_default.keys())

                for key in db_keys:
                    if db_default[key][0] == "complete":
                        non_error_keys.append(key)
            print(len(non_error_keys))
            # sorted_keys, original_name = self._sort_keys(non_error_keys)

            # first read in molecules from entry
            with SqliteDict(
                input_file_name, tablename="entry", autocommit=False
            ) as spice_db:
                logger.debug(f"Processing {filename} entries.")
                for key in tqdm(non_error_keys):

                    val = spice_db[key].dict()
                    # name = key.split("-")[0]
                    # I've encountered a few instances where the record name is not sufficiently unique
                    # (saturated vs unsaturated ring); appending the chemical formula should make it unique
                    name = f'{key[: key.rfind("-")]}_{val["initial_molecule"]["name"]}'

                    # these properties only need to be added once
                    # so we need to check if
                    if not name in dataset.records.keys():
                        source = MetaData(
                            name="source", value=input_file_name.replace(".sqlite", "")
                        )
                        dataset.add_property(name, source)

                        atomic_numbers = []
                        for element in val["initial_molecule"]["symbols"]:
                            atomic_numbers.append(_ATOMIC_ELEMENT_TO_NUMBER[element])

                        atomic_numbers = AtomicNumbers(
                            value=np.array(atomic_numbers).reshape(-1, 1)
                        )
                        dataset.add_property(name, atomic_numbers)

                        molecular_formula = MetaData(
                            name="molecular_formula",
                            value=val["initial_molecule"]["identifiers"][
                                "molecular_formula"
                            ],
                        )
                        dataset.add_property(name, molecular_formula)

                        canonical_isomeric_explicit_hydrogen_mapped_smiles = MetaData(
                            name="canonical_isomeric_explicit_hydrogen_mapped_smiles",
                            value=val["initial_molecule"]["extras"][
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                        )
                        dataset.add_property(
                            name, canonical_isomeric_explicit_hydrogen_mapped_smiles
                        )

            with SqliteDict(
                input_file_name, tablename="default", autocommit=False
            ) as spice_db:
                logger.debug(f"Processing {filename} default spec.")

                for key in tqdm(non_error_keys):
                    # name = key.split("-")[0]
                    trajectory = spice_db[key][1]

                    if final_conformer_only:
                        trajectory = [trajectory[-1]]
                    for state in trajectory:
                        add_record = True
                        properties, config = state
                        name = (
                            f'{key[: key.rfind("-")]}_{properties["molecule_"]["name"]}'
                        )
                        smiles = (
                            dataset.records[name]
                            .meta_data[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ]
                            .value
                        )

                        total_charge_temp = self._calculate_total_charge(smiles)

                        # if set, let us see if the configuration has a force that exceeds the maximum
                        if max_force is not None:
                            force_magnitude = (
                                np.abs(
                                    properties["properties"]["current gradient"]
                                    + properties["properties"][
                                        "dispersion correction gradient"
                                    ]
                                )
                                * unit.hartree
                                / unit.bohr
                            )
                            if np.any(force_magnitude > max_force):
                                add_record = False

                        if add_record:
                            total_charge = TotalCharge(
                                value=np.array(total_charge_temp.m).reshape(1, 1),
                                units=total_charge_temp.u,
                            )
                            dataset.add_property(name, total_charge)

                            positions = Positions(
                                value=config.reshape(1, -1, 3), units=unit.bohr
                            )
                            dataset.add_property(name, positions)

                            # Note need to typecast here because of a bug in the
                            # qcarchive entry: see issue: https://github.com/MolSSI/QCFractal/issues/766
                            dispersion_correction_energy = Energies(
                                name="dispersion_correction_energy",
                                value=np.array(
                                    float(
                                        properties["properties"][
                                            "dispersion correction energy"
                                        ]
                                    )
                                ).reshape(1, 1),
                                units=unit.hartree,
                            )
                            dataset.add_property(name, dispersion_correction_energy)

                            dft_total_energy = Energies(
                                name="dft_total_energy",
                                value=np.array(
                                    properties["properties"]["current energy"]
                                ).reshape(1, 1)
                                + dispersion_correction_energy.value,
                                units=unit.hartree,
                            )
                            dataset.add_property(name, dft_total_energy)

                            dispersion_correction_gradient = Forces(
                                name="dispersion_correction_gradient",
                                value=np.array(
                                    properties["properties"][
                                        "dispersion correction gradient"
                                    ]
                                ).reshape(1, -1, 3),
                                units=unit.hartree / unit.bohr,
                            )
                            dataset.add_property(name, dispersion_correction_gradient)

                            dispersion_correction_force = Forces(
                                name="dispersion_correction_force",
                                value=-dispersion_correction_gradient.value,
                                units=unit.hartree / unit.bohr,
                            )
                            dataset.add_property(name, dispersion_correction_force)

                            dft_total_gradient = Forces(
                                name="dft_total_gradient",
                                value=np.array(
                                    properties["properties"]["current gradient"]
                                ).reshape(1, -1, 3)
                                + dispersion_correction_gradient.value,
                                units=unit.hartree / unit.bohr,
                            )
                            dataset.add_property(name, dft_total_gradient)

                            dft_total_force = Forces(
                                name="dft_total_force",
                                value=-dft_total_gradient.value,
                                units=unit.hartree / unit.bohr,
                            )
                            dataset.add_property(name, dft_total_force)

                            scf_dipole = DipoleMomentPerSystem(
                                name="scf_dipole",
                                value=np.array(
                                    properties["properties"]["scf dipole"]
                                ).reshape(1, 3),
                                units=unit.elementary_charge * unit.bohr,
                            )
                            dataset.add_property(name, scf_dipole)
        dataset.validate_records()
        if total_conformers is not None or max_conformers_per_record is not None:
            conformers_count = 0
            dataset_restricted = SourceDataset("PhAlkEthOH_openff")
            for key in dataset.records.keys():
                if total_conformers is not None:
                    if conformers_count >= total_conformers:
                        break
                n_conformers = dataset.records[key].n_configs

                if max_conformers_per_record is not None:
                    n_conformers = min(n_conformers, max_conformers_per_record)

                if total_conformers is not None:
                    n_conformers = min(
                        n_conformers, total_conformers - conformers_count
                    )
                record_temp = dataset.slice_record(key, 0, n_conformers)
                dataset_restricted.add_record(record_temp)

                conformers_count += n_conformers
            # if any of the records have no configurations, remove them
            for key in dataset_restricted.records.keys():
                if dataset_restricted.records[key].n_configs == 0:
                    dataset_restricted.remove_record(key)
            return dataset_restricted
        # if any of the records have no configurations, remove them

        for key in dataset.records.keys():
            if dataset.records[key].n_configs == 0:
                dataset.remove_record(key)

        return dataset

    def process(
        self,
        force_download: bool = False,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        max_force: Optional[unit.Quantity] = None,
        final_conformer_only=None,
        n_threads=2,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        max_records: int, optional, default=None
            If set to an integer, 'n_r', the routine will only process the first 'n_r' records, useful for unit tests.
            Can be used in conjunction with max_conformers_per_record and total_conformers.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records and total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with max_records and max_conformers_per_record.
        max_force: Optional[float], optional, default=None
            If set this any confirugrations with a force that exceeds this value will be excluded.
        final_conformer_only: Optional[bool], optional, default=None
            If set to True, only the final conformer of each record will be processed.
        n_threads, int, default=2
            Number of concurrent threads for retrieving data from QCArchive
        Examples
        --------
        >>> phalkethoh_openff_data = PhAlkEthOHCuration(hdf5_file_name='phalkethoh_openff_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/phalkethoh_openff_dataset')
        >>> phalkethoh_openff_data.process()

        """
        # if max_records is not None and total_conformers is not None:
        #     raise ValueError(
        #         "max_records and total_conformers cannot be set at the same time."
        #     )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from importlib import resources
        from modelforge.curate.datasets import yaml_files

        import yaml

        yaml_file = resources.files(yaml_files) / "PhAlkEthOH_openff_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "PhAlkEthOHopenff"
        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Using latest version {self.version_select}.")

        dataset_names = data_inputs[self.version_select]["collection_names"]

        specification_names = ["entry", "default"]

        threads = []
        local_database_names = []

        with tqdm() as pbar:
            pbar.total = 0
            with ThreadPoolExecutor(max_workers=n_threads) as e:
                for i, dataset_name in enumerate(dataset_names):
                    local_database_name = f"{dataset_name}.sqlite"
                    local_database_names.append(local_database_name)
                    for specification_name in specification_names:
                        threads.append(
                            e.submit(
                                self._fetch_singlepoint_from_qcarchive,
                                dataset_name=dataset_name,
                                specification_name=specification_name,
                                local_database_name=local_database_name,
                                local_path_dir=self.local_cache_dir,
                                force_download=force_download,
                                max_records=max_records,
                                pbar=pbar,
                            )
                        )
        logger.debug(f"Data fetched.")
        logger.debug(f"Processing downloaded dataset.")

        self.dataset = self._process_downloaded(
            self.local_cache_dir,
            local_database_names,
            dataset_names,
            max_conformers_per_record=max_conformers_per_record,
            total_conformers=total_conformers,
            max_force=max_force,
            final_conformer_only=final_conformer_only,
        )

        logger.info(f"writing file {self.hdf5_file_name} to {self.output_file_dir}")
        self.write_hdf5_and_json_files(
            file_name=self.hdf5_file_name, file_path=self.output_file_dir
        )

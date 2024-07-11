from typing import List, Tuple, Dict, Optional

from modelforge.curation.curation_baseclass import *
from modelforge.utils.io import import_, check_import

check_import("retry")
from retry import retry
from tqdm import tqdm
from openff.units import unit


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
    convert_units: bool, optional, default=True
        Convert from the source units (e.g., angstrom, bohr, hartree)
        to [nanometer, kJ/mol] (i.e., target units)

    Examples
    --------
    >>> PhAlkEthOH_openff_data = PhAlkEthOHOpenFFCuration(hdf5_file_name='PhAlkEthOH_openff_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/PhAlkEthOH_openff_dataset')
    >>> PhAlkEthOH_openff_data.process()

    """

    def _init_dataset_parameters(self):
        self.qcarchive_server = "https://api.qcarchive.molssi.org"

        self.molecule_names = {}

        # dictionary of properties and their input units (i.e., those from QCArchive)
        # and desired output units; unit conversion is performed if convert_units = True
        self.qm_parameters = {
            "geometry": {
                "u_in": unit.bohr,
                "u_out": unit.nanometer,
            },
            "dft_total_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dft_total_gradient": {
                "u_in": unit.hartree / unit.bohr,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "dft_total_force": {
                "u_in": unit.hartree / unit.bohr,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "dispersion_correction_energy": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "dispersion_correction_gradient": {
                "u_in": unit.hartree / unit.bohr,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "scf_dipole": {
                "u_in": unit.elementary_charge * unit.bohr,
                "u_out": unit.elementary_charge * unit.nanometer,
            },
            "total_charge": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
        }

    def _init_record_entries_series(self):
        """
        Init the dictionary that defines the format of the data.

        For data efficiency, information for different conformers will be grouped together
        To make it clear to the dataset loader which pieces of information are common to all
        conformers or which quantities are series (i.e., have different values for each conformer).
        These labels will also allow us to define whether a given entry is per-atom, per-molecule,
        or is a scalar/string that applies to the entire record.
        Options include:
        single_rec, e.g., name, n_configs, smiles
        single_atom, e.g., atomic_numbers (these are the same for all conformers)
        single_mol, e.g., reference energy
        series_atom, e.g., charges
        series_mol, e.g., dft energy, dipole moment, etc.
        These ultimately appear under the "format" attribute in the hdf5 file.

        Examples
        >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
                      ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
        """

        self._record_entries_series = {
            "name": "single_rec",
            "dataset_name": "single_rec",
            "source": "single_rec",
            "total_charge": "single_rec",
            "atomic_numbers": "single_atom",
            "n_configs": "single_rec",
            "molecular_formula": "single_rec",
            "canonical_isomeric_explicit_hydrogen_mapped_smiles": "single_rec",
            "geometry": "series_atom",
            "dft_total_energy": "series_mol",
            "dft_total_gradient": "series_atom",
            "dft_total_force": "series_atom",
            "scf_dipole": "series_mol",
        }

    # we will use the retry package to allow us to resume download if we lose connection to the server
    @retry(delay=1, jitter=1, backoff=2, tries=50, logger=logger, max_delay=10)
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

        SqliteDict = import_("sqlitedict.SqliteDict")
        # from sqlitedict import SqliteDict
        from loguru import logger

        PortalClient = import_("qcportal.PortalClient")
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

        return (int(total_charge) * unit.elementary_charge,)

    def _process_downloaded(
        self,
        local_path_dir: str,
        filenames: List[str],
        dataset_names: List[str],
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        atomic_numbers_to_limit: Optional[List[int]] = None,
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
        atomic_numbers_to_limit: Optional[List[int]], optional, default=None
            If set, this will limit the dataset to only include molecules with atomic numbers in the list.
        """
        from tqdm import tqdm
        import numpy as np

        SqliteDict = import_("sqlitedict.SqliteDict")
        # from sqlitedict import SqliteDict
        from loguru import logger

        qcel = import_("qcelemental")
        from numpy import newaxis

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
                    name = key
                    # if we haven't processed a molecule with this name yet
                    # we will add to the molecule_names dictionary
                    if name not in self.molecule_names.keys():
                        self.molecule_names[name] = len(self.data)

                        data_temp = {}
                        data_temp["name"] = name
                        data_temp["source"] = input_file_name.replace(".sqlite", "")
                        atomic_numbers = []
                        for element in val["initial_molecule"]["symbols"]:
                            atomic_numbers.append(
                                qcel.periodictable.to_atomic_number(element)
                            )
                        data_temp["atomic_numbers"] = np.array(atomic_numbers).reshape(
                            -1, 1
                        )
                        data_temp["molecular_formula"] = val["initial_molecule"][
                            "identifiers"
                        ]["molecular_formula"]
                        data_temp[
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ] = val["initial_molecule"]["extras"][
                            "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                        ]
                        data_temp["n_configs"] = 0

                        (data_temp["total_charge"],) = self._calculate_total_charge(
                            data_temp[
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ]
                        )
                        data_temp["dataset_name"] = dataset_name
                        self.data.append(data_temp)

            with SqliteDict(
                input_file_name, tablename="default", autocommit=False
            ) as spice_db:
                logger.debug(f"Processing {filename} default spec.")

                for key in tqdm(non_error_keys):
                    # name = key.split("-")[0]
                    trajectory = spice_db[key][1]
                    name = key
                    index = self.molecule_names[name]

                    for state in trajectory:
                        properties, config = state
                        self.data[index]["n_configs"] += 1

                        # note, we will use the convention of names being lowercase
                        # and spaces denoted by underscore
                        quantity = "geometry"
                        quantity_o = "geometry"
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = config.reshape(1, -1, 3)
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    config.reshape(1, -1, 3),
                                )
                            )

                        # note, we will use the convention of names being lowercase
                        # and spaces denoted by underscore
                        quantity = "current energy"
                        quantity_o = "dft_total_energy"
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = properties["properties"][
                                quantity
                            ]
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    properties["properties"][quantity],
                                )
                            )

                        quantity = "dispersion correction energy"
                        quantity_o = "dispersion_correction_energy"
                        # Note need to typecast here because of a bug in the
                        # qcarchive entry: see issue: https://github.com/MolSSI/QCFractal/issues/766
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = np.array(
                                float(properties["properties"][quantity])
                            ).reshape(1, 1)
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    np.array(
                                        float(properties["properties"][quantity])
                                    ).reshape(1, 1),
                                ),
                            )

                        quantity = "current gradient"
                        quantity_o = "dft_total_gradient"
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = np.array(
                                properties["properties"][quantity]
                            ).reshape(1, -1, 3)
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    np.array(
                                        properties["properties"][quantity]
                                    ).reshape(1, -1, 3),
                                )
                            )

                        quantity = "dispersion correction gradient"
                        quantity_o = "dispersion_correction_gradient"
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = np.array(
                                properties["properties"][quantity]
                            ).reshape(1, -1, 3)
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    np.array(
                                        properties["properties"][quantity]
                                    ).reshape(1, -1, 3),
                                )
                            )

                        quantity = "scf dipole"
                        quantity_o = "scf_dipole"
                        if quantity_o not in self.data[index].keys():
                            self.data[index][quantity_o] = np.array(
                                properties["properties"][quantity]
                            ).reshape(1, 3)
                        else:
                            self.data[index][quantity_o] = np.vstack(
                                (
                                    self.data[index][quantity_o],
                                    np.array(
                                        properties["properties"][quantity]
                                    ).reshape(1, 3),
                                )
                            )

        # assign units
        for datapoint in self.data:
            for key in datapoint.keys():
                if key in self.qm_parameters:
                    if not isinstance(datapoint[key], unit.Quantity):
                        datapoint[key] = (
                            datapoint[key] * self.qm_parameters[key]["u_in"]
                        )
            # add in the formation energy defined as:
            # dft_total_energy + dispersion_correction_energy - reference_energy

            # the dispersion corrected energy and gradient can be calculated from the raw data
            datapoint["dft_total_energy"] = (
                datapoint["dft_total_energy"]
                + datapoint["dispersion_correction_energy"]
            )
            # we only want to write the dispersion corrected energy to the file to avoid confusion
            datapoint.pop("dispersion_correction_energy")

            datapoint["dft_total_gradient"] = (
                datapoint["dft_total_gradient"]
                + datapoint["dispersion_correction_gradient"]
            )
            datapoint["dft_total_force"] = -datapoint["dft_total_gradient"]
            # we only want to write the dispersion corrected gradient to the file to avoid confusion
            datapoint.pop("dispersion_correction_gradient")

            # datapoint["formation_energy"] = (
            #     datapoint["dft_total_energy"]
            #     - np.array(datapoint["reference_energy"].m * datapoint["n_configs"])
            #     * datapoint["reference_energy"].u
            # )

        if self.convert_units:
            self._convert_units()

        # if we want to limit to a subset of elements, we can quickly post process the dataset
        if atomic_numbers_to_limit is not None:
            data_temp = []
            for datapoint in self.data:
                add_to_record = set(datapoint["atomic_numbers"].flatten()).issubset(
                    atomic_numbers_to_limit
                )
                if add_to_record:
                    data_temp.append(datapoint)
            self.data = data_temp

        if total_conformers is not None or max_conformers_per_record is not None:
            conformers_count = 0
            temp_data = []
            for datapoint in self.data:
                if total_conformers is not None:
                    if conformers_count >= total_conformers:
                        break
                n_conformers = datapoint["n_configs"]
                if max_conformers_per_record is not None:
                    n_conformers = min(n_conformers, max_conformers_per_record)

                if total_conformers is not None:
                    n_conformers = min(
                        n_conformers, total_conformers - conformers_count
                    )

                datapoint["n_configs"] = n_conformers
                datapoint["geometry"] = datapoint["geometry"][0:n_conformers]
                datapoint["dft_total_energy"] = datapoint["dft_total_energy"][
                    0:n_conformers
                ]
                datapoint["dft_total_gradient"] = datapoint["dft_total_gradient"][
                    0:n_conformers
                ]
                datapoint["dft_total_force"] = datapoint["dft_total_force"][
                    0:n_conformers
                ]
                datapoint["scf_dipole"] = datapoint["scf_dipole"][0:n_conformers]

                temp_data.append(datapoint)
                conformers_count += n_conformers
            self.data = temp_data

    def process(
        self,
        force_download: bool = False,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        limit_atomic_species: Optional[list] = None,
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
            Note defining this will only fetch from the "SPICE PubChem Set 1 Single Points Dataset v1.2"
        limit_atomic_species: Optional[list] = None,
            If set to a list of element symbols, records that contain any elements not in this list will be ignored.
        n_threads, int, default=6
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
        from modelforge.curation import yaml_files
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
        self._clear_data()
        self.molecule_names.clear()
        logger.debug(f"Processing downloaded dataset.")

        if limit_atomic_species is not None:
            self.atomic_numbers_to_limit = []
            from openff.units import elements

            for symbol in limit_atomic_species:
                for num, sym in elements.SYMBOLS.items():
                    if sym == symbol:
                        self.atomic_numbers_to_limit.append(num)
        else:
            self.atomic_numbers_to_limit = None

        self._process_downloaded(
            self.local_cache_dir,
            local_database_names,
            dataset_names,
            max_conformers_per_record=max_conformers_per_record,
            total_conformers=total_conformers,
            atomic_numbers_to_limit=self.atomic_numbers_to_limit,
        )

        self._generate_hdf5()

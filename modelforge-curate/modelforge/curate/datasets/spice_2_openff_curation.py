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
    PartialCharges,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration
from modelforge.utils.io import import_, check_import

retry = import_("retry").retry
from tqdm import tqdm
from openff.units import unit
from loguru import logger


class SPICE2OpenFFCuration(DatasetCuration):
    """
      Class for handling SPICE 2.0.1 dataset at the OpenForceField level of theory.

      All QM datapoints retrieved were generated using B3LYP-D3BJ/DZVP level of theory.
      This is the default theory used for force field development by the Open Force Field Initiative.

    Collections that are common to both spice 1 and 2.  Note, in these cases, the original level of theory (generally spec_3) and spice
    level of theory (spec_2 and spec_6) are included in the same collection.

      - 'SPICE Solvated Amino Acids Single Points Dataset v1.1'
      - 'SPICE Dipeptides Single Points Dataset v1.2'
      - 'SPICE DES Monomers Single Points Dataset v1.1'
      - 'SPICE DES370K Single Points Dataset v1.0'
      - 'SPICE DES370K Single Points Dataset Supplement v1.1'
      - 'SPICE PubChem Set 1 Single Points Dataset v1.2'
      - 'SPICE PubChem Set 2 Single Points Dataset v1.2'
      - 'SPICE PubChem Set 3 Single Points Dataset v1.2'
      - 'SPICE PubChem Set 4 Single Points Dataset v1.2'
      - 'SPICE PubChem Set 5 Single Points Dataset v1.2'
      - 'SPICE PubChem Set 6 Single Points Dataset v1.2'

    The following are only part of spice 2.  Note, for clarity, these collections do not include the data at the
    original level of theory for SPICE, but only the data at the openff level of theory; these have "OpenFF" in the name
    to indicate this:
      - 'SPICE PubChem Set 7 Single Points Dataset OpenFF v1.0'
      - 'SPICE PubChem Set 8 Single Points Dataset OpenFF v1.0'
      - 'SPICE PubChem Set 9 Single Points Dataset OpenFF v1.0'
      - 'SPICE PubChem Set 10 Single Points Dataset OpenFF v1.0'
      - 'SPICE Water Clusters OpenFF v1.1'
      - 'SPICE Amino Acid Ligand OpenFF v1.0'
      - 'SPICE Solvated PubChem Set 1 OpenFF v1.0'
      - 'SPICE PubChem Boron Silicon OpenFF v1.0'

      It does not include the following datasets that are part of the official 1.1.4 release of SPICE (calculated
      at the ωB97M-D3(BJ)/def2-TZVPPD level of theory), as the openff level of theory was not used for these datasets:

      "SPICE Ion Pairs Single Points Dataset v1.1",
      "SPICE DES370K Single Points Dataset Supplement v1.0",

    Reference to SPICE 2 publication:
        Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E.
        Nutmeg and SPICE: models and data for biomolecular machine learning.
        Journal of chemical theory and computation, 20(19), 8583-8593 (2024).
        https://doi.org/10.1021/acs.jctc.4c00794

    Reference to original SPICE publication:
      Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
      A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
      Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

    Parameters
    ----------
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.

    Examples
    --------

    """

    def _init_dataset_parameters(self):
        self.qcarchive_server = "https://ml.qcarchive.molssi.org"

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

        dataset_type = "singlepoint"
        client = PortalClient(
            self.qcarchive_server, cache_dir=f"{local_path_dir}/qcarchive_cache"
        )

        ds = client.get_dataset(dataset_type=dataset_type, dataset_name=dataset_name)

        ds.fetch_entry_names()
        entry_names = ds.entry_names
        logger.debug(" Fetching entry names from dataset.")
        logger.debug(f"Found {len(entry_names)} entries in dataset {dataset_name}.")

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
                for name in entry_names:
                    to_fetch.append(name)
            else:
                for name in entry_names:
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
                    ):
                        # ph_db[record[0]] = [record[2].dict(), record[2].trajectory]

                        spice_db[record[0]] = record[2].dict()
                        if pbar is not None:
                            pbar.update(1)

    from functools import lru_cache

    @lru_cache(maxsize=None)
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

    def _sort_keys(self, non_error_keys: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        This will sort record identifiers such that conformers are listed in numerical order.

        This will, if necessarily also sanitize the names of the molecule, to ensure that we have the following
        form {name}-{conformer_number}. In some cases, the original name has a "-" which would causes issues
        with simply splitting based upon a "-" to either get the name or the conformer number.

        The function is called by _process_downloaded.

        Parameters
        ----------
        non_error_keys

        Returns
        -------

        """
        # we need to sanitize the names of the molecule, as
        # some of the names have a dash in them, for example ALA-ALA-1
        # This will replace any hyphens in the name with an underscore.
        # To be able to retain the original name, needed for accessing the record in the sqlite file
        # we will create a simple dictionary that maps the sanitized name to the original name.

        non_error_keys_sanitized = []
        original_name = {}

        for key in non_error_keys:
            s = "_"
            d = "-"
            temp = key.split("-")
            name = d.join([s.join(temp[0:-1]), temp[-1]])
            non_error_keys_sanitized.append(name)
            original_name[name] = key
            print(name, key)

        # We will sort the keys such that conformers are listed in numerical order.
        # This is not strictly necessary, but will help to better retain
        # connection to the original QCArchive data, where in most cases conformer-id will directly correspond to
        # the index of the final array constructed here.
        # Note, if the calculation of an individual conformer failed on qcarchive,
        # it will have been excluded from the non_error_keys list. As such, in such cases,
        # the conformer id in the record name will no longer have a one-to-one correspondence with the
        # index of the conformers in the combined arrays.  This should not be an issue in terms of training,
        # but could cause some confusion when interrogating a specific set of conformers geometries for a molecule.

        sorted_keys = []

        # names of the molecules are of form  {name}-{conformer_number}
        # first sort by name
        s = "_"
        pre_sort = sorted(non_error_keys_sanitized, key=lambda x: (x.split("-")[0]))

        # then sort each molecule by conformer_number
        # we'll do this by simple iteration through the list, and when we encounter a new molecule name, we'll sort the
        # previous temporary list we generated.
        current_val = pre_sort[0].split("-")[0]
        temp_list = []

        for val in pre_sort:
            name = val.split("-")[0]

            if name == current_val:
                temp_list.append(val)
            else:
                sorted_keys += sorted(temp_list, key=lambda x: int(x.split("-")[-1]))
                temp_list = []
                current_val = name
                temp_list.append(val)

        # sort the final batch
        sorted_keys += sorted(temp_list, key=lambda x: int(x.split("-")[-1]))

        return sorted_keys, original_name

    def _process_downloaded(
        self,
        local_path_dir: str,
        filenames: List[str],
        dataset_names: List[str],
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

        for filename, dataset_name in zip(filenames, dataset_names):
            input_file_name = f"{local_path_dir}/{filename}"

            non_error_keys = []
            print("Reading from local database: ", input_file_name)
            # identify the set of molecules that do not have errors

            # identify the set of molecules that do not have errors
            with SqliteDict(
                input_file_name, tablename="spec_2", autocommit=False
            ) as spice_db_spec2:
                spec2_keys = list(spice_db_spec2.keys())

                with SqliteDict(
                    input_file_name, tablename="spec_6", autocommit=False
                ) as spice_db_spec6:
                    for key in spec2_keys:
                        if (
                            spice_db_spec2[key]["status"].value == "complete"
                            and spice_db_spec6[key]["status"].value == "complete"
                        ):
                            non_error_keys.append(key)

            sorted_keys, original_name = self._sort_keys(non_error_keys)
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
                    # I've encountered a few instances in PhAlkEthOH where the record name is not sufficiently unique
                    # (saturated vs unsaturated ring); appending the chemical formula should make it unique
                    # will do this here just to make sure no issues in the spice dataset
                    name = f'{key[: key.rfind("-")]}_{val["molecule"]["name"]}'

                    # these properties only need to be added once
                    # so we need to check if
                    if not name in dataset.records.keys():
                        dataset.create_record(name)
                        source = MetaData(
                            name="source", value=input_file_name.replace(".sqlite", "")
                        )
                        dataset.add_property(name, source)

                        atomic_numbers = []
                        for element in val["molecule"]["symbols"]:
                            atomic_numbers.append(_ATOMIC_ELEMENT_TO_NUMBER[element])

                        atomic_numbers = AtomicNumbers(
                            value=np.array(atomic_numbers).reshape(-1, 1)
                        )
                        dataset.add_property(name, atomic_numbers)

                        molecular_formula = MetaData(
                            name="molecular_formula",
                            value=val["molecule"]["identifiers"]["molecular_formula"],
                        )
                        dataset.add_property(name, molecular_formula)

                        canonical_isomeric_explicit_hydrogen_mapped_smiles = MetaData(
                            name="canonical_isomeric_explicit_hydrogen_mapped_smiles",
                            value=val["molecule"]["extras"][
                                "canonical_isomeric_explicit_hydrogen_mapped_smiles"
                            ],
                        )
                        dataset.add_property(
                            name, canonical_isomeric_explicit_hydrogen_mapped_smiles
                        )

                    positions = Positions(
                        value=val["molecule"]["geometry"].reshape(1, -1, 3),
                        units=unit.bohr,
                    )
                    dataset.add_property(name, positions)

                    total_charge_temp = self._calculate_total_charge(
                        canonical_isomeric_explicit_hydrogen_mapped_smiles.value
                    )

                    total_charge = TotalCharge(
                        value=np.array(total_charge_temp.m).reshape(1, 1),
                        units=total_charge_temp.u,
                    )
                    dataset.add_property(name, total_charge)

            with SqliteDict(
                input_file_name, tablename="spec_2", autocommit=False
            ) as spice_db:
                logger.debug(f"Processing {filename} spec_2.")

                for key in tqdm(non_error_keys):

                    with SqliteDict(
                        input_file_name, tablename="entry", autocommit=False
                    ) as spice_db_entry:
                        val = spice_db_entry[key].dict()

                        # I've encountered a few instances in PhAlkEthOH where the record name is not sufficiently unique
                        # (saturated vs unsaturated ring); appending the chemical formula should make it unique
                    name = f'{key[: key.rfind("-")]}_{val["molecule"]["name"]}'

                    # record = dataset.get_record(name)
                    val = spice_db[key]
                    dft_total_energy = Energies(
                        name="dft_energy",
                        value=np.array(val["properties"]["dft total energy"]).reshape(
                            1, 1
                        ),
                        units=unit.hartree,
                    )
                    dataset.add_property(name, dft_total_energy)

                    dft_total_force = Forces(
                        name="dft_force",
                        value=-np.array(
                            val["properties"]["dft total gradient"]
                        ).reshape(1, -1, 3),
                        units=unit.hartree / unit.bohr,
                    )
                    dataset.add_property(name, dft_total_force)

                    mbis_charges = PartialCharges(
                        name="mbis_charges",
                        value=np.array(val["properties"]["mbis charges"]).reshape(
                            1, -1, 1
                        ),
                        units=unit.elementary_charge,
                    )
                    dataset.add_property(name, mbis_charges)

                    scf_dipole = DipoleMomentPerSystem(
                        name="scf_dipole",
                        value=np.array(val["properties"]["scf dipole"]).reshape(1, 3),
                        units=unit.elementary_charge * unit.bohr,
                    )
                    dataset.add_property(name, scf_dipole)

            # process spec 6 which gives us the dispersion correction energy/gradient
            with SqliteDict(
                input_file_name, tablename="spec_6", autocommit=False
            ) as spice_db:
                logger.debug(f"Processing {filename} spec_6.")

                for key in tqdm(non_error_keys):

                    with SqliteDict(
                        input_file_name, tablename="entry", autocommit=False
                    ) as spice_db_entry:
                        val = spice_db_entry[key].dict()

                        # I've encountered a few instances in PhAlkEthOH where the record name is not sufficiently unique
                        # (saturated vs unsaturated ring); appending the chemical formula should make it unique
                    name = f'{key[: key.rfind("-")]}_{val["molecule"]["name"]}'
                    val = spice_db[key]

                    dispersion_correction_energy = Energies(
                        name="dispersion_correction_energy",
                        value=np.array(
                            float(val["properties"]["dispersion correction energy"])
                        ).reshape(1, 1),
                        units=unit.hartree,
                    )

                    dataset.add_property(name, dispersion_correction_energy)

                    dispersion_correction_force = Forces(
                        name="dispersion_correction_force",
                        value=-np.array(
                            val["properties"]["dispersion correction gradient"]
                        ).reshape(1, -1, 3),
                        units=unit.hartree / unit.bohr,
                    )
                    dataset.add_property(name, dispersion_correction_force)

            # we need to add the dispersion correction energy to the dft energy
            # and also the dispersion correction force to the dft force

            for name in dataset.keys():
                record = dataset.get_record(name)

                # first process energy
                # fetch the properties from the dataset
                dft_energy = record.get_property("dft_energy")
                dispersion_correction_energy = record.get_property(
                    "dispersion_correction_energy"
                )

                # ensure we have the same units otherwise this won't work so well!
                assert dft_energy.units == dispersion_correction_energy.units

                # add a new property that is the sum of the two
                dft_total_energy = Energies(
                    name="dft_total_energy",
                    value=dft_energy.value + dispersion_correction_energy.value,
                    units=dft_energy.units,
                )
                dataset.add_property(name, dft_total_energy)

                # now process forces
                dft_force = record.get_property("dft_force")
                dispersion_correction_force = record.get_property(
                    "dispersion_correction_force"
                )

                assert dft_force.units == dispersion_correction_force.units

                dft_total_force = Forces(
                    name="dft_total_force",
                    value=dft_force.value + dispersion_correction_force.value,
                    units=dft_force.units,
                )
                dataset.add_property(name, dft_total_force)

        return dataset

    def process(
        self,
        force_download: bool = False,
        n_threads=2,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.
        n_threads, int, default=2
            Number of concurrent threads for retrieving data from QCArchive
        Examples
        --------


        """
        # if max_records is not None and total_conformers is not None:
        #     raise ValueError(
        #         "max_records and total_conformers cannot be set at the same time."
        #     )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from importlib import resources
        from modelforge.curate.datasets import yaml_files

        import yaml

        yaml_file = resources.files(yaml_files) / "spice2_openff_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "spice2openff"
        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Using latest version {self.version_select}.")

        dataset_names = data_inputs[self.version_select]["collection_names"]
        logger.debug(f"Using dataset names {dataset_names}.")
        specification_names = ["entry", "spec_2", "spec_6"]

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
                                pbar=pbar,
                            )
                        )
        # with tqdm() as pbar:
        #     pbar.total = 0
        #     for i, dataset_name in enumerate(dataset_names):
        #         local_database_name = f"{dataset_name}.sqlite"
        #         local_database_names.append(local_database_name)
        #         for specification_name in specification_names:
        #             self._fetch_singlepoint_from_qcarchive(
        #                 dataset_name=dataset_name,
        #                 specification_name=specification_name,
        #                 local_database_name=local_database_name,
        #                 local_path_dir=self.local_cache_dir,
        #                 force_download=force_download,
        #                 pbar=pbar,
        #             )

        logger.debug(f"Data fetched.")
        logger.debug(f"Processing downloaded dataset.")

        self.dataset = self._process_downloaded(
            self.local_cache_dir,
            local_database_names,
            dataset_names,
        )

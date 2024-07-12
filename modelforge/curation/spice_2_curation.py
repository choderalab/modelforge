from modelforge.curation.curation_baseclass import DatasetCuration
from typing import Optional
from loguru import logger
from openff.units import unit


class SPICE2Curation(DatasetCuration):
    """
    Routines to fetch  the spice 2 dataset from zenodo and process into a curated hdf5 file.

    Small-molecule/Protein Interaction Chemical Energies (SPICE).
    The SPICE dataset containsconformations for a diverse set of small molecules,
    dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
    uncharged molecules, and a wide range of covalent and non-covalent interactions.
    It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
    using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.

    Reference to the original SPICE 1 dataset publication:
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
        Convert from [e.g., angstrom, bohr, hartree] (i.e., source units)
        to [nanometer, kJ/mol] (i.e., target units)

    Examples
    --------
    >>> spice_2_data = SPICE2Curation(hdf5_file_name='spice_2_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/spice_2_dataset')
    >>> spice_2_data.process()

    """

    def _init_dataset_parameters(self):
        from importlib import resources
        from modelforge.curation import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "spice2_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "spice2"

        if self.version_select == "latest":
            self.version_select = data_inputs["latest"]
            logger.debug(f"Latest version: {self.version_select}")

        self.dataset_download_url = data_inputs[self.version_select][
            "dataset_download_url"
        ]
        self.dataset_md5_checksum = data_inputs[self.version_select][
            "dataset_md5_checksum"
        ]
        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
        )

        self.qm_parameters = {
            "geometry": {
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
            "dft_total_force": {
                "u_in": unit.hartree / unit.bohr,
                "u_out": unit.kilojoule_per_mole / unit.angstrom,
            },
            "mbis_charges": {
                "u_in": unit.elementary_charge,
                "u_out": unit.elementary_charge,
            },
            "total_charge": {
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
        series_atom, e.g., charges
        series_mol, e.g., dft energy, dipole moment, etc.
        These ultimately appear under the "format" attribute in the hdf5 file.

        Examples
        >>> series = {'name': 'single_rec', 'atomic_numbers': 'single_atom',
                      ... 'n_configs': 'single_rec', 'geometry': 'series_atom', 'energy': 'series_mol'}
        """

        self._record_entries_series = {
            "name": "single_rec",
            "atomic_numbers": "single_atom",
            "n_configs": "single_rec",
            "smiles": "single_rec",
            "subset": "single_rec",
            "total_charge": "single_rec",
            "geometry": "series_atom",
            "dft_total_energy": "series_mol",
            "dft_total_gradient": "series_atom",
            "dft_total_force": "series_atom",
            "formation_energy": "series_mol",
            "mayer_indices": "series_atom",
            "mbis_charges": "series_atom",
            "mbis_dipoles": "series_atom",
            "mbis_octupoles": "series_atom",
            "mbis_quadrupoles": "series_atom",
            "scf_dipole": "series_mol",
            "scf_quadrupole": "series_mol",
            "wiberg_lowdin_indices": "series_atom",
        }

    def _calculate_reference_charge(self, smiles: str) -> unit.Quantity:
        """
        Calculate the total charge of a molecule from its SMILES string.

        Parameters
        ----------
        smiles: str, required
            SMILES string of the molecule.

        Returns
        -------
        total_charge: unit.Quantity
        """
        from modelforge.utils.io import import_

        Chem = import_("rdkit").Chem
        from rdkit import Chem

        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
        total_charge = sum(atom.GetFormalCharge() for atom in rdmol.GetAtoms())
        return int(total_charge) * unit.elementary_charge

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        atomic_numbers_to_limit: Optional[list] = None,
    ):
        """
        Processes a downloaded dataset: extracts relevant information.

        Parameters
        ----------
        local_path_dir: str, required
            Path to the directory that contains the raw hdf5 datafile
        name: str, required
            Name of the raw hdf5 file,
        max_records: int, optional, default=None
            If set to an integer, 'n_r', the routine will only process the first 'n_r' records, useful for unit tests.
            Can be used in conjunction with max_conformers_per_record and total_conformers.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records and total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with max_records and max_conformers_per_record.
        atomic_numbers_to_limit: list, optional, default=None
            If set to a list of atomic numbers, only records containing these atomic numbers will be processed.

        Examples
        --------
        """
        import h5py
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"

        need_to_reshape = {"formation_energy": True, "dft_total_energy": True}
        with h5py.File(input_file_name, "r") as hf:
            names = list(hf.keys())
            if max_records is None:
                n_max = len(names)
            elif max_records is not None:
                n_max = max_records

            conformers_counter = 0

            for i, name in tqdm(enumerate(names[0:n_max]), total=n_max):
                if total_conformers is not None:
                    if conformers_counter >= total_conformers:
                        break

                # Extract the total number of conformations for a given molecule
                conformers_per_record = hf[name]["conformations"].shape[0]

                keys_list = list(hf[name].keys())

                # temp dictionary for ANI-1x and ANI-1ccx data
                ds_temp = {}

                ds_temp["name"] = f"{name}"
                ds_temp["smiles"] = hf[name]["smiles"][()][0].decode("utf-8")
                ds_temp["atomic_numbers"] = hf[name]["atomic_numbers"][()].reshape(
                    -1, 1
                )
                if max_conformers_per_record is not None:
                    conformers_per_record = min(
                        conformers_per_record,
                        max_conformers_per_record,
                    )
                if total_conformers is not None:
                    conformers_per_record = min(
                        conformers_per_record, total_conformers - conformers_counter
                    )

                ds_temp["n_configs"] = conformers_per_record

                # param_in is the name of the entry, param_data contains input (u_in) and output (u_out) units
                for param_in, param_data in self.qm_parameters.items():
                    # for consistency between datasets, we will all the particle positions "geometry"
                    param_out = param_in
                    if param_in == "geometry":
                        param_in = "conformations"

                    if param_in in keys_list:
                        temp = hf[name][param_in][()]
                        if param_in in need_to_reshape:
                            temp = temp.reshape(-1, 1)

                        temp = temp[0:conformers_per_record]
                        param_unit = param_data["u_in"]
                        if param_unit is not None:
                            # check that units in the hdf5 file match those we have defined in self.qm_parameters
                            try:
                                assert (
                                    hf[name][param_in].attrs["units"]
                                    == param_data["u_in"]
                                )
                            except:
                                msg1 = f'unit mismatch: units in hdf5 file: {hf[name][param_in].attrs["units"]},'
                                msg2 = f'units defined in curation class: {param_data["u_in"]}.'

                                raise AssertionError(f"{msg1} {msg2}")

                            ds_temp[param_out] = temp * param_unit
                        else:
                            ds_temp[param_out] = temp
                ds_temp["total_charge"] = self._calculate_reference_charge(
                    ds_temp["smiles"]
                )
                ds_temp["dft_total_force"] = -ds_temp["dft_total_gradient"]

                # check if the record contains only the elements we are interested in
                # if this has been defined
                add_to_record = True
                if atomic_numbers_to_limit is not None:
                    add_to_record = set(ds_temp["atomic_numbers"].flatten()).issubset(
                        atomic_numbers_to_limit
                    )

                if add_to_record:
                    self.data.append(ds_temp)
                    conformers_counter += conformers_per_record

        if self.convert_units:
            self._convert_units()

        # From documentation: By default, objects inside group are iterated in alphanumeric order.
        # However, if group is created with track_order=True, the insertion order for the group is remembered (tracked)
        # in HDF5 file, and group contents are iterated in that order.
        # As such, we shouldn't need to do sort the objects to ensure reproducibility.
        # self.data = sorted(self.data, key=lambda x: x["name"])

    def process(
        self,
        force_download: bool = False,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
        limit_atomic_species: Optional[list] = None,
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
            Can be used in conjunction with max_conformers_per_record.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records or total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with  max_conformers_per_record.
        limit_atomic_species: list, optional, default=None
            If set to a list of element symbols, records that contain any elements not in this list will be ignored.


        Examples
        --------
        >>> spice_2_data = SPICE2Curation(hdf5_file_name='spice_2_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/spice_2_dataset')
        >>> spice_2_data.process()

        """
        if max_records is not None and total_conformers is not None:
            raise ValueError(
                "max_records and total_conformers cannot be set at the same time."
            )
        from modelforge.utils.remote import download_from_zenodo

        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_zenodo(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )

        self._clear_data()

        if limit_atomic_species is not None:
            self.atomic_numbers_to_limit = []
            from openff.units import elements

            for symbol in limit_atomic_species:
                for num, sym in elements.SYMBOLS.items():
                    if sym == symbol:
                        self.atomic_numbers_to_limit.append(num)
        else:
            self.atomic_numbers_to_limit = None

        # process the rest of the dataset
        if self.name is None:
            raise Exception("Failed to retrieve name of file from zenodo.")
        self._process_downloaded(
            self.local_cache_dir,
            self.name,
            max_records,
            max_conformers_per_record,
            total_conformers,
            self.atomic_numbers_to_limit,
        )

        self._generate_hdf5()

from modelforge.curate import Record, SourceDataset
from modelforge.curate.datasets.curation_baseclass import DatasetCuration
from modelforge.curate.properties import (
    AtomicNumbers,
    Positions,
    Energies,
    Forces,
    PartialCharges,
    TotalCharge,
    MetaData,
    DipoleMomentPerAtom,
    DipoleMomentPerSystem,
    QuadrupoleMomentPerAtom,
    QuadrupoleMomentPerSystem,
    OctupoleMomentPerAtom,
    BondOrders,
)
from typing import Optional
from loguru import logger
from openff.units import unit
import numpy as np


class SPICE2Curation(DatasetCuration):
    """
    Routines to fetch  the spice 2 dataset from zenodo and process into a curated hdf5 file.

    Small-molecule/Protein Interaction Chemical Energies (SPICE).
    The SPICE dataset containsconformations for a diverse set of small molecules,
    dimers, dipeptides, and solvated amino acids. It includes 15 elements, charged and
    uncharged molecules, and a wide range of covalent and non-covalent interactions.
    It provides both forces and energies calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory,
    using Psi4 1.4.1 along with other useful quantities such as multipole moments and bond orders.

    Reference to SPICE 2 publication:
    Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E.
    Nutmeg and SPICE: models and data for biomolecular machine learning.
    Journal of chemical theory and computation, 20(19), 8583-8593 (2024).
    https://doi.org/10.1021/acs.jctc.4c00794

    Reference to the original SPICE 1 dataset publication:
    Eastman, P., Behara, P.K., Dotson, D.L. et al. SPICE,
    A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials.
    Sci Data 10, 11 (2023). https://doi.org/10.1038/s41597-022-01882-6

    Dataset DOI:
    https://doi.org/10.5281/zenodo.8222043

    Parameters
    ----------
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.

    Examples
    --------
    >>> spice_2_data = SPICE2Curation(local_cache_dir='~/datasets/spice_2_dataset')
    >>> spice_2_data.process()
    >>> spice_2_data.to_hdf5(hdf5_file_name='spice2_dataset.hdf5', output_file_dir='~/datasets/hdf5_files')

    """

    def _init_dataset_parameters(self):
        from importlib import resources
        from modelforge.curate.datasets import yaml_files
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
        self.dataset_filename = data_inputs[self.version_select]["dataset_filename"]
        self.dataset_length = data_inputs[self.version_select]["dataset_length"]

        logger.debug(
            f"Dataset: {self.version_select} version: {data_inputs[self.version_select]['version']}"
        )

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

        Chem = import_("rdkit.Chem")
        # from rdkit import Chem

        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)
        total_charge = sum(atom.GetFormalCharge() for atom in rdmol.GetAtoms())
        return np.array([int(total_charge)]) * unit.elementary_charge

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
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
        from modelforge.utils.misc import OpenWithLock

        input_file_name = f"{local_path_dir}/{name}"

        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )

        with OpenWithLock(f"{input_file_name}.lockfile", "w") as lockfile:
            with h5py.File(input_file_name, "r") as hf:
                names = list(hf.keys())

                for name in tqdm(names, desc="Processing records"):
                    # Extract the total number of conformations for a given molecule
                    conformers_per_record = hf[name]["conformations"].shape[0]

                    # if the record doesn't contain any conformations, skip it
                    if conformers_per_record != 0:

                        record_temp = Record(name=name)
                        keys_list = list(hf[name].keys())

                        smiles = MetaData(
                            name="smiles",
                            value=hf[name]["smiles"][()][0].decode("utf-8"),
                        )
                        record_temp.add_property(smiles)
                        atomic_numbers = AtomicNumbers(
                            name="atomic_numbers",
                            value=hf[name]["atomic_numbers"][()].reshape(-1, 1),
                        )
                        record_temp.add_property(atomic_numbers)
                        n_atoms = atomic_numbers.n_atoms

                        positions = Positions(
                            value=hf[name]["conformations"][()],
                            units=hf[name]["conformations"].attrs["units"],
                        )
                        record_temp.add_property(positions)
                        total_charge_temp = self._calculate_reference_charge(
                            smiles.value
                        )
                        total_charge = TotalCharge(
                            value=total_charge_temp.m
                            * np.ones((conformers_per_record, 1)),
                            units=total_charge_temp.u,
                        )
                        record_temp.add_property(total_charge)
                        dft_total_energy = Energies(
                            name="dft_total_energy",
                            value=hf[name]["dft_total_energy"][()].reshape(-1, 1),
                            units=hf[name]["dft_total_energy"].attrs["units"],
                        )
                        record_temp.add_property(dft_total_energy)
                        # note, swap the sign for the forces
                        dft_total_force = Forces(
                            name="dft_total_force",
                            value=-hf[name]["dft_total_gradient"][()],
                            units=hf[name]["dft_total_gradient"].attrs["units"],
                        )
                        record_temp.add_property(dft_total_force)
                        formation_energy = Energies(
                            name="formation_energy",
                            value=hf[name]["formation_energy"][()].reshape(-1, 1),
                            units=hf[name]["formation_energy"].attrs["units"],
                        )
                        record_temp.add_property(formation_energy)
                        if "mbis_charges" in keys_list:
                            mbis_charges = PartialCharges(
                                name="mbis_charges",
                                value=hf[name]["mbis_charges"][()],
                                units=hf[name]["mbis_charges"].attrs["units"],
                            )
                            record_temp.add_property(mbis_charges)
                        # mbis_dipoles, mbis_quadrupoles, mbis_octupoles are per_atom properties

                        if "mbis_dipoles" in keys_list:
                            mbis_dipoles = DipoleMomentPerAtom(
                                name="mbis_dipoles",
                                value=hf[name]["mbis_dipoles"][()],
                                units=hf[name]["mbis_dipoles"].attrs["units"],
                            )
                            record_temp.add_property(mbis_dipoles)
                        if "mbis_quadrupoles" in keys_list:
                            mbis_quadrupoles = QuadrupoleMomentPerAtom(
                                name="mbis_quadrupoles",
                                value=hf[name]["mbis_quadrupoles"][()],
                                units=hf[name]["mbis_quadrupoles"].attrs["units"],
                            )
                            record_temp.add_property(mbis_quadrupoles)
                        if "mbis_octupoles" in keys_list:
                            mbis_octupoles = OctupoleMomentPerAtom(
                                name="mbis_octupoles",
                                value=hf[name]["mbis_octupoles"][()],
                                units=hf[name]["mbis_octupoles"].attrs["units"],
                            )
                            record_temp.add_property(mbis_octupoles)
                        # scf dipole and scf quadrupole are per_system properties
                        scf_dipole = DipoleMomentPerSystem(
                            name="scf_dipole",
                            value=hf[name]["scf_dipole"][()],
                            units=hf[name]["scf_dipole"].attrs["units"],
                        )
                        record_temp.add_property(scf_dipole)
                        scf_quadrupole = QuadrupoleMomentPerSystem(
                            name="scf_quadrupole",
                            value=hf[name]["scf_quadrupole"][()],
                            units=hf[name]["scf_quadrupole"].attrs["units"],
                        )
                        record_temp.add_property(scf_quadrupole)

                        mayer_indices = BondOrders(
                            name="mayer_indices",
                            value=hf[name]["mayer_indices"][()],
                        )
                        record_temp.add_property(mayer_indices)

                        wiberg_lowdin_indices = BondOrders(
                            name="wiberg_lowdin_indices",
                            value=hf[name]["wiberg_lowdin_indices"][()],
                        )
                        record_temp.add_property(wiberg_lowdin_indices)

                        dataset.add_record(record_temp)

        return dataset
        # From documentation: By default, objects inside group are iterated in alphanumeric order.
        # However, if group is created with track_order=True, the insertion order for the group is remembered (tracked)
        # in HDF5 file, and group contents are iterated in that order.
        # As such, we shouldn't need to do sort the objects to ensure reproducibility.
        # self.data = sorted(self.data, key=lambda x: x["name"])

    def process(
        self,
        force_download: bool = False,
    ) -> None:
        """
        Downloads the dataset, extracts relevant information, and writes an hdf5 file.

        Parameters
        ----------
        force_download: bool, optional, default=False
            If the raw data_file is present in the local_cache_dir, the local copy will be used.
            If True, this will force the software to download the data again, even if present.


        Examples
        --------
        >>> spice_2_data = SPICE2Curation(local_cache_dir='~/datasets/spice_2_dataset')
        >>> spice_2_data.process()

        """

        from modelforge.utils.remote import download_from_url

        url = self.dataset_download_url

        # download the dataset
        download_from_url(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.dataset_filename,
            length=self.dataset_length,
            force_download=force_download,
        )

        # process the rest of the dataset

        self.dataset = self._process_downloaded(
            self.local_cache_dir,
            self.dataset_filename,
        )

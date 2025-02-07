from modelforge.curate import Record, SourceDataset
from modelforge.curate.properties import AtomicNumbers, Positions, Energies, Forces
from modelforge.curate.datasets.curation_baseclass import DatasetCuration

from typing import Optional
from loguru import logger
from openff.units import unit


class ANI2xCuration(DatasetCuration):
    """
    Routines to fetch and process the ANI-2x dataset into a curated hdf5 file.

    The ANI-2x data set includes properties for small organic molecules that contain
    H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for nearly 20,000 molecules.
    This will fetch data generated with the wB97X/631Gd level of theory
    used in the original ANI-2x paper, calculated using Gaussian 09

    Citation: Devereux, C, Zubatyuk, R., Smith, J. et al.
                "Extending the applicability of the ANI deep learning molecular potential to sulfur and halogens."
                Journal of Chemical Theory and Computation 16.7 (2020): 4192-4202.
                https://doi.org/10.1021/acs.jctc.0c00121

    DOI for dataset: 10.5281/zenodo.10108941

    Parameters
    ----------
    hdf5_file_name, str, required
        name of the hdf5 file generated for the ANI-1x dataset
    output_file_dir: str, optional, default='./'
        Path to write the output hdf5 files.
    local_cache_dir: str, optional, default='./AN1_dataset'
        Location to save downloaded dataset.

    Examples
    --------
    >>> ani2_data = ANI2xCuration(hdf5_file_name='ani2x_dataset.hdf5',
    >>>                             local_cache_dir='~/datasets/ani2x_dataset')
    >>> ani2_data.process()

    """

    def _init_dataset_parameters(self) -> None:
        """
        Initializes the dataset parameters.

        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curation import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "ani2x_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "ani2x"

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

    def _process_downloaded(
        self,
        local_path_dir: str,
        name: str,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
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
            Can be used in conjunction with max_conformers_per_record.
        max_conformers_per_record: int, optional, default=None
            If set to an integer, 'n_c', the routine will only process the first 'n_c' conformers per record, useful for unit tests.
            Can be used in conjunction with max_records or total_conformers.
        total_conformers: int, optional, default=None
            If set to an integer, 'n_t', the routine will only process the first 'n_t' conformers in total, useful for unit tests.
            Can be used in conjunction with  max_conformers_per_record.


        Examples
        --------
        """
        import h5py
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"
        logger.debug(f"Processing {input_file_name}.")

        conformers_counter = 0

        dataset = SourceDataset("ani2x")
        with h5py.File(input_file_name, "r") as hf:
            #  The ani2x hdf5 file groups molecules by number of atoms
            # we need to break up each of these groups into individual molecules
            mol_counter = 0

            for num_atoms, properties in hf.items():
                species = properties["species"][:]
                coordinates = properties["coordinates"][:]
                energies = properties["energies"][:]
                forces = properties["forces"][:]

                # in the HDF5 file provided for the ANI2x data set,  all configurations of the same size are grouped
                # together into a single array, even if they correspond to different molecules.
                # As a reasonable way to break these up, we use species array to identify unique molecules.
                # This assumes that the species array is a unique way to define a molecule, which of course
                # may not be true, e.g., isomers, etc. (although, if generated from SMILES they will more than likely
                # be in a different order). To get the numbers to match up with what is reported (indirectly),
                # we need to assuming non-consecutive species patterns corresponded to different molecules.

                import numpy as np

                molecules = {}

                last = species[0]

                molecule_name = (
                    f'{np.array2string(species[0], separator="_")}_m{mol_counter}'
                )

                molecules[molecule_name] = []

                for i in range(species.shape[0]):
                    if np.all(species[i] == last):
                        molecules[molecule_name].append(i)
                    else:
                        mol_counter += 1
                        molecule_name = f'{np.array2string(species[0], separator="_")}_m{mol_counter}'
                        molecules[molecule_name] = [i]
                        last = species[i]

                if max_records is None:
                    n_max = len(molecules)
                else:
                    n_max = min(max_records, len(molecules))
                    max_records -= n_max

                if n_max == 0:
                    break

                for molecule_name in tqdm(
                    ([key for key in molecules.keys()][0:n_max]), total=n_max
                ):
                    # stop processing if we have reached the total number of conformers

                    if total_conformers is not None:
                        if conformers_counter >= total_conformers:
                            break
                    record_temp = Record(name=molecule_name)

                    base_index = molecules[molecule_name][0]
                    indices = molecules[molecule_name]

                    atomic_numbers = AtomicNumbers(
                        value=species[base_index].reshape(-1, 1)
                    )
                    record_temp.add_property(atomic_numbers)

                    conformers_per_molecule = len(molecules[molecule_name])
                    if max_conformers_per_record is not None:
                        conformers_per_molecule = min(
                            conformers_per_molecule, max_conformers_per_record
                        )
                    if total_conformers is not None:
                        conformers_per_molecule = min(
                            conformers_per_molecule,
                            total_conformers - conformers_counter,
                        )
                    positions = Positions(
                        value=coordinates[indices][0:conformers_per_molecule],
                        units=unit.angstrom,
                    )
                    record_temp.add_property(positions)

                    energies_mod = Energies(
                        value=energies[indices].reshape(-1, 1)[
                            0:conformers_per_molecule
                        ],
                        units=unit.hartree,
                    )
                    record_temp.add_property(energies_mod)

                    forces_mod = Forces(
                        value=forces[indices][0:conformers_per_molecule],
                        units=unit.hartree / unit.angstrom,
                    )

                    record_temp.add_property(forces_mod)

                    dataset.add_record(record_temp)
                    conformers_counter += conformers_per_molecule

        return dataset

    def process(
        self,
        force_download: bool = False,
        max_records: Optional[int] = None,
        max_conformers_per_record: Optional[int] = None,
        total_conformers: Optional[int] = None,
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


        Examples
        --------
        >>> ani2_data = ANI2xCuration(hdf5_file_name='ani2x_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/ani2x_dataset')
        >>> ani2_data.process()

        """
        if max_records is not None and total_conformers is not None:
            raise Exception(
                "max_records and total_conformers cannot be set at the same time."
            )

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

        # untar and uncompress the dataset
        from modelforge.utils.misc import extract_tarred_file

        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.dataset_filename,
            output_path_dir=self.local_cache_dir,
            mode="r:gz",
        )

        # the untarred file will be in a directory named 'final_h5' within the local_cache_dir,
        hdf5_filename = f"{self.dataset_filename.replace('.tar.gz', '')}.h5"

        # process the rest of the dataset
        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}/final_h5/",
            hdf5_filename,
            max_records,
            max_conformers_per_record,
            total_conformers,
        )

        logger.info(f"writing file {self.hdf5_file_name} to {self.output_file_dir}")
        self.write_hdf5_and_json_files(
            file_name=self.hdf5_file_name, file_path=self.output_file_dir
        )

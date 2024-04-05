from modelforge.curation.curation_baseclass import DatasetCuration
from modelforge.utils.units import *
from typing import Optional
from loguru import logger


class ANI2xCuration(DatasetCuration):
    """
    Routines to fetch and process the ANI-2x dataset into a curated hdf5 file.

    The ANI-2x data set includes properties for small organic molecules that contain
    H, C, N, O, S, F, and Cl.  This dataset contains 9651712 conformers for nearly 200,000 molecules.
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
    convert_units: bool, optional, default=True
        Convert from [e.g., angstrom, bohr, hartree] (i.e., source units)
        to [nanometer, kJ/mol] (i.e., target units)

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
        self.dataset_download_url = (
            "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz"
        )
        self.dataset_md5_checksum = "cb1d9effb3d07fc1cc6ced7cd0b1e1f2"

        # define the parameters in the dataset with their input and output units
        self.qm_parameters = {
            "geometry": {
                "u_in": unit.angstrom,
                "u_out": unit.nanometer,
            },
            "energies": {
                "u_in": unit.hartree,
                "u_out": unit.kilojoule_per_mole,
            },
            "forces": {
                "u_in": unit.hartree / unit.angstrom,
                "u_out": unit.kilojoule_per_mole / unit.nanometer,
            },
        }

    def _init_record_entries_series(self) -> None:
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
            "geometry": "series_atom",
            "energies": "series_mol",
            "forces": "series_atom",
            "atomic_numbers": "single_atom",
            "n_configs": "single_rec",
            "name": "single_rec",
        }

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
        import h5py
        from tqdm import tqdm

        input_file_name = f"{local_path_dir}/{name}"
        logger.debug(f"Processing {input_file_name}.")
        with h5py.File(input_file_name, "r") as hf:
            #  The ani2x hdf5 file groups molecules by number of atoms
            # we need to break up each of these groups into individual molecules

            for num_atoms, properties in hf.items():
                species = properties["species"][:]
                coordinates = properties["coordinates"][:]
                energies = properties["energies"][:]
                forces = properties["forces"][:]

                # in the HDF5 file provided for the ANI2x data set,  all conformers of the same size are grouped
                # together into a single array, even if they correspond to different molecules.
                # As a reasonable way to break these up, we species array to identify unique molecules.
                # This assumes that the species array is a unique way to define a molecule, which of course
                # may not be true, e.g., isomers, etc. (although, if generated from SMILES they will more than likely
                # be in a different order). However, this is a reasonable way to subdivide the dataset.
                # I'll note that this is purely a data organization issue, and not a problem with the data itself,
                # and ultimately will not affect the results of training with this model.

                import numpy as np

                unique_molecules = np.unique(species, axis=0)

                if unit_testing_max_records is None:
                    n_max = unique_molecules.shape[0]
                else:
                    n_max = min(unit_testing_max_records, unique_molecules.shape[0])
                    unit_testing_max_records -= n_max
                if n_max == 0:
                    break
                for i, molecule in tqdm(
                    enumerate(unique_molecules[0:n_max]), total=n_max
                ):
                    ds_temp = {}
                    # molecule represents an aray of atomic species, e.g., [ 8, 8 ] is O_2
                    # here we will create an array of shape( num_confomer, num_atoms) of bools
                    mask_temp = species == molecule

                    # to get the usable mask for the sort, i.e., of shape (n_conformers,),
                    # we will sum over axis=1 in mask_temp; since a True entry has a value of 1
                    # we have a match if the sum is equal to the number of atoms in the molecule
                    mask = np.sum(mask_temp, axis=1) == molecule.shape[0]

                    # We are assuming that the name of the molecule will always be a string
                    # This will allow us to use the value of the name as a dictionary key
                    # We will just make  a string based representation of
                    # the molecule array which stores the atomic numbers, e.g., [ 8, 8 ] becomes "[8_8]"
                    molecule_as_string = np.array2string(molecule, separator="_")

                    ds_temp["name"] = molecule_as_string
                    ds_temp["atomic_numbers"] = molecule.reshape(-1, 1)

                    ds_temp["n_configs"] = int(np.sum(mask))

                    ds_temp["geometry"] = (
                        coordinates[mask] * self.qm_parameters["geometry"]["u_in"]
                    )
                    ds_temp["energies"] = (
                        energies[mask].reshape(-1, 1)
                        * self.qm_parameters["energies"]["u_in"]
                    )
                    ds_temp["forces"] = (
                        forces[mask] * self.qm_parameters["forces"]["u_in"]
                    )

                    self.data.append(ds_temp)
        if self.convert_units:
            self._convert_units()

    def process(
        self,
        force_download: bool = False,
        unit_testing_max_records: Optional[int] = None,
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
        >>> ani2_data = ANI2xCuration(hdf5_file_name='ani2x_dataset.hdf5',
        >>>                             local_cache_dir='~/datasets/ani2x_dataset')
        >>> ani2_data.process()

        """
        from modelforge.utils.remote import download_from_zenodo

        url = self.dataset_download_url

        # download the dataset
        self.name = download_from_zenodo(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            force_download=force_download,
        )

        if self.name is None:
            raise Exception("Failed to retrieve name of file from Zenodo.")

        # clear any data that might be present so we don't append to it
        self._clear_data()

        # untar and uncompress the dataset
        from modelforge.utils.misc import extract_tarred_file

        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.name,
            output_path_dir=self.local_cache_dir,
            mode="r:gz",
        )

        # the untarred file will be in a directory named 'final_h5' within the local_cache_dir,
        hdf5_filename = f"{self.name.replace('.tar.gz', '')}.h5"

        # process the rest of the dataset
        self._process_downloaded(
            f"{self.local_cache_dir}/final_h5/", hdf5_filename, unit_testing_max_records
        )

        self._generate_hdf5()

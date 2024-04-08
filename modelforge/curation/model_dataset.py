from modelforge.curation.curation_baseclass import DatasetCuration, dict_to_hdf5
from modelforge.utils.units import *

import numpy as np

from typing import Optional, List
from loguru import logger


class ModelDataset(DatasetCuration):
    """
    Routines to fetch and process the model dataset used for examining different approaches to generating
    training data.


    """

    def __init__(
        self,
        hdf5_file_name: str,
        output_file_dir: str,
        local_cache_dir: str,
        convert_units: bool = True,
        seed=12345,
    ):
        super().__init__(
            hdf5_file_name=hdf5_file_name,
            output_file_dir=output_file_dir,
            local_cache_dir=local_cache_dir,
            convert_units=convert_units,
        )
        self.seed = seed

    def _init_dataset_parameters(self):
        self.qm_parameters = {
            "geometry": {"u_in": unit.nanometer, "u_out": unit.nanometer},
            "energy": {
                "u_in": unit.kilojoule_per_mole,
                "u_out": unit.kilojoule_per_mole,
            },
        }

    def _init_record_entries_series(self):
        self._record_entries_series = {
            "name": "single_rec",
            "n_configs": "single_rec",
            "atomic_numbers": "single_atom",
            "geometry": "series_atom",
            "energy": "series_mol",
        }

    def _process_downloaded(
        self,
        local_path_dir: str,
        filename: str,
        model: str,
    ):
        file_path = f"{local_path_dir}/{filename}"

        import h5py

        data_temp = []
        with h5py.File(file_path, "r") as f:
            molecule_names = list(f.keys())
            for molecule_name in molecule_names:
                record_temp = {}
                molecule = f[molecule_name]
                for key in molecule.keys():
                    temp = molecule[key][()]
                    if "u" in molecule[key].attrs:
                        temp = temp * unit(molecule[key].attrs["u"])
                    record_temp[key] = temp
                record_temp["name"] = molecule_name
                data_temp.append(record_temp)

        self.data = []
        self.test_data_molecules = []
        self.test_data_conformers = []

        # figure out how which molecules we have in our holdout set
        # we will keep 10 % of the data for testing
        n_molecules = len(data_temp)
        from numpy.random import RandomState

        prng = RandomState(self.seed)
        hold_out = prng.randint(n_molecules, size=(int(n_molecules * 0.1)))

        if model == "PURE_MM":
            for i, record in enumerate(data_temp):
                temp = {}
                temp["name"] = record["name"]
                temp["atomic_numbers"] = record["atomic_numbers"].reshape(-1, 1)

                temp_conf_holdout = {}
                temp_conf_holdout["name"] = record["name"]
                temp_conf_holdout["atomic_numbers"] = record["atomic_numbers"].reshape(
                    -1, 1
                )

                if i in hold_out:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["MM_emin_ML_energy"].m).reshape(-1, 1),
                                record["MM_300_ML_energy"].m.reshape(-1, 1),
                                record["MM_100_ML_energy"].m.reshape(-1, 1),
                            )
                        )
                        * record["MM_emin_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["MM_emin_coords"].m.reshape(1, -1, 3),
                                record["MM_coords_300"].m,
                                record["MM_coords_100"].m,
                            )
                        )
                        * record["MM_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].m.shape[0]
                    self.test_data_molecules.append(temp)
                else:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["MM_emin_ML_energy"].m).reshape(-1, 1),
                                record["MM_300_ML_energy"][0:9].m.reshape(-1, 1),
                                record["MM_100_ML_energy"][0:9].m.reshape(-1, 1),
                            )
                        )
                        * record["MM_emin_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["MM_emin_coords"].m.reshape(1, -1, 3),
                                record["MM_coords_300"][0:9].m,
                                record["MM_coords_100"][0:9].m,
                            )
                        )
                        * record["MM_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].m.shape[0]

                    self.data.append(temp)

                    temp_conf_holdout["energy"] = (
                        np.vstack(
                            (
                                record["MM_300_ML_energy"][9:10].m.reshape(-1, 1),
                                record["MM_100_ML_energy"][9:10].m.reshape(-1, 1),
                            )
                        )
                        * record["MM_300_ML_energy"].u
                    )
                    temp_conf_holdout["geometry"] = (
                        np.vstack(
                            (
                                record["MM_coords_300"][9:10].m,
                                record["MM_coords_100"][9:10].m,
                            )
                        )
                        * record["MM_emin_coords"].u
                    )
                    temp_conf_holdout["n_configs"] = temp_conf_holdout[
                        "geometry"
                    ].m.shape[0]
                    self.test_data_conformers.append(temp_conf_holdout)

        if model == "PURE_MM_low_temp_correction":
            for i, record in enumerate(data_temp):
                temp = {}
                temp["name"] = record["name"]
                temp["atomic_numbers"] = record["atomic_numbers"].reshape(-1, 1)

                temp_conf_holdout = {}
                temp_conf_holdout["name"] = record["name"]
                temp_conf_holdout["atomic_numbers"] = record["atomic_numbers"].reshape(
                    -1, 1
                )

                if i in hold_out:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["MM_emin_ML_energy"].m).reshape(-1, 1),
                                record["MM_300_ML_energy"].m.reshape(-1, 1),
                                record["MM_100_ML_energy"].m.reshape(-1, 1),
                                record["MM100_ML_emin_ML_energy"].m.reshape(-1, 1),
                            )
                        )
                        * record["MM_300_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["MM_emin_coords"].m.reshape(1, -1, 3),
                                record["MM_coords_300"].m,
                                record["MM_coords_100"].m,
                                record["MM100_ML_emin_coords"].m,
                            )
                        )
                        * record["MM_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].m.shape[0]
                    self.test_data_molecules.append(temp)
                else:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["MM_emin_ML_energy"].m).reshape(-1, 1),
                                record["MM_300_ML_energy"][0:9].m.reshape(-1, 1),
                                record["MM_100_ML_energy"][0:9].m.reshape(-1, 1),
                                record["MM100_ML_emin_ML_energy"][0:9].m.reshape(-1, 1),
                            )
                        )
                        * record["MM_emin_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["MM_emin_coords"].m.reshape(1, -1, 3),
                                record["MM_coords_300"][0:9].m,
                                record["MM_coords_100"][0:9].m,
                                record["MM100_ML_emin_coords"][0:9].m,
                            )
                        )
                        * record["MM_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].m.shape[0]
                    self.data.append(temp)

                    temp_conf_holdout["energy"] = (
                        np.vstack(
                            (
                                record["MM_300_ML_energy"][9:10].m.reshape(-1, 1),
                                record["MM_100_ML_energy"][9:10].m.reshape(-1, 1),
                                record["MM100_ML_emin_ML_energy"][9:10].m.reshape(
                                    -1, 1
                                ),
                            )
                        )
                        * record["MM_300_ML_energy"].u
                    )
                    temp_conf_holdout["geometry"] = (
                        np.vstack(
                            (
                                record["MM_coords_300"][9:10].m,
                                record["MM_coords_100"][9:10].m,
                                record["MM100_ML_emin_coords"][9:10].m,
                            )
                        )
                        * record["MM_coords_300"].u
                    )
                    temp_conf_holdout["n_configs"] = temp_conf_holdout[
                        "geometry"
                    ].shape[0]
                    self.test_data_conformers.append(temp_conf_holdout)

        if model == "PURE_ML":
            for i, record in enumerate(data_temp):
                temp = {}
                temp["name"] = record["name"]
                temp["atomic_numbers"] = record["atomic_numbers"].reshape(-1, 1)

                temp_conf_holdout = {}
                temp_conf_holdout["name"] = record["name"]
                temp_conf_holdout["atomic_numbers"] = record["atomic_numbers"].reshape(
                    -1, 1
                )

                if i in hold_out:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["ML_emin_ML_energy"].m).reshape(-1, 1),
                                record["ML_300_ML_energy"].m.reshape(-1, 1),
                                record["ML_100_ML_energy"].m.reshape(-1, 1),
                            )
                        )
                        * record["ML_emin_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["ML_emin_coords"].m.reshape(1, -1, 3),
                                record["ML_coords_300"].m,
                                record["ML_coords_100"].m,
                            )
                        )
                        * record["ML_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].shape[0]
                    self.test_data_molecules.append(temp)
                else:
                    temp["energy"] = (
                        np.vstack(
                            (
                                np.array(record["ML_emin_ML_energy"].m).reshape(-1, 1),
                                record["ML_300_ML_energy"][0:9].m.reshape(-1, 1),
                                record["ML_100_ML_energy"][0:9].m.reshape(-1, 1),
                            )
                        )
                        * record["ML_emin_ML_energy"].u
                    )
                    temp["geometry"] = (
                        np.vstack(
                            (
                                record["ML_emin_coords"].m.reshape(1, -1, 3),
                                record["ML_coords_300"][0:9].m,
                                record["ML_coords_100"][0:9].m,
                            )
                        )
                        * record["ML_emin_coords"].u
                    )
                    temp["n_configs"] = temp["geometry"].m.shape[0]
                    self.data.append(temp)

                    temp_conf_holdout["energy"] = (
                        np.vstack(
                            (
                                record["ML_300_ML_energy"][9:10].m.reshape(-1, 1),
                                record["ML_100_ML_energy"][9:10].m.reshape(-1, 1),
                            )
                        )
                        * record["ML_300_ML_energy"].u
                    )
                    temp_conf_holdout["geometry"] = (
                        np.vstack(
                            (
                                record["ML_coords_300"][9:10].m,
                                record["ML_coords_100"][9:10].m,
                            )
                        )
                        * record["ML_coords_300"].u
                    )
                    temp_conf_holdout["n_configs"] = temp_conf_holdout[
                        "geometry"
                    ].shape[0]
                    self.test_data_conformers.append(temp_conf_holdout)

    def _generate_hdf5_file(self, data, output_file_path, filename):
        full_file_path = f"{output_file_path}/{filename}"
        logger.debug("Writing data HDF5 file.")
        import os

        os.makedirs(output_file_path, exist_ok=True)

        dict_to_hdf5(
            full_file_path,
            data,
            series_info=self._record_entries_series,
            id_key="name",
        )

    def process(
        self,
        input_data_path="./",
        input_data_file="molecule_data.hdf5",
        data_combination="pure_MM",
    ) -> None:
        """
        Process the dataset into a curated hdf5 file.

        Parameters
        ----------
        force_download : Optional[bool], optional
            Force download of the dataset, by default False
        unit_testing_max_records : Optional[int], optional
            Maximum number of records to process, by default None

        """
        self.data_combination = data_combination
        self._clear_data()
        self._process_downloaded(
            input_data_path, input_data_file, self.data_combination
        )
        if self.convert_units:
            self._convert_units()

        # for datapoint in self.data:
        #     print(datapoint["name"])

        self._generate_hdf5_file(self.data, self.output_file_dir, self.hdf5_file_name)

        fileout = self.hdf5_file_name.replace(".hdf5", "_test_conformers.hdf5")
        self._generate_hdf5_file(
            self.test_data_conformers, self.output_file_dir, fileout
        )
        fileout = self.hdf5_file_name.replace(".hdf5", "_test_molecules.hdf5")
        self._generate_hdf5_file(
            self.test_data_molecules, self.output_file_dir, fileout
        )

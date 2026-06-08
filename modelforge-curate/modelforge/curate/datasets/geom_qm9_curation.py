from modelforge.curate import Record, SourceDataset, TotalCharge
from modelforge.curate.properties import (
    AtomicNumbers,
    DipoleMomentScalarPerSystem,
    Energies,
    MetaData,
    PropertyBaseModel,
    PartialCharges,
    Polarizability,
    Positions,
)
from modelforge.curate.datasets.curation_baseclass import DatasetCuration

import numpy as np

from typing import Optional, List
from loguru import logger
from openff.units import unit


class GEOMQM9Curation(DatasetCuration):
    """
    Routines to fetch and process the QM9 subset of the GEOM dataset into a curated hdf5 file.

    The original QM9 dataset includes 133,885 organic molecules with up to nine total heavy atoms (C,O,N,or F; excluding H).

    Citation:   Ramakrishnan, R., Dral, P., Rupp, M. et al.
               "Quantum chemistry structures and properties of 134 kilo molecules."
                Sci Data 1, 140022 (2014).
                https://doi.org/10.1038/sdata.2014.22

    DOI for original qm9 dataset: 10.6084/m9.figshare.c.978904.v5

    The Geometric Ensemble Of Molecules (GEOM) dataset contains conformers for 133,000 species from the QM9 dataset.
    Conformer structures were generated with CREST and GFN2-XTB.  Energies were evaluated using
    DFT via ORCA 5.0.2  using the r2scan-3c functional and mTZVPP basis.

    Citation:   Axelrod, S., Gómez-Bombarelli, R.
                GEOM, energy-annotated molecular conformations for property prediction and molecular generation.
                Sci Data 9, 185 (2022).
                https://doi.org/10.1038/s41597-022-01288-4

    DOI for dataset: https://doi.org/10.7910/DVN/JNGTDF
    Download link for qm9 subset of the GEOM dataset:  https://dataverse.harvard.edu/api/access/datafile/4327190

    Parameters
    ----------
    local_cache_dir: str, optional, default='./'
        Location to save downloaded dataset.

    Examples
    --------
    >>> geom_qm9_data = GEOMQM9Curation(local_cache_dir='~/mf_datasets/geom_qm9_dataset')
    >>> geom_qm9_data.process()
    >>> geom_qm9_data.to_hd5(hdf5_file_name='geom_qm9_dataset.hdf5', output_file_dir='~/datasets/hdf5_files')

    """

    def _init_dataset_parameters(self) -> None:
        """
        Define the key parameters for the QM9 dataset.
        """
        # read in the yaml file that defines the dataset download url and md5 checksum
        # this yaml file should be stored along with the curated dataset

        from importlib import resources
        from modelforge.curate.datasets import yaml_files
        import yaml

        yaml_file = resources.files(yaml_files) / "geom_qm9_curation.yaml"
        logger.debug(f"Loading config data from {yaml_file}")
        with open(yaml_file, "r") as file:
            data_inputs = yaml.safe_load(file)

        assert data_inputs["dataset_name"] == "geom_qm9"

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
        local_file_path: str,
    ) -> SourceDataset:
        """
        Processes the downloaded dataset

        Parameters
        ----------
        local_file_path: str, required
            Path to the extracted msgpack file


        Examples
        --------

        """
        from tqdm import tqdm
        import os
        import msgpack

        # first check that the file exists
        if os.path.exists(local_file_path) == False:
            raise FileNotFoundError(f"{local_file_path} does not exist.")

        # create the source dataset where we will store the records
        dataset = SourceDataset(
            name=self.dataset_name, local_db_dir=self.local_cache_dir
        )

        # load up the message pack file
        source_unpacker = msgpack.Unpacker(open(local_file_path, "rb"))

        # grab all the records so we don't have issues with records
        # being broken up across different chunks
        # then having to stitch them back together

        all_records = {}
        for data_chunk in source_unpacker:
            all_records.update(data_chunk)

        data_keys = list(all_records.keys())

        # data_keys correspond to the SMILES string representing the molecule

        for data_key in tqdm(data_keys):
            # create a record to store the info
            record = Record(name=data_key, append_property=True)

            # add a metadata entry for the smiles
            smiles_prop = MetaData(name="smiles", value=data_key)

            # total charge of the system
            total_system_charge = all_records[data_key]["charge"]

            first = True
            # now let us loop over each conformer
            n_confs = len(all_records[data_key]["conformers"])
            for conformer in all_records[data_key]["conformers"]:
                xyz_temp = conformer["xyz"]
                energy_temp = conformer["totalenergy"]

                # parse xyz_temp to separate atomic numbers and positions
                # stored in standard atomic_number, x, y, z format
                atomic_numbers = np.array(xyz_temp)[:, 0]
                if first:
                    atomic_numbers_prop = AtomicNumbers(
                        value=atomic_numbers.reshape(-1, 1)
                    )
                    record.add_property(atomic_numbers_prop)
                    first = False

                positions = np.array(xyz_temp)[:, 1:4]
                positions_prop = Positions(
                    value=positions.reshape(1, -1, 3), units=unit.angstrom
                )
                record.add_property(positions_prop)

                energy_prop = Energies(
                    name="dft_total_energy",
                    value=np.array(energy_temp).reshape(1, 1),
                    units=unit.hartree,
                )
                record.add_property(energy_prop)

            # we want a total_charge entry for each conformer
            total_system_charge = np.ones([n_confs, 1]) * total_system_charge
            total_charge_prop = TotalCharge(
                value=total_system_charge, units=unit.elementary_charge
            )
            record.add_properties([smiles_prop, total_charge_prop])
            dataset.add_record(record)

        return dataset

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
        >>> geom_qm9_data = GEOMQM9Curation(local_cache_dir='~/datasets/geom_qm9_dataset')
        >>> geom_qm9_data.process()

        """

        from modelforge.utils.remote import download_from_url

        url = self.dataset_download_url

        # download the dataset
        status = download_from_url(
            url=url,
            md5_checksum=self.dataset_md5_checksum,
            output_path=self.local_cache_dir,
            output_filename=self.dataset_filename,
            length=self.dataset_length,
            force_download=force_download,
        )
        if status == False:
            logger.info(f"Could not download file from {url}")
            raise Exception("Failed to download file")

        # untar the dataset
        from modelforge.utils.misc import extract_tarred_file

        # extract the tar.gz file into the local_cache_dir
        extract_tarred_file(
            input_path_dir=self.local_cache_dir,
            file_name=self.dataset_filename,
            output_path_dir=f"{self.local_cache_dir}",
            mode="r:gz",
        )

        self.extracted_filename = self.dataset_filename.replace(".tar.gz", "")

        self.dataset = self._process_downloaded(
            f"{self.local_cache_dir}/{self.extracted_filename}",
        )

import os
from typing import List

import h5py
import torch
import qcportal as ptl

from .dataset import Dataset

class QM9Dataset(Dataset):
    """
    QM9 dataset as curated by qcarchive.
    """

    def set_raw_dataset_file(self, raw_input_file: str):
        """
        Defines raw hd5f dataset input from qcarchive.

        Parameters
        ----------
            raw_input : str
                file path to the hdf5 input raw dataset
        """
        # convert to exceptions
        if not raw_input_file.endswith('.hdf5'):
            raise Exception('Must be an .hdf5 file.')
        if not os.path.isfile(raw_input_file):
            raise Exception('File not found.')

        self._raw_input_file = raw_input_file

    def save_raw_dataset(self, raw_output_file: str):
        """
        Defines the hdf5 file for saving save the raw dataset.

        Loading this file via set_raw_dataset_file will avoid
        the need to re-download the data.

        Parameters
        ----------
            raw_output_file : str
                file path to save raw dataset
        """
        # convert to exceptions
        if not raw_output_file.endswith('.hdf5'):
            raise Exception('Must be an .hdf5 file.')

        try:
            self.qm9.to_file(path=raw_output_file, encoding='hdf5')
        except:
            print('failed to save file')


    def load(self):
        """
        Loads the raw dataset from qcarchive.

        If a valid qcarchive generated hdf5 file is not pass to the
        set_raw_dataset_file function, the code will download the
        raw dataset from qcarchive. Note, to save
        """

        qcp_client = ptl.FractalClient()

        # to get QM9 from qcportal, we need to define which collection and QM9
        # we will first check to see if it exists
        qcportal_data = {'collection': 'Dataset', 'dataset': 'QM9'}

        try:
            self.qm9 = qcp_client.get_collection(qcportal_data['collection'], qcportal_data['dataset'])
        except:
            print(f"Dataset {qcportal_data['dataset']} is not available in collection {qcportal_data['collection']}.")

        # if we didn't set the code to use a raw input file, we will just download
        try:
            self.qm9.set_view(self._raw_input_file)
        except:
            self.qm9.download()

    # this might need to take a dictionary or something of
    # what data we want to add to the pytorch tensor
    def prepare_dataset(self):
        """"
        Parse the QM9 datset into dictionaries that we will set in a tensor

        Note: this function takes a while to run, as retrieving the records
        appears to be a bit slow given the dataset size.
        """

        molecules = self.qm9.get_molecules()
        records = self.qm9.get_records(method='b3lyp')

        records_dict = {}
        molecules_dict = {}

        names = []

        for i in range(molecules.shape[0]):
            name = molecules.iloc[i].name
            molecules_dict[name] = molecules.iloc[i][0]
            names.append(name)

        for i in range(len(records)):
            rec = records.iloc[i].record
            name = records.index[i]
            records_dict[name] = rec

        # this return is just for initial testing purposes
        return (names, molecules_dict, records_dict)

        def save_prepared_dataset(self):
            print('not implemented yet')
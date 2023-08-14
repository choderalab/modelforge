from .dataset import BaseDataset


class QM9Dataset(BaseDataset):
    """
    QM9 dataset as curated by qcarchive.
    """

    def __init__(self, dataset_name: str = "QM9"):
        self.dataset_name = dataset_name
        super().__init__()

    def download_hdf_file(self):
        self._download_from_gdrive()

    def _download_from_gdrive(self):
        import gdown

        id = "1h3eh-79wQy69_I7Fr-BoYNvHW6wYisPc"
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, self.raw_dataset_file, quiet=False)

    @property
    def qcportal_data(self):
        return {"collection": "Dataset", "dataset": "QM9"}

    @property
    def _dataset_records(self):
        return {
            "method": "b3lyp",
            "basis": "def2-svp",
            "program": "psi4",
        }

    def transform_y(self, energy):
        return energy

    def transform_x(self, geometry):
        return geometry

    def from_cache(self):
        pass

    def load(self):
        pass

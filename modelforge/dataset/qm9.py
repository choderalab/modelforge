from .dataset import BaseDataset


class QM9Dataset(BaseDataset):
    """
    QM9 dataset as curated by qcarchive.
    """

    def __init__(self, dataset_name: str = "QM9"):
        self.dataset_name = dataset_name
        super().__init__()

    @property
    def url(self):
        return "https://www.dropbox.com/scl/fo/e74c3s2obow921uoexie8/h?rlkey=9yn6moonczjlqt1aekoqu0eju&dl=0"

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

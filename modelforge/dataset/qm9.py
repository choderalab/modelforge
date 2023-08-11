from .dataset import Dataset


class QM9Dataset(Dataset):
    """
    QM9 dataset as curated by qcarchive.
    """

    @property
    def qcportal_data(self):
        return {"collection": "Dataset", "dataset": "QM9"}

    @property
    def name(self):
        return "QM9"

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


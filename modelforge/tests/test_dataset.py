import sys
import pytest
from modelforge.dataset import QM9Dataset


def test_dataset_imported():
    """Sample test, will always pass so long as import statement worked."""
    import modelforge.dataset

    assert "modelforge.dataset" in sys.modules


# fixture let's you pass different datasets and performs the same download operation on them
@pytest.mark.parametrize("dataset", [QM9Dataset])
def test_download_dataset(dataset):
    print(dataset.name)
    dataset("tmp.hdf5")

import sys
import pytest
import modelforge
from modelforge.utils.zenodo import *


def test_zenodo_fetch():
    # qm9 dataset
    files = hdf5_from_zenodo(record="10.5281/zenodo.3588339")
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = hdf5_from_zenodo(record="https://dx.doi.org/10.5281/zenodo.3588339")
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = hdf5_from_zenodo(record="https://zenodo.org/record/3588339")
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = hdf5_from_zenodo(record="3588339")
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    # comp6 dataset
    files = hdf5_from_zenodo("10.5281/zenodo.3588368")
    assert len(files) == 6
    assert any("205.hdf5.gz" in file for file in files)
    assert any("207.hdf5.gz" in file for file in files)
    assert any("208.hdf5.gz" in file for file in files)
    assert any("209.hdf5.gz" in file for file in files)
    assert any("210.hdf5.gz" in file for file in files)
    assert any("211.hdf5.gz" in file for file in files)


def test_fetch_record_id():
    record = fetch_url_from_doi(doi="10.5281/zenodo.3588339")
    assert record == "https://zenodo.org/record/3588339"

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.fake.3588339")

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.3588339", timeout=0.0000000000001)


def test_fetch_data():
    files = get_zenodo_datafiles(record_id="3588339", file_extension=".hdf5.gz")

    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = get_zenodo_datafiles(record_id="3588339", file_extension=".gz")
    assert len(files) == 2
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )
    assert (
        files[1]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.tar.gz"
    )

    files = get_zenodo_datafiles(record_id="3588339", file_extension=".txt")
    assert len(files) == 0

    with pytest.raises(Exception):
        get_zenodo_datafiles(record_id="3588339bad", file_extension=".txt")

    with pytest.raises(Exception):
        get_zenodo_datafiles(
            record_id="3588339", timeout=0.000000000001, file_extension=".hdf5.gz"
        )

import sys
import pytest
import modelforge
import os

from modelforge.utils.remote import *


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("remote_test")
    return fn


def test_is_url():
    assert (
        is_url(query="https://dx.doi.org/10.5281/zenodo.3588339", hostname="doi.org")
        == True
    )
    assert (
        is_url(query="https://zenodo.org/record/3588339", hostname="zenodo.org") == True
    )
    assert (
        is_url(query="https://zenodo.org/record/3588339", hostname="choderalab.org")
        == False
    )


def test_download_from_figshare(prep_temp_dir):
    url = "https://figshare.com/ndownloader/files/22247589"
    name = download_from_figshare(
        url=url,
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    with pytest.raises(Exception):
        url = "https://choderalab.com/ndownloader/files/22247589"
        name = download_from_figshare(
            url=url,
            output_path=str(prep_temp_dir),
            force_download=True,
        )


def test_record_id_parse():
    assert (
        parse_zenodo_record_id_from_url("https://zenodo.org/record/3588339")
        == "3588339"
    )
    assert (
        parse_zenodo_record_id_from_url("https://zenodo.org/record/3588339/")
        == "3588339"
    )
    with pytest.raises(Exception):
        parse_zenodo_record_id_from_url("https://zenodo.org/record/bad/3588339")


def test_zenodo_fetch():
    # qm9 dataset
    files = datafiles_from_zenodo(
        record="10.5281/zenodo.3588339", file_extension="hdf5.gz"
    )
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = datafiles_from_zenodo(
        record="https://dx.doi.org/10.5281/zenodo.3588339", file_extension="hdf5.gz"
    )
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = datafiles_from_zenodo(
        record="https://zenodo.org/record/3588339", file_extension="hdf5.gz"
    )
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    files = datafiles_from_zenodo(record="3588339", file_extension="hdf5.gz")
    assert len(files) == 1
    assert (
        files[0]
        == "https://zenodo.org/api/files/0a55a53a-69c3-4cd8-8ab8-d031ca0d6853/155.hdf5.gz"
    )

    # comp6 dataset
    files = datafiles_from_zenodo("10.5281/zenodo.3588368", file_extension="hdf5.gz")
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


def test_fetch_datafiles():
    files = get_zenodo_datafiles(
        record_id="https://zenodo.org/record/3588339", file_extension=".hdf5.gz"
    )

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


def test_download_from_zenodo(prep_temp_dir):
    url = "https://zenodo.org/record/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    name = download_from_zenodo(
        url=url,
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    with pytest.raises(Exception):
        url = "https://choderalab.com/22247589"
        name = download_from_zenodo(
            url=url,
            output_path=str(prep_temp_dir),
            force_download=True,
        )

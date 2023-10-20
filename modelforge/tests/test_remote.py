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


def test_fetch_record_id():
    record = fetch_url_from_doi(doi="10.5281/zenodo.3588339")
    assert record == "https://zenodo.org/records/3588339"

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.fake.3588339")

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.3588339", timeout=0.0000000000001)


def test_download_from_zenodo(prep_temp_dir):
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    zenodo_checksum = "194cde222565dca8657d8521e5df1fd8"
    name = download_from_zenodo(
        url=url,
        zenodo_md5_checksum=zenodo_checksum,
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    with pytest.raises(Exception):
        url = "https://choderalab.com/22247589"
        name = download_from_zenodo(
            url=url,
            zenodo_md5_checksum=zenodo_checksum,
            output_path=str(prep_temp_dir),
            force_download=True,
        )


def test_md5_calculation(prep_temp_dir):
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    zenodo_checksum = "194cde222565dca8657d8521e5df1fd8"

    name = download_from_zenodo(
        url=url,
        zenodo_md5_checksum=zenodo_checksum,
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    # explicit direct check of the function, even though included in download_from_zenodo
    calculated_checksum = calculate_md5_checksum(
        file_name=name, file_path=str(prep_temp_dir)
    )

    assert zenodo_checksum == calculated_checksum

    with pytest.raises(Exception):
        bad_checksum = "294badmd5checksumthatwontwork9de"
        name = download_from_zenodo(
            url=url,
            zenodo_md5_checksum=bad_checksum,
            output_path=str(prep_temp_dir),
            force_download=True,
        )

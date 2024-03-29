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


def test_download_from_url(prep_temp_dir):
    url = "https://raw.githubusercontent.com/choderalab/modelforge/e3e65e15e23ccc55d03dd7abb4b9add7a7dd15c3/modelforge/modelforge.py"
    checksum = "66ec18ca5db3df5791ff1ffc584363a8"
    # Download the file
    download_from_url(
        url,
        md5_checksum=checksum,
        output_path=str(prep_temp_dir),
        output_filename="modelforge.py",
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + "/modelforge.py"
    assert os.path.isfile(file_name_path)

    # create a dummy document to test the case where
    # the checksum doesn't match so it will redownload
    with open(file_name_path, "w") as f:
        f.write("dummy document")

    # This will force a download because the checksum doesn't match
    download_from_url(
        url,
        md5_checksum=checksum,
        output_path=str(prep_temp_dir),
        output_filename="modelforge.py",
        force_download=False,
    )

    file_name_path = str(prep_temp_dir) + "/modelforge.py"
    assert os.path.isfile(file_name_path)

    # let us change the expected checksum to cause a failure
    with pytest.raises(Exception):
        url = "https://choderalab.com/modelforge.py"
        download_from_url(
            url,
            md5_checksum="checksum_garbage",
            output_path=str(prep_temp_dir),
            output_filename="modelforge.py",
            force_download=True,
        )


def test_download_from_figshare(prep_temp_dir):
    url = "https://figshare.com/ndownloader/files/22247589"
    name = download_from_figshare(
        url=url,
        md5_checksum="c1459c5ddce7bb94800032aa3d04788e",
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    # create a dummy document to test the case where
    # the checksum doesn't match so it will redownload
    with open(file_name_path, "w") as f:
        f.write("dummy document")

    # This will force a download because the checksum doesn't match
    url = "https://figshare.com/ndownloader/files/22247589"
    name = download_from_figshare(
        url=url,
        md5_checksum="c1459c5ddce7bb94800032aa3d04788e",
        output_path=str(prep_temp_dir),
        force_download=False,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    # the length of this file isn't listed in the headers
    # this will check to make sure we can handle this case
    url = "https://figshare.com/ndownloader/files/30975751"
    name = download_from_figshare(
        url=url,
        md5_checksum="efa40abff1f71c121f6f0d444c18d5b3",
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    with pytest.raises(Exception):
        url = "https://choderalab.com/ndownloader/files/22247589"
        name = download_from_figshare(
            url=url,
            md5_checksum="c1459c5ddce7bb94800032aa3d04788e",
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
        md5_checksum=zenodo_checksum,
        output_path=str(prep_temp_dir),
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    # create a dummy document to test the case where
    # the checksum doesn't match so it will redownload
    with open(file_name_path, "w") as f:
        f.write("dummy document")

    # make sure that we redownload the file because the checksum of the
    # existing file doesn't match
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    zenodo_checksum = "194cde222565dca8657d8521e5df1fd8"
    name = download_from_zenodo(
        url=url,
        md5_checksum=zenodo_checksum,
        output_path=str(prep_temp_dir),
        force_download=False,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    with pytest.raises(Exception):
        url = "https://choderalab.com/22247589"
        name = download_from_zenodo(
            url=url,
            md5_checksum=zenodo_checksum,
            output_path=str(prep_temp_dir),
            force_download=True,
        )


def test_md5_calculation(prep_temp_dir):
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    zenodo_checksum = "194cde222565dca8657d8521e5df1fd8"

    name = download_from_zenodo(
        url=url,
        md5_checksum=zenodo_checksum,
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
            md5_checksum=bad_checksum,
            output_path=str(prep_temp_dir),
            force_download=True,
        )

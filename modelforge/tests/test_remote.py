import sys
import pytest
import modelforge
import os

from modelforge.utils.remote import *


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_remote_temp")
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
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    checksum = "194cde222565dca8657d8521e5df1fd8"

    name = "atools_ml-v0.1.zip"
    # Download the file
    download_from_url(
        url,
        md5_checksum=checksum,
        output_path=str(prep_temp_dir),
        output_filename=name,
        force_download=True,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
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
        output_filename=name,
        force_download=False,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    # let us change the expected checksum to cause a failure
    # this will see this as not matching and will redownload,
    # but since the new file doesn't match it will raise an exception
    with pytest.raises(Exception):
        download_from_url(
            url,
            md5_checksum="checksum_garbage",
            output_path=str(prep_temp_dir),
            output_filename=name,
            force_download=True,
        )


def test_fetch_record_id():
    record = fetch_url_from_doi(doi="10.5281/zenodo.3588339")
    assert record == "https://zenodo.org/records/3588339"

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.fake.3588339")

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.3588339", timeout=0.0000000000001)


def test_md5_calculation(prep_temp_dir):
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    zenodo_checksum = "194cde222565dca8657d8521e5df1fd8"

    name = "atools_ml-v0.1.zip"
    download_from_url(
        url=url,
        md5_checksum=zenodo_checksum,
        output_path=str(prep_temp_dir),
        output_filename=name,
        force_download=True,
    )

    # explicit direct check of the function, even though included in download_from_zenodo
    calculated_checksum = calculate_md5_checksum(
        file_name=name, file_path=str(prep_temp_dir)
    )

    assert zenodo_checksum == calculated_checksum

    with pytest.raises(Exception):
        bad_checksum = "294badmd5checksumthatwontwork9de"
        download_from_url(
            url=url,
            md5_checksum=bad_checksum,
            output_path=str(prep_temp_dir),
            output_filename=name,
            force_download=True,
        )


def test_load_from_wandb(prep_temp_dir):
    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    nn_potential = NeuralNetworkPotentialFactory().load_from_wandb(
        run_path="modelforge_nnps/test_ANI2x_on_dataset/model-qloqn6gk",
        version="v0",
        local_cache_dir=f"{prep_temp_dir}/test_wandb",
    )

    assert os.path.isfile(f"{prep_temp_dir}/test_wandb/model.ckpt")

    assert nn_potential is not None

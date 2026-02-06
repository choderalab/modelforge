import sys
import pytest
import modelforge
import os

from modelforge.utils.remote import *

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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

    output_dir = f"{str(prep_temp_dir)}/download_test"
    name = "atools_ml-v0.1.zip"
    # Download the file
    download_from_url(
        url,
        md5_checksum=checksum,
        output_path=output_dir,
        output_filename=name,
        force_download=True,
        scheme="auto",
    )

    file_name_path = f"{output_dir}/{name}"
    assert os.path.isfile(file_name_path)

    # create a dummy document to test the case where
    # the checksum doesn't match so it will redownload
    with open(file_name_path, "w") as f:
        f.write("dummy document")

    # This will force a download because the checksum doesn't match
    download_from_url(
        url,
        md5_checksum=checksum,
        output_path=output_dir,
        output_filename=name,
        force_download=False,
        scheme="auto",
    )

    file_name_path = f"{output_dir}/{name}"
    assert os.path.isfile(file_name_path)

    # let us change the expected checksum to cause a failure
    # this will see this as not matching and will redownload,
    # but since the new file doesn't match it will raise an exception
    with pytest.raises(Exception):
        download_from_url(
            url,
            md5_checksum="checksum_garbage",
            output_path=output_dir,
            output_filename=name,
            force_download=True,
            scheme="auto",
        )


# we will just xfail this test in CI, as it will randomly be failing due to the servers we are contacted,
# not necessarily the code itself; we do not want the CI to fail overall, especially since we aren't really
# using this feature in any part of the code at this moment
@pytest.mark.xfail(
    IN_GITHUB_ACTIONS,
    reason="Network-dependent DOI/figshare resolution is flaky in GitHub Actions CI",
)
def test_fetch_record_id():
    record = fetch_url_from_doi(doi="10.6084/m9.figshare.4573048")
    assert (
        record
        == "https://figshare.com/articles/poster/SI2-SSE_Development_of_a_Software_Framework_for_Formalizing_ForceField_Atom-Typing_for_Molecular_Simulation/4573048"
    )

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.5281/zenodo.fake.3588339")

    with pytest.raises(Exception):
        fetch_url_from_doi(doi="10.6084/m9.figshare.4573048", timeout=0.0000000000001)


def test_md5_calculation(prep_temp_dir):
    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    checksum = "194cde222565dca8657d8521e5df1fd8"

    name = "atools_ml-v0.1.zip"
    download_from_url(
        url=url,
        md5_checksum=checksum,
        output_path=str(prep_temp_dir),
        output_filename=name,
        force_download=True,
        scheme="wget",
    )

    # explicit direct check of the function, even though included in download_from_zenodo
    calculated_checksum = calculate_md5_checksum(
        file_name=name, file_path=str(prep_temp_dir)
    )

    assert checksum == calculated_checksum
    bad_checksum = "294badmd5checksumthatwontwork9de"

    assert bad_checksum != calculated_checksum

    status = download_from_url(
        url=url,
        md5_checksum=bad_checksum,
        output_path=str(prep_temp_dir),
        output_filename=name,
        force_download=True,
        scheme="wget",
    )
    assert status == False


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Skipping; requires authentication which cannot be done via PR from fork ",
)
def test_load_from_wandb(prep_temp_dir):
    from modelforge.potential.potential import NeuralNetworkPotentialFactory

    nn_potential = NeuralNetworkPotentialFactory().load_from_wandb(
        run_path="modelforge_nnps/test_ANI2x_on_dataset/model-qloqn6gk",
        version="v0",
        local_cache_dir=f"{prep_temp_dir}/test_wandb",
        only_unique_pairs=True,
        old_config_only_local_cutoff=True,
    )

    assert os.path.isfile(f"{prep_temp_dir}/test_wandb/model.ckpt")

    assert nn_potential is not None


@pytest.mark.parametrize("scheme", ["auto", "curl", "wget"])
# define a test to ensure we can download using curl or wget in addition to requests
def test_download_with_various_schemes_zenodo(prep_temp_dir, scheme):

    url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    checksum = "194cde222565dca8657d8521e5df1fd8"

    output_dir = f"{str(prep_temp_dir)}/download_{scheme}"
    name = "atools_ml-v0.1.zip"
    # Download the file
    status = download_from_url(
        url,
        md5_checksum=checksum,
        output_path=output_dir,
        output_filename=name,
        force_download=True,
        scheme=scheme,
    )
    assert status == True

    file_name_path = f"{output_dir}/{name}"
    assert os.path.isfile(file_name_path)

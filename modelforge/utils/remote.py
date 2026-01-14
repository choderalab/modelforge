"""Module for querying remote sources and fetching datafiles"""

from typing import Optional, List, Dict
from loguru import logger
from sympy import true


def is_url(query: str, hostname: str) -> bool:
    """Validate if a string is a URL associated with a given domain.

    Parameters
    ----------
    query : str, required
        The string to check.
    hostname : str, required
        Check if the query URL domain includes the hostname.

    Returns
    -------
    bool
        If True, the string is a url where the domain
        includes the specified hostname.

    Examples
    -------
    >>> is_url(query="https://dx.doi.org/10.5281/zenodo.3588339", hostname="doi.org")
    """
    from urllib.parse import urlparse

    # parse the url using urllib parsing function
    parsed = urlparse(query)
    # if the string does not start with http
    # we will just return false
    if not "http" in parsed.scheme:
        return False
    # check to see if hostname is part of the url domain
    if not hostname in parsed.netloc:
        return False
    return True


def fetch_url_from_doi(doi: str, timeout: Optional[int] = 10) -> str:
    """Retrieve URL associated with a DOI.

    Parameters
    ----------
    doi : str, required
        The DOI to be considered.  This can be formatted as a URL.
    timeout : int, optional, default=10
        The number of seconds to wait to establish a connection

    Returns
    -------
    url : str
        The target URL linked to the DOI.

    Examples
    --------
    >>> fetch_url_from_doi(doi="10.5281/zenodo.3588339")
    """
    import requests

    # force to use ipv4; my ubuntu machine was timing out when it first tries ipv6
    # but that seemed to be a config issue on my machine and was resolved
    # will leave this import in here as well commented out, in case it is needed in the future
    # requests.packages.urllib3.util.connection.HAS_IPV6 = False

    doi_org_url = "https://dx.doi.org/"

    if is_url(doi, hostname="doi.org"):
        input_url = doi
    else:
        input_url = doi_org_url + doi

    try:
        response = requests.get(input_url, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        raise Exception("Fetching url for DOI timed out.")

    # if not response.ok:
    #     raise Exception(f"{doi} could not be accessed.")

    return response.url


def calculate_md5_checksum(file_name: str, file_path: str) -> str:
    import hashlib
    import os

    # make sure we can handle a path with a ~ in it

    if file_path is not None:
        file_path = os.path.expanduser(file_path)
        full_file_path = f"{file_path}/{file_name}"
    else:
        full_file_path = file_name

    from modelforge.utils.misc import OpenWithLock

    # we will use the OpenWithLock context manager to open the file
    # because we do not want to calculate the checksum if the file is still being written
    with OpenWithLock(f"{full_file_path}_checksum.lockfile", "w") as fl:
        with open(f"{full_file_path}", "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

    return file_hash.hexdigest()


def download_from_url(
    url: str,
    md5_checksum: str,
    output_path: str,
    output_filename: str,
    length: Optional[int] = None,
    force_download=False,
    max_retries: int = 30,
    retry_delay: List[int] = [1, 2, 5, 10, 20],  # in seconds
):
    """
    Download a file from a URL, with retries and checksum verification.

    Parameters
    ----------
    url : str, required
        The URL to download the file from.
    md5_checksum : str, required
        The expected MD5 checksum of the file.
    output_path : str, required
        The path to save the downloaded file to.
    output_filename : str, required
        The name to save the downloaded file as.
    length : int, optional, default=None
        The expected length of the file in bytes. If provided, this will be used to update the progress bar.
    force_download : bool, optional, default=False
        If True, the file will be re-downloaded even if it already exists and the checksum matches.
    max_retries : int, optional, default=10
        The maximum number of times to retry the download if it fails.
        Note, it will retry if the connection cannot be established, or if the checksum does not match (as can happen if a download is interrupted).
    retry_delay : List[int], optional, default=[1, 5, 5, 5, 10]
        The delay in seconds between retries. The length of this list should be equal to max_retries.
        If the length of this list is less than max_retries, the last value will be used for the remaining retries.

    Returns
    -------
    None

    Example
    -------
    >>> download_from_url(
    ...     url="https://zenodo.org/record/3588339/files/tmqm_openff_dataset_v1.2.hdf5",
    ...     md5_checksum="3b5c3f6e8e2f4c3e8e2f4c3e8e2f4c3",
    ...     output_path="~/data",
    ...     output_filename="tmqm_openff_dataset_v1.2.hdf5",
    ...     length=123456789,
    ...     force_download=False,
    ...     max_retries=3,
    ...     retry_delay=[1, 5, 10]
    ... )

    """

    import requests
    import os
    from tqdm import tqdm
    import time

    chunk_size = 512

    # if the length of the retry_delay list is less than max_retries, we will extend the list with the last value
    if len(retry_delay) < max_retries:
        retry_delay.extend([retry_delay[-1]] * (max_retries - len(retry_delay)))

    # make sure we can handle a path with a ~ in it
    output_path = os.path.expanduser(output_path)

    # if the output path doesn't exist create it
    os.makedirs(output_path, exist_ok=True)
    use_zenodo_token = False

    if "zenodo.org" in url.lower():

        ACCESS_TOKEN = os.environ.get("ZENODO_TOKEN")
        print(ACCESS_TOKEN)
        if ACCESS_TOKEN is None:
            logger.info("No Zenodo access token found in the environment variables.")
            logger.info(
                "It is recommended to generate and access token and set it as an environment variable, ZENODO_TOKEN "
            )
            logger.info("as this may prevent downloads from zenodo being rate limited.")
        else:
            logger.info("Using Zenodo access token")
            use_zenodo_token = True

    from modelforge.utils.misc import OpenWithLock

    with OpenWithLock(f"{output_path}/{output_filename}_download.lockfile", "w") as fl:

        if os.path.isfile(f"{output_path}/{output_filename}"):
            # if the file exists, we need to check to make sure that the file that is stored in the output path
            # note, we will check if the file has a lock on it inside calculate_md5_checksum to ensure
            # that we aren't looking at a file that is still being written to
            calculated_checksum = calculate_md5_checksum(
                file_name=output_filename, file_path=output_path
            )
            if calculated_checksum != md5_checksum:
                force_download = True
                logger.debug(
                    f"Checksum {calculated_checksum} of existing file {output_filename} does not match expected checksum {md5_checksum}, re-downloading."
                )

        if not os.path.isfile(f"{output_path}/{output_filename}") or force_download:
            logger.debug(
                f"Downloading datafile from {url} to {output_path}/{output_filename}."
            )
            for attempt in range(max_retries):
                try:
                    if use_zenodo_token:
                        r = requests.get(
                            url, stream=True, params={"access_token": ACCESS_TOKEN}
                        )
                    else:
                        r = requests.get(url, stream=True)

                    r.raise_for_status()

                    os.makedirs(output_path, exist_ok=True)
                    if length is not None:
                        total = int(length / chunk_size) + 1
                    else:
                        total = None

                    with open(f"{output_path}/{output_filename}", "wb") as fd:
                        for chunk in tqdm(
                            r.iter_content(chunk_size=chunk_size),
                            ascii=True,
                            desc="downloading",
                            total=total,
                        ):
                            fd.write(chunk)

                    calculated_checksum = calculate_md5_checksum(
                        file_name=output_filename, file_path=output_path
                    )
                    if calculated_checksum != md5_checksum:
                        raise Exception(
                            f"Checksum of downloaded file {calculated_checksum} does not match expected checksum {md5_checksum}."
                        )
                    else:
                        return  # success, exit the function
                except requests.exceptions.RequestException as e:
                    if attempt < (max_retries - 1):
                        logger.debug(
                            f"Attempt {attempt+1} to download file from {url} failed, retrying."
                        )
                        time.sleep(retry_delay[attempt])
                    else:
                        logger.debug(
                            f"Attempt {attempt+1} to download file from {url} failed, no more retries left."
                        )
                        raise Exception(
                            f"Could not download file from {url} after {max_retries} attempts."
                        ) from e

        else:  # if the file exists and we don't set force_download to True, just use the cached version
            logger.debug(f"Datafile {output_filename} already exists in {output_path}.")
            logger.debug(
                "Using previously downloaded file; set force_download=True to re-download."
            )

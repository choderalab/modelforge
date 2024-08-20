"""Module for querying remote sources and fetching datafiles"""

from typing import Optional, List, Dict
from loguru import logger


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

    if not response.ok:
        raise Exception(f"{doi} could not be accessed.")

    return response.url


def calculate_md5_checksum(file_name: str, file_path: str) -> str:
    import hashlib
    import os

    # make sure we can handle a path with a ~ in it
    file_path = os.path.expanduser(file_path)

    from modelforge.utils.misc import OpenWithLock

    # we will use the OpenWithLock context manager to open the file
    # because we do not want to calculate the checksum if the file is still being written
    with OpenWithLock(f"{file_path}/{file_name}.lockfile", "w") as fl:
        with open(f"{file_path}/{file_name}", "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

    os.remove(f"{file_path}/{file_name}.lockfile")

    return file_hash.hexdigest()


def download_from_url(
    url: str,
    md5_checksum: str,
    output_path: str,
    output_filename: str,
    length: Optional[int] = None,
    force_download=False,
):

    import requests
    import os
    from tqdm import tqdm

    chunk_size = 512

    # make sure we can handle a path with a ~ in it
    output_path = os.path.expanduser(output_path)

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

        r = requests.get(url, stream=True)

        os.makedirs(output_path, exist_ok=True)
        if length is not None:
            total = int(length / chunk_size) + 1
        else:
            total = None

        from modelforge.utils.misc import OpenWithLock

        with OpenWithLock(f"{output_path}/{output_filename}.lockfile", "w") as fl:
            with open(f"{output_path}/{output_filename}", "wb") as fd:
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    ascii=True,
                    desc="downloading",
                    total=total,
                ):
                    fd.write(chunk)
        os.remove(f"{output_path}/{output_filename}.lockfile")
        calculated_checksum = calculate_md5_checksum(
            file_name=output_filename, file_path=output_path
        )
        if calculated_checksum != md5_checksum:
            raise Exception(
                f"Checksum of downloaded file {calculated_checksum} does not match expected checksum {md5_checksum}."
            )

    else:  # if the file exists and we don't set force_download to True, just use the cached version
        logger.debug(f"Datafile {output_filename} already exists in {output_path}.")
        logger.debug(
            "Using previously downloaded file; set force_download=True to re-download."
        )



"""Module for querying and fetching datafiles from remote sources"""

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

    # force to use ipv4; my ubuntu machine is timing out when it first tries ipv6
    #requests.packages.urllib3.util.connection.HAS_IPV6 = False

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

    with open(f"{file_path}/{file_name}", "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()


# Figshare helper functions
def download_from_figshare(url: str, output_path: str, force_download=False) -> str:
    """
    Downloads a dataset from figshare for a given ndownloader url.

    Parameters
    ----------
    url: str, required
        Figshare ndownloader url (i.e., link to the data downloader)
    output_path: str, required
        Location to download the file to.
    force_download: str, default=False
        If False: if the file exists in output_path, code will will use the local version.
        If True, the file will be downloaded, even if it exists in output_path.

    Returns
    -------
    str
        Name of the file downloaded.

    Examples
    --------
    >>> url = 'https://springernature.figshare.com/ndownloader/files/18112775'
    >>> output_path = '/path/to/directory'
    >>> downloaded_file_name = download_from_figshare(url, output_path)

    """

    import requests
    import os
    from tqdm import tqdm

    # force to use ipv4; my ubuntu machine is timing out when it first tries ipv6
    #requests.packages.urllib3.util.connection.HAS_IPV6 = False

    chunk_size = 512
    # check to make sure the url we are given is hosted by figshare.com
    if not is_url(url, "figshare.com"):
        raise Exception(f"{url} is not a valid figshare.com url")

    # get the head of the request
    head = requests.head(url)

    # Because the url on figshare calls a downloader, instead of the direct file,
    # we need to figure out where the original file is stored to know how big it is.
    # Here we will parse the header info to get the file the downloader links to
    # and then get the head info from this link to fetch the length.
    # This is not actually necessary, but useful for updating the download status bar.
    # We also fetch the name of the file from the header of the download link
    temp_url = head.headers["location"].split("?")[0]
    name = head.headers["X-Filename"].split("/")[-1]

    logger.debug(f"Downloading datafile from figshare to {output_path}/{name}.")

    if not os.path.isfile(f"{output_path}/{name}") or force_download:
        temp_url_headers = requests.head(temp_url)
        length = int(temp_url_headers.headers["Content-Length"])
        figshare_md5_checksum = temp_url_headers.headers["ETag"].strip('"')
        r = requests.get(url, stream=True)

        os.makedirs(output_path, exist_ok=True)

        with open(f"{output_path}/{name}", "wb") as fd:
            for chunk in tqdm(
                r.iter_content(chunk_size=chunk_size),
                ascii=True,
                desc="downloading",
                total=(int(length / chunk_size) + 1),
            ):
                fd.write(chunk)

        calculated_checksum = calculate_md5_checksum(
            file_name=name, file_path=output_path
        )
        if calculated_checksum != figshare_md5_checksum:
            raise Exception(
                "Checksum of downloaded file does not match expected checksum"
            )
    else:  # if the file exists and we don't set force_download to True, just use the cached version
        logger.debug(f"Datafile {name} already exists in {output_path}.")
        logger.debug(
            "Using previously downloaded file; set force_download=True to re-download."
        )

    return name


def download_from_zenodo(
    url: str, zenodo_md5_checksum: str, output_path: str, force_download=False
) -> str:
    """
    Downloads a dataset from zenodo for a given url.

    If the datafile exists in the output_path, by default it will not be redownloaded.

    Parameters
    ----------
    url : str, required
        Direct link to datafile to download.
    output_path: str, required
        Location to download the file to.
    force_download: str, default=False
        If False: if the file exists in output_path, code will will use the local version.
        If True, the file will be downloaded, even if it exists in output_path.

    Returns
    -------
    str
        Name of the file downloaded.

    Examples
    --------
    >>> url = "https://zenodo.org/records/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    >>> output_path = '/path/to/directory'
    >>> downloaded_file_name = download_from_zenodo(url, output_path)

    """

    import requests
    import os
    from tqdm import tqdm

    # force to use ipv4; my ubuntu machine is timing out when it first tries ipv6
    #requests.packages.urllib3.util.connection.HAS_IPV6 = False

    chunk_size = 512
    # check to make sure the url we are given is hosted by figshare.com

    if not is_url(url, "zenodo.org"):
        raise Exception(f"{url} is not a valid zenodo.org url")

    # get the head of the request
    head = requests.head(url)

    # Because the url on figshare calls a downloader, instead of the direct file,
    # we need to figure out where the original file is stored to know how big it is.
    # Here we will parse the header info to get the file the downloader links to
    # and then get the head info from this link to fetch the length.
    # This is not actually necessary, but useful for updating the download status bar.
    # We also fetch the name of the file from the header of the download link
    name = head.headers["Content-Disposition"].split("filename=")[-1]
    length = int(head.headers["Content-Length"])

    if not os.path.isfile(f"{output_path}/{name}") or force_download:
        logger.debug(f"Downloading datafile from zenodo to {output_path}/{name}.")

        r = requests.get(url, stream=True)

        os.makedirs(output_path, exist_ok=True)

        with open(f"{output_path}/{name}", "wb") as fd:
            for chunk in tqdm(
                r.iter_content(chunk_size=chunk_size),
                ascii=True,
                desc="downloading",
                total=(int(length / chunk_size) + 1),
            ):
                fd.write(chunk)
        calculated_checksum = calculate_md5_checksum(
            file_name=name, file_path=output_path
        )
        if calculated_checksum != zenodo_md5_checksum:
            raise Exception(
                "Checksum of downloaded file does not match expected checksum"
            )

    else:  # if the file exists and we don't set force_download to True, just use the cached version
        logger.debug(f"Datafile {name} already exists in {output_path}.")
        logger.debug(
            "Using previously downloaded file; set force_download=True to re-download."
        )

    return name

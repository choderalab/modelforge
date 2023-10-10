"""Module for querying and fetching datafiles from remote sources"""

from typing import Optional, List
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
        length = int(requests.head(temp_url).headers["Content-Length"])

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
    else:  # if the file exists and we don't set force_download to True, just use the cached version
        logger.debug(f"Datafile {name} already exists in {output_path}.")
        logger.debug(
            "Using previously downloaded file; set force_download=True to re-download."
        )

    return name


# Zenodo specific helper functions
def parse_zenodo_record_id_from_url(url: str) -> str:
    """Return the record id from a zenodo.org record URL.

    This function will raise an exception if the URL
    is malformed. Expected format:
    https://zenodo.org/record/{RECORD_ID}

    Parameters
    ----------
    url : str, required
        The url to parse.
    Returns
    -------
    record_id : str
        Zenodo record id.

    Examples
    --------
    >>> parse_zenodo_record_id_from_url(url = "https://zenodo.org/record/3588339")
    """
    from urllib.parse import urlparse

    # parse the url using urllib parsing function
    parsed = urlparse(url)

    parsed_path = list(filter(None, parsed.path.split("/")))

    # make sure that we only have two elements in the path
    # part of the url, namely ['record', f'{record_id}']
    if len(parsed_path) != 2:
        raise Exception(f"Malformed zenodo.org record URL: {url}.")
    record_id = parsed_path[-1]

    return record_id


def get_zenodo_datafiles(
    record_id: str, file_extension: str, timeout: Optional[int] = 10
) -> List[str]:
    """Retrieve link(s) to datafiles on zenodo with a given extension.

    Parameters
    ----------
    record_id : str, required
        zenodo.org record id.  Can also provide url to a record.
    file_extension : str, required
        Return file(s) with extensions that match file_extension
    timeout : int, optional, default=10
        The number of seconds to wait to establish a connection

    Returns
    -------
    data_urls : list object, dtype=str
        Each entry in the list links to files with the given file extension.

    Examples
    --------
    >>> files = get_zenodo_datafiles(record_id="3588339", file_extension=".hdf5.gz")
    >>> files = get_zenodo_datafiles(
        record_id="https://zenodo.org/record/3588339", file_extension=".hdf5.gz"
    )
    """
    import requests

    zenodo_base = "https://zenodo.org/api/records/"

    # if we are provided the url, santize
    if is_url(record_id, "zenodo.org"):
        record_id = parse_zenodo_record_id_from_url(record_id)

    zenodo_api_url = zenodo_base + record_id

    try:
        data_request = requests.get(zenodo_api_url, timeout=timeout)
    except requests.exceptions.ConnectTimeout:
        raise Exception("Attempt to access Zenodo timed out")

    if not data_request.ok:
        raise Exception(f"Record id {record_id} could not be accessed.")

    # grab the data from zenodo
    json_content = data_request.json()
    files = json_content["files"]

    # search through the list of files to find those with desired extension
    data_urls = []
    for file in files:
        if file["links"]["self"].endswith(file_extension):
            data_urls.append(file["links"]["self"])

    return data_urls


def datafiles_from_zenodo(record: str, file_extension: str) -> List[str]:
    """For a given zenodo DOI or record_id, return links to all gzipped hdf5 files.

    Parameters
    ----------
    record : str, required
        This can be either Zenodo DOI or Zenodo record id.
        Either of these can be formatted as a URL, e.g.,
        https://dx.doi.org/{DOI} or https://zenodo.org/record/{record_id}
    file_extension: str, required
        File extension for filtering results
    Returns
    -------
    data_urls : list object, dtype=str
        Each entry contains the direct link to all hdf5.gz files.

    Examples
    --------
    >>> files = hdf5_from_zenodo(record="10.5281/zenodo.3588339")
    >>> files = hdf5_from_zenodo(record="https://dx.doi.org/10.5281/zenodo.3588339")
    >>> files = hdf5_from_zenodo(record="https://zenodo.org/record/3588339")
    >>> files = hdf5_from_zenodo(record="3588339")
    """
    record_is_doi = True
    # first determine if we are dealing with a doi or a record_id
    if is_url(record, hostname="zenodo.org"):
        record_is_doi = False
    elif is_url(record, hostname="doi.org"):
        record_is_doi = True
    elif not "zenodo." in record.split("/")[-1]:
        record_is_doi = False

    if record_is_doi:
        record_id = fetch_url_from_doi(record)
        data_urls = get_zenodo_datafiles(record_id, file_extension=file_extension)
    else:
        data_urls = get_zenodo_datafiles(record, file_extension=file_extension)

    # Make sure files were found.
    if len(data_urls) == 0:
        raise Exception(f"No files with extension {file_extension} were found.")

    return data_urls


def download_from_zenodo(url: str, output_path: str, force_download=False) -> str:
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
    >>> url = "https://zenodo.org/record/3401581/files/PTC-CMC/atools_ml-v0.1.zip"
    >>> output_path = '/path/to/directory'
    >>> downloaded_file_name = download_from_zenodo(url, output_path)

    """

    import requests
    import os
    from tqdm import tqdm

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
    else:  # if the file exists and we don't set force_download to True, just use the cached version
        logger.debug(f"Datafile {name} already exists in {output_path}.")
        logger.debug(
            "Using previously downloaded file; set force_download=True to re-download."
        )

    return name

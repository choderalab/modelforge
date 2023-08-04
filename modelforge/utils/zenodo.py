"""Module for fetching datafiles from zenodo"""

import requests

from urllib.parse import urlparse
from typing import Optional


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
    """

    # parse the url
    parsed = urlparse(query)
    # if the string does not start with http
    # we will just return false
    if not "http" in parsed.scheme:
        return False
    # check to see if hostname is part of the url domain
    if not hostname in parsed.netloc:
        return False
    return True


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
    """
    parsed = urlparse(url)

    parsed_path = list(filter(None, parsed.path.split("/")))

    # make sure that we only have two elements in the path
    # part of the url, namely ['record', f'{record_id}']
    if len(parsed_path) != 2:
        raise Exception(f"Malformed zenodo.org record URL: {url}.")
    record_id = parsed_path[-1]

    return record_id


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
    """

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


def get_zenodo_datafiles(
    record_id: str, file_extension: str, timeout: Optional[int] = 10
) -> str:
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
    data_urls : list-like object, dtype=str
        Direct links to files with the given file extension.
    """

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


def hdf5_from_zenodo(record: str) -> str:
    """For a given zenodo DOI or record_id, return links to all hdf5 files.

    Parameters
    ----------
    record : str, required
        This can be either Zenodo DOI or Zenodo record id.
        Either of these can be formatted as a URL, e.g.,
        https://dx.doi.org/{DOI} or https://zenodo.org/record/{record_id}
        Note: this assumes files on Zenodo are gzipped (i.e., extension hdf5.gz).

    Returns
    -------
    data_urls : list-like, dtype=str
        Direct link to gzipped hdf5 files.
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
        data_urls = get_zenodo_datafiles(record_id, file_extension="hdf5.gz")
    else:
        data_urls = get_zenodo_datafiles(record, file_extension="hdf5.gz")

    # Make sure files were found.
    if len(data_urls) == 0:
        raise Exception(f"No files with extension hdf5.gz were found.")

    return data_urls

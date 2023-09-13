import os
from loguru import logger


def dict_to_hdf5(file_name: str, data: list, series_info: dict, id_key: str) -> None:
    """
    Writes an hdf5 file from a list of dicts.

    This will include units, if provided as attributes.

    Parameters
    ----------
    file_name: str, required
        Name and path of hdf5 file to write.
    data: list of dicts, required
        List that contains dictionaries of properties for each molecule to write to file.
    id_key: str, required
        Name of the key in the dicts that uniquely describes each record.

    Examples
    --------
    >>> dict_to_hdf5(file_name='qm9.hdf5', data=data, series_info=series, id_key='name')
    """

    import h5py
    from tqdm import tqdm
    import numpy as np
    import pint
    from openff.units import unit, Quantity

    assert file_name.endswith(".hdf5")

    dt = h5py.special_dtype(vlen=str)

    with h5py.File(file_name, "w") as f:
        for datapoint in tqdm(data):
            try:
                record_name = datapoint[id_key]
            except Exception:
                print(f"id_key {id_key} not found in the data.")
            group = f.create_group(record_name)
            for key, val in datapoint.items():
                if key != id_key:
                    if isinstance(val, pint.Quantity):
                        val_m = val.m
                        val_u = str(val.u)
                    else:
                        val_m = val
                        val_u = None
                    if isinstance(val_m, str):
                        group.create_dataset(name=key, data=val_m, dtype=dt)
                    elif isinstance(val_m, (float, int)):
                        group.create_dataset(name=key, data=val_m)
                    elif isinstance(val_m, np.ndarray):
                        group.create_dataset(name=key, data=val_m, shape=val_m.shape)
                    if not val_u is None:
                        group[key].attrs["u"] = val_u
                    if series_info[key] == "series":
                        group[key].attrs["series"] = True
                    else:
                        group[key].attrs["series"] = False


def mkdir(path: str) -> bool:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            return True
        except Exception:
            print("Could not create directory {path}.")
    else:
        return False


def download_from_figshare(url: str, output_path: str, force_download=False) -> str:
    """
    Downloads a dataset from figshare.

    Parameters
    ----------
    ndownloader_url: str, required
        Figshare url to the data downloader
    output_path: str, required
        Location to download the file to.
    force_download: str, default=False
        If False: if the file does not exist in output_path it will use the local version.
        If True, the file will be downloaded even if it exists in output_path.

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
    from tqdm import tqdm

    chunk_size = 512

    # get the head of the request
    head = requests.head(url)
    # Because the url on figshare calls a downloader, instead of the direct file,
    # we need to figure out where the original file is to know how big it is.
    # Here we will parse the header info to get the file the downloader links to
    # and then get the head info from this link to fetch the length.
    # This is not actually necessary, but useful for updating download status bar.
    # We also fetch the name of the file from the header of the download link
    temp_url = head.headers["location"].split("?")[0]
    name = head.headers["X-Filename"].split("/")[-1]

    logger.debug(f"Downloading datafile from figshare to {output_path}/{name}.")

    if not os.path.isfile(f"{output_path}/{name}") or force_download:
        length = int(requests.head(temp_url).headers["Content-Length"])

        r = requests.get(url, stream=True)

        mkdir(output_path)

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
            "Using already downloaded file; use force_download=True to re-download."
        )

    return name


def list_files(directory: str, extension: str) -> list:
    """
    Returns a list of files in a directory with a given extension.

    Parameters
    ----------
    directory: str, required
        Directory of interest.
    extension: str, required
        Only consider files with this given file extension

    Returns
    -------
    list
        List of files in the given directory with desired extension.

    Examples
    --------
    List only the xyz files in a test_directory
    >>> files = list_files('test_directory', '.xyz')
    """

    logger.debug(f"Gathering {extension} files in {directory}.")

    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files.append(file)
    files.sort()
    return files


def str_to_float(x: str) -> float:
    """
    Converts a string to a float, changing Mathematica style scientific notion to python style.

    For example, this will convert str(1*^-6) to float(1e-6).

    Parameters
    ----------
    x : str, required
        String to process.

    Returns
    -------
    float
        Float value of the string.
    """
    xf = float(x.replace("*^", "e"))
    return xf

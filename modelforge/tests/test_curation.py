import os
import sys
from pathlib import Path

import numpy as np
import pytest
from openff.units import unit, Quantity
import pint
import importlib_resources as resources

from modelforge.curation.utils import *

from modelforge.curation.qm9_curation import *


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("hdf5_data")
    return fn

    # generate test data into a temporary path


def test_dict_to_hdf5(prep_temp_dir):
    # generate an hdf5 file from simple test data
    # then read it in and see that we can reconstruct the same data
    file_path = str(prep_temp_dir)
    test_data = [
        {
            "name": "test1",
            "energy": 123 * unit.hartree,
            "geometry": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) * unit.angstrom,
        },
        {
            "name": "test2",
            "energy": 456 * unit.hartree,
            "geometry": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) * unit.angstrom,
        },
    ]
    file_name_path = file_path + "/test.hdf5"
    dict_to_hdf5(file_name=file_name_path, data=test_data, id_key="name")

    # check we wrote the file
    assert os.path.isfile(file_name_path)

    # read in the hdf5 file
    records = []
    with h5py.File(file_name_path, "r") as hf:
        test_names = list(hf.keys())

        # validate names
        assert test_names == ["test1", "test2"]

        for name in test_names:
            temp_record = {}
            temp_record["name"] = name
            properties = list(hf[name].keys())

            for property in properties:
                # validate properties name
                assert property in ["energy", "geometry"]
                if "units" in hf[name][property].attrs:
                    u = hf[name][property].attrs["units"]
                    temp_record[property] = hf[name][property][
                        ()
                    ] * unit.parse_expression(u)

                else:
                    temp_record[property] = hf[name][property][()]

            records.append(temp_record)

    # loop over reconstructed list of dictionaries and compare contents
    for i in range(len(records)):
        for key in test_data[i].keys():
            if isinstance(records[i][key], pint.Quantity):
                record_m = records[i][key].m
                test_data_m = test_data[i][key].m

                if isinstance(record_m, np.ndarray):
                    print(record_m, test_data_m)
                    assert np.all(record_m == test_data_m)
                else:
                    assert record_m == test_data_m
            else:
                assert records[i][key] == test_data[i][key]


def test_qm9_curation_helper_functions(prep_temp_dir):
    qm9_data = QM9_curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_path=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )

    val = qm9_data._str_to_float("1*^6")
    assert val == 1e6

    # check the function to list directory contents
    files = qm9_data._list_files(str(prep_temp_dir), ".hdf5")

    # check to see if test.hdf5 is in the files
    assert "test.hdf5" in files


def test_qm9_curation_parse_xyz(prep_temp_dir):
    qm9_data = QM9_curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_path=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )

    # check to ensure we can parse the properties line correctly
    # This is data is modified from dsgdb9nsd_000001.xyz, with floats truncated to one decimal place
    temp_line = "gdb 1  157.7  157.7  157.7  0.  13.2  -0.3  0.1  0.5  35.3  0.0  -40.4  -40.4  -40.4  -40.4  6.4"
    temp_dict = qm9_data._parse_properties(temp_line)

    assert len(temp_dict) == 17
    assert temp_dict["tag"] == "gdb"
    assert temp_dict["idx"] == "1"
    assert temp_dict["rotational constant A"] == 157.7 * unit.gigahertz
    assert temp_dict["rotational constant B"] == 157.7 * unit.gigahertz
    assert temp_dict["rotational constant C"] == 157.7 * unit.gigahertz
    assert temp_dict["dipole moment"] == 0 * unit.debye
    assert temp_dict["isotropic polarizability"] == 13.2 * unit.angstrom**3
    assert temp_dict["energy of homo"] == -0.3 * unit.hartree
    assert temp_dict["energy of lumo"] == 0.1 * unit.hartree
    assert temp_dict["gap"] == 0.5 * unit.hartree
    assert temp_dict["electronic spatial extent"] == 35.3 * unit.angstrom**2
    assert temp_dict["zero point vibrational energy"] == 0.0 * unit.hartree
    assert temp_dict["internal energy at 0K"] == -40.4 * unit.hartree
    assert temp_dict["internal energy at 298.15K"] == -40.4 * unit.hartree
    assert temp_dict["enthalpy at 298.15K"] == -40.4 * unit.hartree
    assert temp_dict["free energy at 298.15K"] == -40.4 * unit.hartree
    assert (
        temp_dict["heat capacity at 298.15K"]
        == 6.4 * unit.calorie_per_mole / unit.kelvin
    )

    fn = resources.files("modelforge").joinpath("tests", "data", "dsgdb9nsd_000001.xyz")
    data_dict_temp = qm9_data._parse_xyzfile(str(fn))

    # spot check values
    assert np.all(
        np.isclose(
            data_dict_temp["geometry"],
            np.array(
                [
                    [-1.26981359e-03, 1.08580416e-01, 8.00099580e-04],
                    [2.15041600e-04, -6.03131760e-04, 1.97612040e-04],
                    [1.01173084e-01, 1.46375116e-01, 2.76574800e-05],
                    [-5.40815069e-02, 1.44752661e-01, -8.76643715e-02],
                    [-5.23813634e-02, 1.43793264e-01, 9.06397294e-02],
                ]
            )
            * unit.nanometer,
        )
    )

    assert np.all(
        data_dict_temp["charges"]
        == np.array([-0.535689, 0.133921, 0.133922, 0.133923, 0.133923])
        * unit.elementary_charge
    )
    assert data_dict_temp["isotropic polarizability"] == 13.21 * unit.angstroms**3
    assert (
        data_dict_temp["energy of homo"]
        == -1017.9062102263447 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["energy of lumo"] == 307.4460077830925 * unit.kilojoule_per_mole
    )
    assert data_dict_temp["gap"] == 1325.3522180094374 * unit.kilojoule_per_mole
    assert data_dict_temp["electronic spatial extent"] == 35.3641 * unit.angstrom**2
    assert (
        data_dict_temp["zero point vibrational energy"]
        == 117.4884833670846 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["internal energy at 0K"]
        == -106277.4161215308 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["internal energy at 298.15K"]
        == -106269.88618856476 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["enthalpy at 298.15K"]
        == -106267.40509140545 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["free energy at 298.15K"]
        == -106329.05182294044 * unit.kilojoule_per_mole
    )
    assert (
        data_dict_temp["heat capacity at 298.15K"]
        == 0.027066296000000004 * unit.kilojoule_per_mole / unit.kelvin
    )
    assert np.all(data_dict_temp["atomic numbers"] == np.array([6, 1, 1, 1, 1]))
    assert data_dict_temp["smiles gdb-17"] == "C"
    assert data_dict_temp["smiles b3lyp"] == "C"
    assert data_dict_temp["inchi Corina"] == "1S/CH4/h1H4"
    assert data_dict_temp["inchi B3LYP"] == "1S/CH4/h1H4"
    assert data_dict_temp["rotational constant A"] == 157.7118 * unit.gigahertz
    assert data_dict_temp["rotational constant B"] == 157.70997 * unit.gigahertz
    assert data_dict_temp["rotational constant C"] == 157.70699 * unit.gigahertz
    assert data_dict_temp["dipole moment"] == 0.0 * unit.debye
    assert np.all(
        data_dict_temp["harmonic vibrational frequencies"]
        == np.array(
            [
                1341.307,
                1341.3284,
                1341.365,
                1562.6731,
                1562.7453,
                3038.3205,
                3151.6034,
                3151.6788,
                3151.7078,
            ]
        )
        / unit.centimeter
    )


def test_qm9_local_archive(prep_temp_dir):
    # test file extraction, parsing, and generation of hdf5 file
    # from a local archive.
    qm9_data = QM9_curation(
        hdf5_file_name="qm9_test10.hdf5",
        output_file_path=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )

    fn = resources.files("modelforge").joinpath("tests", "data")

    qm9_data._process_downloaded(str(fn), "first10.tar.bz2", unit_testing=True)

    assert len(qm9_data.data) == 10

    file_name_path = str(fn) + "/first10.tar.bz2"
    assert os.path.isfile(file_name_path)

    names = {
        "dsgdb9nsd_000001": -106277.4161215308,
        "dsgdb9nsd_000002": -148408.69593977975,
        "dsgdb9nsd_000003": -200600.51755556674,
        "dsgdb9nsd_000004": -202973.24721725564,
        "dsgdb9nsd_000005": -245252.87826713378,
        "dsgdb9nsd_000006": -300576.6846578527,
        "dsgdb9nsd_000007": -209420.75231941737,
        "dsgdb9nsd_000008": -303715.5298633426,
        "dsgdb9nsd_000009": -306158.32885940996,
        "dsgdb9nsd_000010": -348451.454977435,
    }
    file_name_path = str(prep_temp_dir) + "/qm9_test10.hdf5"

    with h5py.File(file_name_path, "r") as hf:
        for key in hf.keys():
            # check record names
            assert key in list(names.keys())
            assert np.isclose(hf[key]["internal energy at 0K"][()], names[key])


def test_qm9_download(prep_temp_dir):
    qm9_data = QM9_curation(
        hdf5_file_name="qm9_test10.hdf5",
        output_file_path=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )
    name = qm9_data.dataset_description["dataset_filename"]
    url = qm9_data.dataset_description["dataset_download_url"]

    qm9_data._download(
        url=url,
        name=name,
        output_path=str(prep_temp_dir),
        force_download=False,
    )

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)


def test_qm9_curation(prep_temp_dir):
    # test file download and extraction
    # this downloads the entire archive and extracts it
    # but only processes the first 10 records
    qm9_data = QM9_curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_path=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )

    # test all the functions will run
    qm9_data.process(unit_testing=True)

    name = qm9_data.dataset_description["dataset_filename"]

    file_name_path = str(prep_temp_dir) + f"/{name}"
    assert os.path.isfile(file_name_path)

    # ensure we processed 10 records
    assert len(qm9_data.data) == 10

    file_name_path = str(prep_temp_dir) + "/qm9_dataset.hdf5"
    assert os.path.isfile(file_name_path)

    names = {
        "dsgdb9nsd_000001": -106277.4161215308,
        "dsgdb9nsd_000002": -148408.69593977975,
        "dsgdb9nsd_000003": -200600.51755556674,
        "dsgdb9nsd_000004": -202973.24721725564,
        "dsgdb9nsd_000005": -245252.87826713378,
        "dsgdb9nsd_000006": -300576.6846578527,
        "dsgdb9nsd_000007": -209420.75231941737,
        "dsgdb9nsd_000008": -303715.5298633426,
        "dsgdb9nsd_000009": -306158.32885940996,
        "dsgdb9nsd_000010": -348451.454977435,
    }

    with h5py.File(file_name_path, "r") as hf:
        for key in hf.keys():
            # check record names
            assert key in list(names.keys())
            assert np.isclose(hf[key]["internal energy at 0K"][()], names[key])

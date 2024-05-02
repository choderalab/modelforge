import h5py
import os
import pytest
from openff.units import unit
import importlib_resources as resources
import numpy as np
from modelforge.utils.misc import *
import pint

from modelforge.curation.qm9_curation import QM9Curation
from modelforge.curation.ani1x_curation import ANI1xCuration
from modelforge.curation.spice_114_curation import SPICE114Curation
from modelforge.curation.spice_openff_curation import SPICEOpenFFCuration
from modelforge.curation.spice_2_from_qcarchive_curation import SPICE2Curation

from modelforge.curation.curation_baseclass import dict_to_hdf5


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("hdf5_data")
    return fn


def test_dict_to_hdf5(prep_temp_dir):
    # generate an hdf5 file from simple test data
    # then read it in and see that we can reconstruct the same data
    file_path = str(prep_temp_dir)
    record_entries_series = {
        "name": "single",
        "n_configs": "single",
        "energy": "single",
        "geometry": "single",
    }
    test_data = [
        {
            "name": "test1",
            "n_configs": 1,
            "energy": 123 * unit.hartree,
            "geometry": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) * unit.angstrom,
        },
        {
            "name": "test2",
            "n_configs": 1,
            "energy": 456 * unit.hartree,
            "geometry": np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]) * unit.angstrom,
        },
    ]
    file_name_path = file_path + "/test.hdf5"
    dict_to_hdf5(
        file_name=file_name_path,
        data=test_data,
        series_info=record_entries_series,
        id_key="name",
    )

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
                assert property in ["n_configs", "energy", "geometry"]

                if "u" in hf[name][property].attrs:
                    u = hf[name][property].attrs["u"]
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

    with pytest.raises(Exception):
        dict_to_hdf5(
            file_name=file_name_path,
            data=test_data,
            series_info=record_entries_series,
            id_key="name_should_fail",
        )

    # test to see if we can catch a ValueError for an int64 (from np.sum) rather than int
    test_data = [
        {
            "name": "test1",
            "n_configs": np.sum([1, 2]),
            "energy": 123 * unit.hartree,
            "geometry": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) * unit.angstrom,
        }
    ]

    with pytest.raises(ValueError):
        dict_to_hdf5(
            file_name=file_name_path,
            data=test_data,
            series_info=record_entries_series,
            id_key="name",
        )


def test_series_dict_to_hdf5(prep_temp_dir):
    # generate an hdf5 file from simple test data
    # then read it in and see that we can reconstruct the same data
    # here this will test defining if an attribute is part of a series

    file_path = str(prep_temp_dir)
    record_entries_series = {
        "name": "single_rec",
        "n_configs": "single_rec",
        "energy": "series_mol",
        "geometry": "series_atom",
    }
    test_data = [
        {
            "name": "test1",
            "n_configs": 2,
            "energy": np.array([123, 234]).reshape(2, 1) * unit.hartree,
            "geometry": np.array(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
            )
            * unit.angstrom,
        },
    ]
    file_name_path = file_path + "/test_series.hdf5"
    dict_to_hdf5(
        file_name=file_name_path,
        data=test_data,
        series_info=record_entries_series,
        id_key="name",
    )

    # check we wrote the file
    assert os.path.isfile(file_name_path)

    # read in the hdf5 file
    records = []
    with h5py.File(file_name_path, "r") as hf:
        test_names = list(hf.keys())

        # validate names
        assert test_names == ["test1"]

        for name in test_names:
            temp_record = {}
            temp_record["name"] = name
            properties = list(hf[name].keys())

            for property in properties:
                # validate properties name
                assert property in ["n_configs", "energy", "geometry"]

            n_configs = hf[name]["n_configs"][()]
            temp_record["n_configs"] = n_configs

            for property in ["energy", "geometry"]:
                format = hf[name][property].attrs["format"]
                if format.split("_")[0] == "series":
                    temp = []
                    for i in range(n_configs):
                        temp.append(hf[name][property][i])

                    if "u" in hf[name][property].attrs:
                        u = hf[name][property].attrs["u"]
                        temp_record[property] = np.array(temp) * unit.parse_expression(
                            u
                        )
                    else:
                        temp_record[property] = np.array(temp)

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


def test_list_files(prep_temp_dir):
    # test.hdf5 was generated in test_dict_to_hdf5
    files = list_files(str(prep_temp_dir), ".hdf5")

    # check to see if test.hdf5 is in the files
    assert "test.hdf5" in files

    with pytest.raises(Exception):
        list_files("/path/that/should/not/exist/", ".hdf5")


def test_str_to_float(prep_temp_dir):
    val = str_to_float("1*^6")
    assert val == 1e6

    val = str_to_float("100")
    assert val == 100


def test_qm9_curation_init_parameters(prep_temp_dir):
    qm9_data = QM9Curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=False,
    )

    assert qm9_data.hdf5_file_name == "qm9_dataset.hdf5"
    assert qm9_data.output_file_dir == str(prep_temp_dir)
    assert qm9_data.local_cache_dir == str(prep_temp_dir)
    assert qm9_data.convert_units == False


def test_qm9_reference_energy(prep_temp_dir):
    qm9_data = QM9Curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )
    assert (
        qm9_data._calculate_reference_thermochemistry(["C", "H", "H", "H", "H"], "U_0K")
        == -39.847864 * unit.hartree
    )


def test_qm9_curation_parse_xyz(prep_temp_dir):
    qm9_data = QM9Curation(
        hdf5_file_name="qm9_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )

    # check to ensure we can parse the properties line correctly
    # This is data is modified from dsgdb9nsd_000001.xyz, with floats truncated to one decimal place
    temp_line = "gdb 1  157.7  157.7  157.7  0.  13.2  -0.3  0.1  0.5  35.3  0.0  -40.4  -40.4  -40.4  -40.4  6.4"
    temp_dict = qm9_data._parse_properties(temp_line)

    # units are applied in _parse_properties
    assert len(temp_dict) == 17
    assert temp_dict["tag"] == "gdb"
    assert temp_dict["idx"] == "1"
    assert temp_dict["rotational_constant_A"] == 157.7
    assert temp_dict["rotational_constant_B"] == 157.7
    assert temp_dict["rotational_constant_C"] == 157.7
    assert temp_dict["dipole_moment"] == 0
    assert temp_dict["isotropic_polarizability"] == 13.2
    assert temp_dict["energy_of_homo"] == -0.3
    assert temp_dict["energy_of_lumo"] == 0.1
    assert temp_dict["lumo-homo_gap"] == 0.5
    assert temp_dict["electronic_spatial_extent"] == 35.3
    assert temp_dict["zero_point_vibrational_energy"] == 0.0
    assert temp_dict["internal_energy_at_0K"] == -40.4
    assert temp_dict["internal_energy_at_298.15K"] == -40.4
    assert temp_dict["enthalpy_at_298.15K"] == -40.4
    assert temp_dict["free_energy_at_298.15K"] == -40.4
    assert temp_dict["heat_capacity_at_298.15K"] == 6.4

    # test parsing an entire file from our data directory with unit conversions
    fn = resources.files("modelforge").joinpath("tests", "data", "dsgdb9nsd_000001.xyz")
    data_dict_temp = qm9_data._parse_xyzfile(str(fn))

    # spot check values
    # geometry is a per atom property so will be of shape [m,n,3]
    # where for qm9, the number of conformers, m = 1
    assert np.all(
        np.isclose(
            data_dict_temp["geometry"],
            np.array(
                [
                    [
                        [-1.26981359e-03, 1.08580416e-01, 8.00099580e-04],
                        [2.15041600e-04, -6.03131760e-04, 1.97612040e-04],
                        [1.01173084e-01, 1.46375116e-01, 2.76574800e-05],
                        [-5.40815069e-02, 1.44752661e-01, -8.76643715e-02],
                        [-5.23813634e-02, 1.43793264e-01, 9.06397294e-02],
                    ]
                ]
            )
            * unit.nanometer,
        )
    )
    # [m, n, 1] shape
    assert np.all(
        data_dict_temp["charges"]
        == np.array([[[-0.535689], [0.133921], [0.133922], [0.133923], [0.133923]]])
        * unit.elementary_charge
    )
    assert data_dict_temp["isotropic_polarizability"] == [[13.21]] * unit.angstroms**3
    assert data_dict_temp["energy_of_homo"] == [[-0.3877]] * unit.hartree
    assert data_dict_temp["energy_of_lumo"] == [[0.1171]] * unit.hartree
    assert data_dict_temp["lumo-homo_gap"] == [[0.5048]] * unit.hartree
    assert data_dict_temp["electronic_spatial_extent"] == [[35.3641]] * unit.angstrom**2
    assert (
        data_dict_temp["zero_point_vibrational_energy"] == [[0.044749]] * unit.hartree
    )
    assert data_dict_temp["internal_energy_at_0K"] == [[-40.47893]] * unit.hartree
    assert data_dict_temp["internal_energy_at_298.15K"] == [[-40.476062]] * unit.hartree
    assert data_dict_temp["enthalpy_at_298.15K"] == [[-40.475117]] * unit.hartree
    assert data_dict_temp["free_energy_at_298.15K"] == [[-40.498597]] * unit.hartree
    assert (
        data_dict_temp["heat_capacity_at_298.15K"]
        == [[6.469]] * unit.calorie_per_mole / unit.kelvin
    )
    # atomic_numbers do not change with conformers, so it is defined as [n,1]
    assert np.all(
        data_dict_temp["atomic_numbers"] == np.array([[6], [1], [1], [1], [1]])
    )
    assert data_dict_temp["smiles_gdb-17"] == "C"
    assert data_dict_temp["smiles_b3lyp"] == "C"
    assert data_dict_temp["inchi_corina"] == "1S/CH4/h1H4"
    assert data_dict_temp["inchi_b3lyp"] == "1S/CH4/h1H4"
    # per molecule property, shape [m,3]
    assert np.all(
        data_dict_temp["rotational_constants"]
        == np.array([[157.7118, 157.70997, 157.70699]]) * unit.gigahertz
    )

    assert data_dict_temp["dipole_moment"] == [[0.0]] * unit.debye
    assert np.all(
        data_dict_temp["harmonic_vibrational_frequencies"]
        == np.array(
            [
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
            ]
        )
        / unit.centimeter
    )
    assert np.allclose(
        data_dict_temp["reference_energy_at_0K"],
        np.array([-39.847864]).reshape(1, 1) * unit.hartree,
    )
    assert np.allclose(
        data_dict_temp["formation_energy_at_0K"],
        np.array([-0.631066]).reshape(1, 1) * unit.hartree,
    )
    assert np.allclose(
        data_dict_temp["reference_energy_at_298.15K"],
        np.array([-39.840783]).reshape(1, 1) * unit.hartree,
    )
    assert np.allclose(
        data_dict_temp["reference_enthalpy_at_298.15K"],
        np.array([-39.836059]).reshape(1, 1) * unit.hartree,
    )
    assert np.allclose(
        data_dict_temp["reference_free_energy_at_298.15K"],
        np.array([-39.905025]).reshape(1, 1) * unit.hartree,
    )


def test_qm9_local_archive(prep_temp_dir):
    # test file extraction, parsing, and generation of hdf5 file from a local archive.
    qm9_data = QM9Curation(
        hdf5_file_name="qm9_test10.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    file_name_path = str(local_data_path) + "/first10.tar.bz2"
    assert os.path.isfile(file_name_path)

    from modelforge.utils.misc import extract_tarred_file

    extract_tarred_file(
        str(local_data_path), "first10.tar.bz2", str(prep_temp_dir), mode="r:bz2"
    )

    # pass the local file to the process_downloaded function
    qm9_data._process_downloaded(str(prep_temp_dir))

    assert len(qm9_data.data) == 10
    # internal_energy_at_0K in kj/mol
    names = {
        "dsgdb9nsd_000001": np.array([[-106277.4161215308]]),
        "dsgdb9nsd_000002": np.array([[-148408.69593977975]]),
        "dsgdb9nsd_000003": np.array([[-200600.51755556674]]),
        "dsgdb9nsd_000004": np.array([[-202973.24721725564]]),
        "dsgdb9nsd_000005": np.array([[-245252.87826713378]]),
        "dsgdb9nsd_000006": np.array([[-300576.6846578527]]),
        "dsgdb9nsd_000007": np.array([[-209420.75231941737]]),
        "dsgdb9nsd_000008": np.array([[-303715.5298633426]]),
        "dsgdb9nsd_000009": np.array([[-306158.32885940996]]),
        "dsgdb9nsd_000010": np.array([[-348451.454977435]]),
    }
    #
    # output file
    file_name_path = str(prep_temp_dir) + "/qm9_test10.hdf5"
    qm9_data._generate_hdf5()

    assert os.path.isfile(file_name_path)

    with h5py.File(file_name_path, "r") as hf:
        for key in hf.keys():
            # check record names
            assert key in list(names.keys())
            assert np.isclose(hf[key]["internal_energy_at_0K"][()], names[key])
            assert np.all(
                hf[key]["internal_energy_at_0K"][()].shape == names[key].shape
            )

    # clear out the
    qm9_data._clear_data()
    assert len(qm9_data.data) == 0

    qm9_data._process_downloaded(str(prep_temp_dir), max_records=5)

    assert len(qm9_data.data) == 5
    qm9_data._clear_data()

    qm9_data._process_downloaded(str(prep_temp_dir), total_conformers=5)
    assert len(qm9_data.data) == 5
    assert qm9_data.total_conformers == 5
    # only one conformer per record so these should be the same
    assert qm9_data.total_records == 5

    qm9_data._clear_data()
    qm9_data._process_downloaded(
        str(prep_temp_dir),
        max_records=2,
        total_conformers=5,
    )
    assert qm9_data.total_conformers == 5
    assert len(qm9_data.data) == 5


def test_an1_process_download_short(prep_temp_dir):
    # first check where we don't convert units
    ani1_data = ANI1xCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "ani1_n5.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    ani1_data._process_downloaded(str(local_data_path), hdf5_file)

    # ani1_n5.hdf5 datafile includes entries [843, 861, 872, 930, 932] from the full datafile
    # Example code snippet used to generate this file
    #
    # input_file_name = "ani1x-release.h5"
    #
    # with h5py.File(input_file_name, "r") as hf_in:
    #     with h5py.File('ani1_n5.hdf5', "w") as hf_out:
    #         test_names = list(hf_in.keys())
    #         test_names2 = []
    #         for i in [843, 861, 872, 930, 932]:
    #             test_names2.append(test_names[i])
    #         for test_name in test_names2:
    #             hf_in.copy(hf_in[test_name], hf_out)

    assert len(ani1_data.data) == 5

    assert ani1_data.data[0]["name"] == "C1H4N4O4"
    assert ani1_data.data[1]["name"] == "C1H5N3O5"
    assert ani1_data.data[2]["name"] == "C1H6N2O5"
    assert ani1_data.data[3]["name"] == "C2H10N4O3"
    assert ani1_data.data[4]["name"] == "C2H10N6O1"

    ani1_data._clear_data()
    assert len(ani1_data.data) == 0

    # test max records exclusion
    ani1_data._process_downloaded(str(local_data_path), hdf5_file, max_records=2)
    assert len(ani1_data.data) == 2


def test_an1_process_download_no_conversion(prep_temp_dir):
    from numpy import array, float32, uint8
    from openff.units import unit

    # first check where we don't convert units
    ani1_data = ANI1xCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=False,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "ani1_n5.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    ani1_data._process_downloaded(str(local_data_path), hdf5_file, max_records=1)

    #

    assert ani1_data.data[0]["name"] == "C1H4N4O4"
    assert np.all(
        ani1_data.data[0]["atomic_numbers"]
        == array([6, 1, 1, 1, 1, 7, 7, 7, 7, 8, 8, 8, 8], dtype=uint8).reshape(-1, 1)
    )
    assert ani1_data.data[0]["n_configs"] == 2

    assert np.all(
        np.isclose(
            ani1_data.data[0]["geometry"][0],
            array(
                [
                    [1.0234026, 0.82911, 0.10283028],
                    [-1.300884, 0.38319817, 1.3651426],
                    [-3.184876, -1.2480949, -0.64869606],
                    [0.17089044, -0.2244574, 1.6787908],
                    [2.1853921, 0.6403561, -1.6791393],
                    [0.7845485, -0.25825635, 0.926529],
                    [1.346796, -1.388733, 0.46308622],
                    [1.9548454, -1.1122795, -0.61758876],
                    [1.7804027, 0.20914805, -0.8599248],
                    [-1.1696881, -0.22681895, 0.548168],
                    [-1.9126405, 0.1927124, -0.5437298],
                    [-3.2172801, -0.28912666, -0.30542406],
                    [0.6649727, 1.9888812, 0.21126188],
                ],
                dtype=float32,
            )
            * unit.parse_expression("angstrom"),
        )
    )
    assert ani1_data.data[0]["wb97x_dz.energy"][
        0
    ] == -559.9673266512569 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["wb97x_tz.energy"][
        0
    ] == -560.215362918279 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["ccsd(t)_cbs.energy"][
        0
    ] == -559.6590647156545 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["hf_dz.energy"][
        0
    ] == -557.1375898559961 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["hf_tz.energy"][
        0
    ] == -557.3013494778872 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["hf_qz.energy"][
        0
    ] == -557.3426482230868 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["npno_ccsd(t)_dz.corr_energy"][
        0
    ] == -1.6281721046206972 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["npno_ccsd(t)_tz.corr_energy"][
        0
    ] == -2.02456426080263 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["tpno_ccsd(t)_dz.corr_energy"][
        0
    ] == -1.6309148176133395 * unit.parse_expression("hartree")
    assert ani1_data.data[0]["mp2_dz.corr_energy"][
        0
    ] == -1.5539720835219866 * unit.parse_expression("hartree / angstrom")
    assert ani1_data.data[0]["mp2_tz.corr_energy"][
        0
    ] == -1.9429519127460972 * unit.parse_expression("hartree / angstrom")
    assert ani1_data.data[0]["mp2_qz.corr_energy"][
        0
    ] == -2.0852302230766 * unit.parse_expression("hartree / angstrom")
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.forces"][0],
            array(
                [
                    [0.01657831, 0.00208556, 0.00961986],
                    [-0.03721527, -0.0103774, -0.08432362],
                    [0.00524313, 0.06732982, 0.02280491],
                    [0.00821256, -0.02387543, 0.06603277],
                    [-0.00206462, -0.00357482, 0.00431916],
                    [0.20106082, 0.01689906, -0.02965967],
                    [-0.00325114, 0.00842358, 0.00410941],
                    [-0.00429628, -0.00360803, 0.00523179],
                    [0.00421975, 0.00448409, -0.00442295],
                    [-0.13058083, 0.01246929, 0.01397585],
                    [-0.04244819, 0.00138314, 0.02921354],
                    [-0.01240513, -0.05592296, -0.03804544],
                    [-0.00305312, -0.0157159, 0.00114439],
                ],
                dtype=float32,
            )
            * unit.parse_expression("hartree / angstrom"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.forces"][0],
            array(
                [
                    [0.01855607, -0.00214226, 0.01092401],
                    [-0.03279239, -0.0199622, -0.09091964],
                    [0.00151677, 0.0789296, 0.02715147],
                    [0.01231755, -0.02255309, 0.05784418],
                    [-0.00507628, -0.00623896, 0.01045448],
                    [0.19719525, 0.01809859, -0.02661953],
                    [0.00315192, 0.02208645, -0.00703623],
                    [-0.01387949, 0.00197019, 0.01935457],
                    [0.00491522, 0.0051114, -0.00407311],
                    [-0.13600205, 0.0243378, 0.01847638],
                    [-0.0431021, -0.00464446, 0.03555103],
                    [-0.00483578, -0.06529695, -0.04488963],
                    [0.00254801, -0.02906634, -0.00092595],
                ],
                dtype=float32,
            )
            * unit.parse_expression("hartree / angstrom"),
        )
    )
    assert np.all(np.isnan(ani1_data.data[0]["wb97x_dz.dipole"][0].m))

    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.dipole"][0],
            array([-0.15512912, -0.2733479, 0.07883724], dtype=float32)
            * unit.parse_expression("angstrom * elementary_charge"),
        )
    )
    assert np.all(np.isnan(ani1_data.data[0]["wb97x_dz.quadrupole"][0].m))
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.cm5_charges"][0],
            array(
                [
                    0.351663,
                    0.349011,
                    0.346528,
                    0.386354,
                    0.374394,
                    -0.32615,
                    -0.086787,
                    -0.097374,
                    -0.292904,
                    -0.278695,
                    -0.039002,
                    -0.312688,
                    -0.374352,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.hirshfeld_charges"][0],
            array(
                [
                    0.171673,
                    0.180061,
                    0.18349,
                    0.137237,
                    0.169678,
                    -0.036654,
                    -0.062321,
                    -0.074566,
                    -0.035906,
                    -0.117717,
                    -0.010697,
                    -0.165545,
                    -0.338735,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_charges"][0],
            array(
                [
                    0.78357184,
                    0.37897816,
                    0.41236782,
                    0.38061607,
                    0.39502603,
                    -0.3705582,
                    -0.07132047,
                    -0.08506158,
                    -0.4143781,
                    -0.37768,
                    0.04198774,
                    -0.42587414,
                    -0.6540033,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_dipoles"][0],
            array(
                [
                    0.01980357,
                    0.03066077,
                    0.02896599,
                    0.03629951,
                    0.0400122,
                    0.1549512,
                    0.23462445,
                    0.23667449,
                    0.13459457,
                    0.17466189,
                    0.19845475,
                    0.13078435,
                    0.1239773,
                ],
                dtype=float32,
            ).reshape(-1, 1),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_quadrupoles"][0],
            array(
                [
                    0.17707957,
                    0.0253119,
                    0.02866661,
                    0.01287407,
                    0.02173789,
                    0.08036502,
                    0.37641832,
                    0.36518043,
                    0.10376307,
                    0.25798967,
                    0.3756285,
                    0.38644522,
                    0.13592605,
                ],
                dtype=float32,
            ).reshape(-1, 1),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_octupoles"][0],
            array(
                [
                    0.46536684,
                    0.02301415,
                    0.0209137,
                    0.00959256,
                    0.00437022,
                    0.86948353,
                    1.1523615,
                    1.1775111,
                    0.9538975,
                    0.6581496,
                    0.7705625,
                    0.8619628,
                    0.58347076,
                ],
                dtype=float32,
            ).reshape(-1, 1),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_volumes"][0],
            array(
                [
                    21.487135,
                    1.94764,
                    1.6454029,
                    1.2957854,
                    1.3529116,
                    28.231989,
                    27.362038,
                    27.493708,
                    28.782227,
                    23.989635,
                    20.909227,
                    25.17341,
                    26.948938,
                ],
                dtype=float32,
            ).reshape(-1, 1),
        )
    )

    # check that the shape of the arrays are what we expect
    assert ani1_data.data[0]["atomic_numbers"].shape == (13, 1)
    assert ani1_data.data[0]["geometry"].shape == (2, 13, 3)
    assert ani1_data.data[0]["wb97x_dz.energy"].shape == (2, 1)
    assert ani1_data.data[0]["wb97x_tz.energy"].shape == (2, 1)
    assert ani1_data.data[0]["ccsd(t)_cbs.energy"].shape == (2, 1)
    assert ani1_data.data[0]["hf_dz.energy"].shape == (2, 1)
    assert ani1_data.data[0]["hf_tz.energy"].shape == (2, 1)
    assert ani1_data.data[0]["hf_qz.energy"].shape == (2, 1)
    assert ani1_data.data[0]["npno_ccsd(t)_dz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["npno_ccsd(t)_tz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["tpno_ccsd(t)_dz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["mp2_dz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["mp2_tz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["mp2_qz.corr_energy"].shape == (2, 1)
    assert ani1_data.data[0]["wb97x_dz.forces"].shape == (2, 13, 3)
    assert ani1_data.data[0]["wb97x_tz.forces"].shape == (2, 13, 3)
    assert ani1_data.data[0]["wb97x_dz.dipole"].shape == (2, 3)
    assert ani1_data.data[0]["wb97x_tz.dipole"].shape == (2, 3)
    assert ani1_data.data[0]["wb97x_dz.quadrupole"].shape == (2, 6)
    assert ani1_data.data[0]["wb97x_dz.cm5_charges"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_dz.hirshfeld_charges"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_tz.mbis_charges"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_tz.mbis_dipoles"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_tz.mbis_quadrupoles"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_tz.mbis_octupoles"].shape == (2, 13, 1)
    assert ani1_data.data[0]["wb97x_tz.mbis_volumes"].shape == (2, 13, 1)


def test_an1_process_download_unit_conversion(prep_temp_dir):
    from numpy import array, float32, uint8
    from openff.units import unit

    # first check where we don't convert units
    ani1_data = ANI1xCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "ani1_n5.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    ani1_data._process_downloaded(str(local_data_path), hdf5_file, max_records=1)

    #

    assert ani1_data.data[0]["name"] == "C1H4N4O4"
    assert np.all(
        ani1_data.data[0]["atomic_numbers"]
        == array([6, 1, 1, 1, 1, 7, 7, 7, 7, 8, 8, 8, 8], dtype=uint8).reshape(-1, 1)
    )
    assert ani1_data.data[0]["n_configs"] == 2

    assert np.all(
        np.isclose(
            ani1_data.data[0]["geometry"][0],
            array(
                [
                    [0.10234026, 0.08291101, 0.01028303],
                    [-0.1300884, 0.03831982, 0.13651426],
                    [-0.3184876, -0.1248095, -0.06486961],
                    [0.01708904, -0.02244574, 0.16787909],
                    [0.21853922, 0.06403562, -0.16791393],
                    [0.07845485, -0.02582563, 0.0926529],
                    [0.1346796, -0.13887331, 0.04630862],
                    [0.19548455, -0.11122795, -0.06175888],
                    [0.17804027, 0.02091481, -0.08599248],
                    [-0.11696881, -0.0226819, 0.0548168],
                    [-0.19126405, 0.01927124, -0.05437298],
                    [-0.32172802, -0.02891267, -0.03054241],
                    [0.06649727, 0.19888812, 0.02112619],
                ],
                dtype=float32,
            )
            * unit.parse_expression("nanometer"),
        )
    )
    assert ani1_data.data[0]["wb97x_dz.energy"][
        0
    ] == -1470194.0142433804 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["wb97x_tz.energy"][
        0
    ] == -1470845.2333730247 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["ccsd(t)_cbs.energy"][
        0
    ] == -1469384.6726425907 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["hf_dz.energy"][
        0
    ] == -1462764.5413076002 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["hf_tz.energy"][
        0
    ] == -1463194.4921358367 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["hf_qz.energy"][
        0
    ] == -1463302.9219764692 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["npno_ccsd(t)_dz.corr_energy"][
        0
    ] == -4274.765273692818 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["npno_ccsd(t)_tz.corr_energy"][
        0
    ] == -5315.49273684113 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["tpno_ccsd(t)_dz.corr_energy"][
        0
    ] == -4281.966265666197 * unit.parse_expression("kilojoule_per_mole")
    assert ani1_data.data[0]["mp2_dz.corr_energy"][
        0
    ] == -4079.953145048755 * unit.parse_expression("kilojoule_per_mole / angstrom")
    assert ani1_data.data[0]["mp2_tz.corr_energy"][
        0
    ] == -5101.219546441598 * unit.parse_expression("kilojoule_per_mole / angstrom")
    assert ani1_data.data[0]["mp2_qz.corr_energy"][
        0
    ] == -5474.771198920137 * unit.parse_expression("kilojoule_per_mole / angstrom")
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.forces"][0],
            array(
                [
                    [43.526363, 5.475642, 25.25695],
                    [-97.708694, -27.245872, -221.39165],
                    [13.765839, 176.77443, 59.874275],
                    [21.56208, -62.68494, 173.36903],
                    [-5.420667, -9.385698, 11.339941],
                    [527.88513, 44.36848, -77.871445],
                    [-8.535871, 22.116117, 10.789246],
                    [-11.279875, -9.472887, 13.736053],
                    [11.078953, 11.772984, -11.612447],
                    [-342.83994, 32.73812, 36.693592],
                    [-111.44771, 3.6314259, 76.70014],
                    [-32.56966, -146.82571, -99.888306],
                    [-8.015977, -41.2621, 3.004605],
                ],
                dtype=float32,
            )
            * unit.parse_expression("kilojoule_per_mole / angstrom"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.forces"][0],
            array(
                [
                    [48.718956, -5.6245003, 28.680979],
                    [-86.09641, -52.410763, -238.7095],
                    [3.9822836, 207.22966, 71.28617],
                    [32.339718, -59.213142, 151.86987],
                    [-13.327763, -16.38038, 27.448246],
                    [517.7361, 47.51783, -69.88956],
                    [8.275371, 57.98798, -18.473612],
                    [-36.4406, 5.172723, 50.815422],
                    [12.904908, 13.419983, -10.693946],
                    [-357.07333, 63.8989, 48.509743],
                    [-113.16454, -12.194016, 93.33922],
                    [-12.696348, -171.43712, -117.85771],
                    [6.689795, -76.31368, -2.4310894],
                ],
                dtype=float32,
            )
            * unit.parse_expression("kilojoule_per_mole / angstrom"),
        )
    )
    assert ani1_data.data[0]["wb97x_dz.dipole"][0].u == unit.parse_expression("debye")
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.dipole"][0],
            array([-0.7451169, -1.312946, 0.37867138], dtype=float32)
            * unit.parse_expression("debye"),
        )
    )
    assert ani1_data.data[0]["wb97x_dz.quadrupole"][0].u == unit.parse_expression(
        "kilojoule_per_mole / angstrom ** 2"
    )

    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.cm5_charges"][0],
            array(
                [
                    0.351663,
                    0.349011,
                    0.346528,
                    0.386354,
                    0.374394,
                    -0.32615,
                    -0.086787,
                    -0.097374,
                    -0.292904,
                    -0.278695,
                    -0.039002,
                    -0.312688,
                    -0.374352,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_dz.hirshfeld_charges"][0],
            array(
                [
                    0.171673,
                    0.180061,
                    0.18349,
                    0.137237,
                    0.169678,
                    -0.036654,
                    -0.062321,
                    -0.074566,
                    -0.035906,
                    -0.117717,
                    -0.010697,
                    -0.165545,
                    -0.338735,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            ani1_data.data[0]["wb97x_tz.mbis_charges"][0],
            array(
                [
                    0.78357184,
                    0.37897816,
                    0.41236782,
                    0.38061607,
                    0.39502603,
                    -0.3705582,
                    -0.07132047,
                    -0.08506158,
                    -0.4143781,
                    -0.37768,
                    0.04198774,
                    -0.42587414,
                    -0.6540033,
                ],
                dtype=float32,
            ).reshape(-1, 1)
            * unit.parse_expression("elementary_charge"),
        )
    )


def spice114_process_download_short(prep_temp_dir):
    # first check where we don't convert units
    spice_data = SPICE114Curation(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "SPICE-1.1.4_n2.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    spice_data._process_downloaded(str(local_data_path), hdf5_file)

    # SPICE-1.1.4_n2.hdf5 datafile includes entries [0, 14643] from the full datafile
    # 14643 does not include mbis calculations; allowing us to check that
    # Example code snippet used to generate this file
    #
    # input_file_name = "SPICE-1.1.4.hdf5"
    #
    # with h5py.File(input_file_name, "r") as hf_in:
    #     with h5py.File("SPICE-1.1.4_n2.hdf5", "w") as hf_out:
    #         test_names = list(hf_in.keys())
    #         test_names2 = []
    #         for i in [0, 14643]:
    #             test_names2.append(test_names[i])
    #         for test_name in test_names2:
    #             hf_in.copy(hf_in[test_name], hf_out)

    assert len(spice_data.data) == 2

    assert spice_data.data[0]["name"] == "103147721"
    assert spice_data.data[1]["name"] == "C#C.[Na+]"

    spice_data._clear_data()
    assert len(spice_data.data) == 0

    # test max records exclusion
    spice_data._process_downloaded(str(local_data_path), hdf5_file, max_records=1)
    assert len(spice_data.data) == 1

    assert spice_data.data[0]["atomic_numbers"].shape == (27, 1)
    assert spice_data.data[0]["geometry"].shape == (50, 27, 3)
    assert spice_data.data[0]["formation_energy"].shape == (50, 1)
    assert spice_data.data[0]["dft_total_energy"].shape == (50, 1)
    assert spice_data.data[0]["dft_total_gradient"].shape == (50, 27, 3)
    assert spice_data.data[0]["mbis_charges"].shape == (50, 27, 1)
    assert spice_data.data[0]["mbis_dipoles"].shape == (50, 27, 3)
    assert spice_data.data[0]["mbis_quadrupoles"].shape == (50, 27, 3, 3)
    assert spice_data.data[0]["mbis_octupoles"].shape == (50, 27, 3, 3, 3)
    assert spice_data.data[0]["scf_dipole"].shape == (50, 3)
    assert spice_data.data[0]["scf_quadrupole"].shape == (50, 3, 3)
    assert spice_data.data[0]["mayer_indices"].shape == (50, 27, 27)
    assert spice_data.data[0]["wiberg_lowdin_indices"].shape == (50, 27, 27)


def test_baseclass_unit_conversion(prep_temp_dir):
    spice_data = SPICE114Curation(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=False,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "SPICE-1.1.4_n2.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    spice_data._process_downloaded(str(local_data_path), hdf5_file)

    assert spice_data.data[0]["geometry"].u == unit.bohr
    assert (
        spice_data.data[0]["dft_total_energy"][0] == -370.43397424571714 * unit.hartree
    )

    spice_data._convert_units()
    assert spice_data.data[0]["geometry"].u == unit.nanometer
    assert (
        spice_data.data[0]["dft_total_energy"][0]
        == -972574.265833225 * unit.kilojoule_per_mole
    )
    spice_data.qm_parameters["geometry"] = {"u_in": unit.bohr, "u_out": unit.hartree}

    with pytest.raises(Exception):
        spice_data._convert_units()

    spice_data.qm_parameters["geometry"] = {
        "u_in": unit.angstrom,
        "u_out": unit.nanometer,
    }
    with pytest.raises(AssertionError):
        spice_data._process_downloaded(str(local_data_path), hdf5_file)


def test_spice114_process_download_no_conversion(prep_temp_dir):
    from numpy import array, float32, uint8
    from openff.units import unit

    # first check where we don't convert units
    spice_data = SPICE114Curation(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=False,
    )

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "SPICE-1.1.4_n2.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    spice_data._process_downloaded(str(local_data_path), hdf5_file, max_records=1)

    #

    assert spice_data.data[0]["name"] == "103147721"
    assert np.all(
        spice_data.data[0]["atomic_numbers"]
        == array(
            [
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ).reshape(-1, 1)
    )
    assert spice_data.data[0]["n_configs"] == 50

    assert np.all(
        np.isclose(
            spice_data.data[0]["geometry"][0][0],
            array([1.3423489, 4.156236, -3.2724566], dtype=float32)
            * unit.parse_expression("bohr"),
        )
    )
    assert spice_data.data[0]["formation_energy"][
        0
    ] == -4.271002275159901 * unit.parse_expression("hartree")
    assert spice_data.data[0]["dft_total_energy"][
        0
    ] == -370.43397424571714 * unit.parse_expression("hartree")
    assert np.all(
        np.isclose(
            spice_data.data[0]["dft_total_gradient"][0][0],
            array([-0.00179922, 0.03140596, -0.01925333], dtype=float32)
            * unit.parse_expression("hartree / bohr"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_charges"][0][0],
            array([-0.22982304], dtype=float32)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_dipoles"][0][0],
            array([0.00291166, -0.03312059, 0.06175293], dtype=float32)
            * unit.parse_expression("bohr * elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_quadrupoles"][0][0],
            array(
                [
                    [-4.712843, 0.02795503, -0.01111934],
                    [0.02795503, -4.703707, -0.02350861],
                    [-0.01111934, -0.02350861, -4.731845],
                ],
                dtype=float32,
            )
            * unit.parse_expression("bohr ** 2 * elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_octupoles"][0][0],
            array(
                [
                    [
                        [-0.00367656, 0.1068773, 0.05573696],
                        [0.1068773, -0.19440877, 0.16380504],
                        [0.05573696, 0.16380504, 0.21165511],
                    ],
                    [
                        [0.1068773, -0.19440877, 0.16380504],
                        [-0.19440877, -0.03464365, 0.08447122],
                        [0.16380504, 0.08447122, -0.1936863],
                    ],
                    [
                        [0.05573696, 0.16380504, 0.21165511],
                        [0.16380504, 0.08447122, -0.1936863],
                        [0.21165511, -0.1936863, 0.10259467],
                    ],
                ],
                dtype=float32,
            )
            * unit.parse_expression("bohr ** 3 * elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["scf_dipole"][0][0],
            1.4204609 * unit.parse_expression("bohr * elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["scf_quadrupole"][0][0],
            array([-37.290867, -4.6239295, 3.8637419], dtype=float32)
            * unit.parse_expression("bohr ** 2 * elementary_charge"),
        )
    )


def test_spice114_process_download_conversion(prep_temp_dir):
    from numpy import array, float32, uint8
    from openff.units import unit

    # first check where we don't convert units
    spice_data = SPICE114Curation(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=str(prep_temp_dir),
        local_cache_dir=str(prep_temp_dir),
        convert_units=True,
    )

    charge = spice_data._calculate_reference_charge("C")
    assert charge == 0.0 * unit.elementary_charge

    charge = spice_data._calculate_reference_charge("[Na+]")
    assert charge == 1.0 * unit.elementary_charge

    local_data_path = resources.files("modelforge").joinpath("tests", "data")
    # make sure the data archive exists
    hdf5_file = "SPICE-1.1.4_n2.hdf5"
    file_name_path = str(local_data_path) + "/" + hdf5_file
    assert os.path.isfile(file_name_path)

    spice_data._process_downloaded(str(local_data_path), hdf5_file, max_records=1)

    #

    assert spice_data.data[0]["name"] == "103147721"
    assert np.all(
        spice_data.data[0]["atomic_numbers"]
        == array(
            [
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        ).reshape(-1, 1)
    )
    assert spice_data.data[0]["n_configs"] == 50

    assert np.all(
        np.isclose(
            spice_data.data[0]["geometry"][0][0],
            array([0.07103405, 0.21993855, -0.17317095], dtype=float32)
            * unit.parse_expression("nanometer"),
        )
    )
    assert spice_data.data[0]["formation_energy"][
        0
    ] == -11213.514933650016 * unit.parse_expression("kilojoule_per_mole")
    assert spice_data.data[0]["dft_total_energy"][
        0
    ] == -972574.265833225 * unit.parse_expression("kilojoule_per_mole")
    assert np.all(
        np.isclose(
            spice_data.data[0]["dft_total_gradient"][0][0],
            array([-8.926773, 155.8199, -95.524925], dtype=float32)
            * unit.parse_expression("kilojoule_per_mole / angstrom"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_charges"][0][0],
            array([-0.22982304], dtype=float32)
            * unit.parse_expression("elementary_charge"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_dipoles"][0][0],
            array([0.00015408, -0.00175267, 0.00326782], dtype=float32)
            * unit.parse_expression("elementary_charge * nanometer"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_quadrupoles"][0][0],
            array(
                [
                    [-1.3197304e-02, 7.8282050e-05, -3.1137311e-05],
                    [7.8282050e-05, -1.3171721e-02, -6.5830805e-05],
                    [-3.1137311e-05, -6.5830805e-05, -1.3250515e-02],
                ],
                dtype=float32,
            )
            * unit.parse_expression("elementary_charge * nanometer ** 2"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["mbis_octupoles"][0][0],
            array(
                [
                    [
                        [-5.44809382e-07, 1.58375824e-05, 8.25936604e-06],
                        [1.58375824e-05, -2.88084084e-05, 2.42734022e-05],
                        [8.25936604e-06, 2.42734022e-05, 3.13640521e-05],
                    ],
                    [
                        [1.58375824e-05, -2.88084084e-05, 2.42734022e-05],
                        [-2.88084084e-05, -5.13365876e-06, 1.25173437e-05],
                        [2.42734022e-05, 1.25173437e-05, -2.87013499e-05],
                    ],
                    [
                        [8.25936604e-06, 2.42734022e-05, 3.13640521e-05],
                        [2.42734022e-05, 1.25173437e-05, -2.87013499e-05],
                        [3.13640521e-05, -2.87013499e-05, 1.52029625e-05],
                    ],
                ],
                dtype=float32,
            )
            * unit.parse_expression("elementary_charge * nanometer ** 3"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["scf_dipole"][0][0],
            0.07516756 * unit.parse_expression("elementary_charge * nanometer"),
        )
    )
    assert np.all(
        np.isclose(
            spice_data.data[0]["scf_quadrupole"][0][0],
            array([-0.10442506, -0.01294832, 0.01081958], dtype=float32)
            * unit.parse_expression("elementary_charge * nanometer ** 2"),
        )
    )


def test_ani2x(prep_temp_dir):
    from modelforge.curation.ani2x_curation import ANI2xCuration

    local_path_dir = str(prep_temp_dir)
    local_data_path = resources.files("modelforge").joinpath("tests", "data")

    # create an hdf5 file that only contains the dimer data, generated using the following script
    # so we can test the _process_downloaded function
    # with h5py.File("ANI-2x-wB97X-631Gd.h5", "r") as hf_in:
    #     with h5py.File('ani2x_n10.hdf5', "w") as hf_out:
    #         for num_atoms, properties in hf_in.items():
    #             hf_in.copy(hf_in[num_atoms], hf_out)
    #             break

    filename = "ani2x_minimal.hdf5"

    ani2x_dataset = ANI2xCuration(
        hdf5_file_name="ani2x_dataset.hdf5",
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
    )
    ani2x_dataset._process_downloaded(local_data_path, filename, max_records=1)

    assert len(ani2x_dataset.data) == 1
    assert ani2x_dataset.data[0]["name"] == "[1_9]"
    assert ani2x_dataset.data[0]["n_configs"] == 517
    assert ani2x_dataset.data[0]["energies"].shape == (517, 1)
    assert ani2x_dataset.data[0]["atomic_numbers"].shape == (2, 1)
    assert ani2x_dataset.data[0]["geometry"].shape == (517, 2, 3)
    assert ani2x_dataset.data[0]["forces"].shape == (517, 2, 3)
    assert np.all(
        ani2x_dataset.data[0]["energies"][0:5].m
        == np.array(
            [
                [-263595.6246798465],
                [-263591.3453779841],
                [-263596.28778916295],
                [-263592.7221558636],
                [-263591.0017998503],
            ]
        ),
    )

    assert np.all(
        ani2x_dataset.data[0]["forces"][0:1].m
        == np.array(
            [[[-0.0, -0.0, 1058.7310900538516], [-0.0, -0.0, -1058.7310900538516]]]
        )
    )
    assert np.all(
        ani2x_dataset.data[0]["geometry"][0].m
        == np.array(
            [[0.0, 0.0, -0.08543934673070908], [0.0, 0.0, 0.009493260644376278]]
        )
    )


def test_spice114_openff_test_fetching(prep_temp_dir):
    from tqdm import tqdm
    from sqlitedict import SqliteDict

    local_path_dir = str(prep_temp_dir)
    local_database_name = "test.sqlite"
    specification_name = "entry"

    spice_openff_data = SPICEOpenFFCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="1.1.4",
    )

    # test downloading two new records and saving to the sqlite db
    spice_openff_data._fetch_singlepoint_from_qcarchive(
        dataset_name="SPICE PubChem Set 1 Single Points Dataset v1.2",
        specification_name=specification_name,
        local_database_name=local_database_name,
        local_path_dir=local_path_dir,
        force_download=True,
        max_records=2,
    )

    with SqliteDict(
        f"{local_path_dir}/{local_database_name}",
        tablename=specification_name,
        autocommit=True,
    ) as spice_db:
        keys = list(spice_db.keys())

        assert len(keys) == 2

    # same test as above, but we will pass pbar
    # pbar.total gets updated by the number of records
    # we need to fetch. Since force_download=True
    # we should fetch the 2 records again
    pbar = tqdm()
    pbar.total = 0

    spice_openff_data._fetch_singlepoint_from_qcarchive(
        dataset_name="SPICE PubChem Set 1 Single Points Dataset v1.2",
        specification_name=specification_name,
        local_database_name=local_database_name,
        local_path_dir=local_path_dir,
        force_download=True,
        max_records=2,
        pbar=pbar,
    )

    assert pbar.total == 2

    with SqliteDict(
        f"{local_path_dir}/{local_database_name}",
        tablename=specification_name,
        autocommit=True,
    ) as spice_db:
        keys = list(spice_db.keys())

        assert len(keys) == 2

    # test using sqlite db, by setting force_download=False
    pbar = tqdm()
    pbar.total = 0

    spice_openff_data._fetch_singlepoint_from_qcarchive(
        dataset_name="SPICE PubChem Set 1 Single Points Dataset v1.2",
        specification_name=specification_name,
        local_database_name=local_database_name,
        local_path_dir=local_path_dir,
        force_download=False,
        max_records=2,
    )

    assert pbar.total == 0

    with SqliteDict(
        f"{local_path_dir}/{local_database_name}",
        tablename=specification_name,
        autocommit=True,
    ) as spice_db:
        keys = list(spice_db.keys())

        assert len(keys) == 2

    # test fetching additional records
    # we already have 2 and thus should only need to
    # fetch 8

    pbar = tqdm()
    pbar.total = 0

    spice_openff_data._fetch_singlepoint_from_qcarchive(
        dataset_name="SPICE PubChem Set 1 Single Points Dataset v1.2",
        specification_name=specification_name,
        local_database_name=local_database_name,
        local_path_dir=local_path_dir,
        force_download=False,
        max_records=10,
        pbar=pbar,
    )

    assert pbar.total == 8

    with SqliteDict(
        f"{local_path_dir}/{local_database_name}",
        tablename=specification_name,
        autocommit=True,
    ) as spice_db:
        keys = list(spice_db.keys())

        assert len(keys) == 10


def test_spice114_rename(prep_temp_dir):
    local_path_dir = str(prep_temp_dir)
    local_database_name = "test.sqlite"
    specification_names = ["entry", "spec_2", "spec_6"]
    dataset_name = "SPICE PubChem Set 1 Single Points Dataset v1.2"

    spice_openff_data = SPICEOpenFFCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="1.1.4",
    )

    test_keys = ["ALA-1", "GLU-0", "GLU-1", "ALA-0", "ALA-2", "GLU-10", "GLU-3"]
    sorted_keys, original_keys = spice_openff_data._sort_keys(test_keys)

    assert np.all(
        sorted_keys == ["ALA-0", "ALA-1", "ALA-2", "GLU-0", "GLU-1", "GLU-3", "GLU-10"]
    )

    test_keys = [
        "ALA-ALA-1",
        "ALA-GLU-0",
        "ALA-GLU-1",
        "ALA-ALA-0",
        "ALA-ALA-2",
        "GLU-GLU-1",
        "GLU-GLU-0",
    ]
    assert original_keys[sorted_keys[0]] == "ALA-0"
    assert original_keys[sorted_keys[1]] == "ALA-1"
    assert original_keys[sorted_keys[2]] == "ALA-2"
    assert original_keys[sorted_keys[3]] == "GLU-0"
    assert original_keys[sorted_keys[4]] == "GLU-1"
    assert original_keys[sorted_keys[5]] == "GLU-3"
    assert original_keys[sorted_keys[6]] == "GLU-10"

    sorted_keys, original_keys = spice_openff_data._sort_keys(test_keys)
    assert np.all(
        sorted_keys
        == [
            "ALA_ALA-0",
            "ALA_ALA-1",
            "ALA_ALA-2",
            "ALA_GLU-0",
            "ALA_GLU-1",
            "GLU_GLU-0",
            "GLU_GLU-1",
        ]
    )
    assert original_keys[sorted_keys[0]] == "ALA-ALA-0"
    assert original_keys[sorted_keys[1]] == "ALA-ALA-1"
    assert original_keys[sorted_keys[2]] == "ALA-ALA-2"
    assert original_keys[sorted_keys[3]] == "ALA-GLU-0"
    assert original_keys[sorted_keys[4]] == "ALA-GLU-1"
    assert original_keys[sorted_keys[5]] == "GLU-GLU-0"
    assert original_keys[sorted_keys[6]] == "GLU-GLU-1"


def test_spice114_openff_test_process_downloaded(prep_temp_dir):
    from tqdm import tqdm
    from sqlitedict import SqliteDict

    local_path_dir = str(prep_temp_dir)
    local_database_name = "test.sqlite"
    specification_names = ["entry", "spec_2", "spec_6"]
    dataset_name = "SPICE PubChem Set 1 Single Points Dataset v1.2"

    spice_openff_data = SPICEOpenFFCuration(
        hdf5_file_name="test_dataset.hdf5",
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="1.1.4",
    )

    for specification_name in specification_names:
        # test downloading two new records and saving to the sqlite db
        spice_openff_data._fetch_singlepoint_from_qcarchive(
            dataset_name=dataset_name,
            specification_name=specification_name,
            local_database_name=local_database_name,
            local_path_dir=local_path_dir,
            force_download=True,
            umax_records=2,
        )

    spice_openff_data._process_downloaded(
        local_path_dir, [local_database_name], [dataset_name]
    )


def test_spice_114_openff_process_datasets(prep_temp_dir):
    from numpy import array, float32

    local_path_dir = str(prep_temp_dir)
    hdf5_file_name = "test_dataset.hdf5"

    spice_openff_data = SPICEOpenFFCuration(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="1.1.4",
    )

    self_energy, charge = spice_openff_data._calculate_reference_energy_and_charge("C")

    assert np.isclose(self_energy, -37.8726451 * unit.hartree)
    assert charge == 0.0 * unit.elementary_charge

    self_energy, charge = spice_openff_data._calculate_reference_energy_and_charge(
        "[Na+]"
    )

    assert np.isclose(self_energy, -162.113665 * unit.hartree)
    assert charge == 1.0 * unit.elementary_charge

    spice_openff_data.process(force_download=True, max_records=10, n_threads=3)

    # note that when we fetch the data, all the records are conformers of the same molecule
    # so we only end up with one molecule in data, but with 10 conformers
    assert sum([datapoint["n_configs"] for datapoint in spice_openff_data.data]) == 10

    assert spice_openff_data.data[0]["atomic_numbers"].shape == (32, 1)
    assert np.all(
        spice_openff_data.data[0]["atomic_numbers"]
        == np.array(
            [
                [7],
                [6],
                [7],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ]
        )
    )

    # spot check the energy
    assert np.all(
        spice_openff_data.data[0]["dft_total_energy"].m
        == np.array(
            [
                [-1516718.0904709378],
                [-1516683.816274856],
                [-1516700.3863245035],
                [-1516680.5107079088],
                [-1516640.745521272],
                [-1516695.1003979458],
                [-1516703.3300155026],
                [-1516644.1624653281],
                [-1516731.5604999226],
                [-1516641.2112121284],
            ]
        )
    )


def test_spice2_renaming(prep_temp_dir):
    local_path_dir = str(prep_temp_dir)
    hdf5_file_name = "test_spice2_dataset.hdf5"

    spice_2_data = SPICE2Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="2",
    )
    test_keys = ["ALA-1", "GLU-0", "GLU-1", "ALA-0", "ALA-2", "GLU-10", "GLU-3"]
    sorted_keys, original_keys, names = spice_2_data._sort_keys(test_keys)

    assert np.all(
        sorted_keys == ["ALA-0", "ALA-1", "ALA-2", "GLU-0", "GLU-1", "GLU-3", "GLU-10"]
    )

    test_keys = [
        "ALA-ALA-1",
        "ALA-GLU-0",
        "ALA-GLU-1",
        "ALA-ALA-0",
        "ALA-ALA-2",
        "GLU-GLU-1",
        "GLU-GLU-0",
    ]
    assert original_keys[sorted_keys[0]] == "ALA-0"
    assert original_keys[sorted_keys[1]] == "ALA-1"
    assert original_keys[sorted_keys[2]] == "ALA-2"
    assert original_keys[sorted_keys[3]] == "GLU-0"
    assert original_keys[sorted_keys[4]] == "GLU-1"
    assert original_keys[sorted_keys[5]] == "GLU-3"
    assert original_keys[sorted_keys[6]] == "GLU-10"

    assert names[sorted_keys[0]] == "ALA"
    assert names[sorted_keys[1]] == "ALA"
    assert names[sorted_keys[2]] == "ALA"
    assert names[sorted_keys[3]] == "GLU"
    assert names[sorted_keys[4]] == "GLU"
    assert names[sorted_keys[5]] == "GLU"
    assert names[sorted_keys[6]] == "GLU"

    sorted_keys, original_keys, names = spice_2_data._sort_keys(test_keys)
    assert np.all(
        sorted_keys
        == [
            "ALA_ALA-0",
            "ALA_ALA-1",
            "ALA_ALA-2",
            "ALA_GLU-0",
            "ALA_GLU-1",
            "GLU_GLU-0",
            "GLU_GLU-1",
        ]
    )
    assert original_keys[sorted_keys[0]] == "ALA-ALA-0"
    assert original_keys[sorted_keys[1]] == "ALA-ALA-1"
    assert original_keys[sorted_keys[2]] == "ALA-ALA-2"
    assert original_keys[sorted_keys[3]] == "ALA-GLU-0"
    assert original_keys[sorted_keys[4]] == "ALA-GLU-1"
    assert original_keys[sorted_keys[5]] == "GLU-GLU-0"
    assert original_keys[sorted_keys[6]] == "GLU-GLU-1"

    assert names[sorted_keys[0]] == "ALA_ALA"
    assert names[sorted_keys[1]] == "ALA_ALA"
    assert names[sorted_keys[2]] == "ALA_ALA"
    assert names[sorted_keys[3]] == "ALA_GLU"
    assert names[sorted_keys[4]] == "ALA_GLU"
    assert names[sorted_keys[5]] == "GLU_GLU"
    assert names[sorted_keys[6]] == "GLU_GLU"


def test_spice_2_process_datasets(prep_temp_dir):
    from numpy import array, float32

    local_path_dir = str(prep_temp_dir)
    hdf5_file_name = "test_spice2_dataset.hdf5"

    spice_2_data = SPICE2Curation(
        hdf5_file_name=hdf5_file_name,
        output_file_dir=local_path_dir,
        local_cache_dir=local_path_dir,
        convert_units=True,
        release_version="2",
    )

    self_energy, charge = spice_2_data._calculate_reference_energy_and_charge("C")

    assert np.isclose(self_energy, -37.8726451 * unit.hartree)
    assert charge == 0.0 * unit.elementary_charge

    self_energy, charge = spice_2_data._calculate_reference_energy_and_charge("[Na+]")

    assert np.isclose(self_energy, -162.113665 * unit.hartree)
    assert charge == 1.0 * unit.elementary_charge

    spice_2_data.process(force_download=True, max_records=10, n_threads=2)

    # note that when we fetch the data, all the records are conformers of the same molecule
    # so we only end up with one molecule in data, but with 10 conformers
    assert sum([datapoint["n_configs"] for datapoint in spice_2_data.data]) == 10

    assert spice_2_data.data[0]["atomic_numbers"].shape == (32, 1)
    assert spice_2_data.data[0]["dft_total_energy"].shape == (10, 1)
    assert spice_2_data.data[0]["dft_total_gradient"].shape == (10, 32, 3)

    # spot check the energy
    # kilojoules per mole
    known_energies = np.array(
        [
            [-1517627.69992024],
            [-1517601.06744853],
            [-1517609.47613857],
            [-1517603.27857799],
            [-1517565.96533027],
            [-1517617.0327266],
            [-1517629.78098516],
            [-1517569.06731524],
            [-1517649.95002649],
            [-1517562.26691007],
        ]
    )
    assert np.allclose(spice_2_data.data[0]["dft_total_energy"].m, known_energies)

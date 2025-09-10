import pytest
import os
import numpy as np
from openff.units import unit

from modelforge.curate import Record, SourceDataset
from modelforge.utils.units import GlobalUnitSystem
from modelforge.curate.properties import *

from modelforge.curate.datasets.curation_baseclass import DatasetCuration


def setup_test_dataset(dataset_name, local_cache_dir):
    class TestCuration(DatasetCuration):
        def _init_dataset_parameters(self):
            self.dataset = SourceDataset(
                name=self.dataset_name, local_db_dir=self.local_cache_dir
            )
            for i in range(5):
                atomic_numbers = AtomicNumbers(value=[[6 + i], [1]])
                positions = Positions(
                    value=[
                        [[i, 0, 0], [i, 1, 1]],
                        [[i, 0, 0], [i, 2, 2]],
                        [[i, 0, 0], [i, 3, 3]],
                    ],
                    units=unit.nanometer,
                )
                energy = Energies(
                    value=[[i], [i + 1.0], [i + 2.0]], units=unit.kilojoule_per_mole
                )
                forces = Forces(
                    value=[
                        [[0, 0, 0], [1, 1, 1]],
                        [[0, 0, 0], [2, 2, 2]],
                        [[0, 0, 0], [3, 3, 3]],
                    ],
                    units=unit.kilojoule_per_mole / unit.nanometer,
                )
                forces_with_different_key = Forces(
                    name="dft_total_force",
                    value=[
                        [[0, 0, 0], [5, 1, 1]],
                        [[0, 0, 0], [2, 2, 2]],
                        [[0, 0, 0], [3, 3, 3]],
                    ],
                    units=unit.kilojoule_per_mole / unit.nanometer,
                )
                record = Record(f"record_{i}")
                record.add_properties(
                    [
                        atomic_numbers,
                        positions,
                        energy,
                        forces,
                        forces_with_different_key,
                    ]
                )
                self.dataset.add_record(record)

    return TestCuration(dataset_name=dataset_name, local_cache_dir=local_cache_dir)


def test_center_of_mass(prep_temp_dir):
    class TestCuration(DatasetCuration):
        def _init_dataset_parameters(self):
            pass

    # test first where a molecule has the center of mass at zero
    atomic_numbers = np.array([[6], [1], [1], [1], [1]])
    # note this helper function accepts just a single molecule at a time
    # so the shape of positions is (n_atoms, 3)
    positions = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]])

    dataset_curation = TestCuration(
        dataset_name="test_dataset", local_cache_dir=str(prep_temp_dir)
    )

    center_of_mass = dataset_curation._calc_center_of_mass(
        atomic_numbers=atomic_numbers, positions=positions
    )

    assert np.allclose(center_of_mass, np.array([0.0, 0.0, 0.0]))

    # test a molecule where the center of mass is not at zero
    atomic_numbers = np.array([[8], [1], [1]])
    positions = np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]])

    center_of_mass = dataset_curation._calc_center_of_mass(
        atomic_numbers=atomic_numbers, positions=positions
    )
    assert np.allclose(center_of_mass, np.array([8.0, 0.0, 0.0]))

    # first test that we fail if we do not have the same number of atoms in atomic_numbers and positions
    with pytest.raises(ValueError):
        atomic_numbers = np.array([[8], [1], [1]])
        positions = np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        center_of_mass = dataset_curation._calc_center_of_mass(
            atomic_numbers=atomic_numbers, positions=positions
        )
    # test that we raise an error if the atomic_number shape is wrong
    with pytest.raises(ValueError):
        atomic_numbers = np.array([8, 1, 1])
        positions = np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        center_of_mass = dataset_curation._calc_center_of_mass(
            atomic_numbers=atomic_numbers, positions=positions
        )
    with pytest.raises(ValueError):
        atomic_numbers = np.array([[8, 1], [1, 1]])
        positions = np.array([[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        center_of_mass = dataset_curation._calc_center_of_mass(
            atomic_numbers=atomic_numbers, positions=positions
        )

    # test that we raise an error if the positions shape is wrong
    with pytest.raises(ValueError):
        atomic_numbers = np.array([[8], [1], [1]])
        positions = np.array([[[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]]])
        center_of_mass = dataset_curation._calc_center_of_mass(
            atomic_numbers=atomic_numbers, positions=positions
        )
    with pytest.raises(ValueError):
        atomic_numbers = np.array([[8], [1], [1]])
        positions = np.array(
            [[[8.0, 0.0, 0.0, 1.0], [9.0, 0.0, 0.0, 1.0], [7.0, 0.0, 0.0, 1.0]]]
        )
        center_of_mass = dataset_curation._calc_center_of_mass(
            atomic_numbers=atomic_numbers, positions=positions
        )


def test_dipolemoment_calculation(prep_temp_dir):
    class TestCuration(DatasetCuration):
        def _init_dataset_parameters(self):
            pass

    # testing on  007 ARG from spice2
    positions = Positions(
        value=np.array(
            [
                [
                    [0.41130853, -0.29579133, 0.05571501],
                    [0.70705056, 0.1196055, 0.06803529],
                    [0.4835338, -0.28912774, 0.15450846],
                    [0.6589673, 0.04629339, 0.15341634],
                    [0.09838394, 0.05620532, -0.12645122],
                    [-0.6303358, -0.31090915, -0.24944457],
                    [-0.4936998, -0.28256306, -0.24720705],
                    [-0.7212378, -0.21769346, -0.19913486],
                    [-0.44830486, -0.16090095, -0.1952675],
                    [-0.6755626, -0.09615002, -0.1471275],
                    [-0.5383345, -0.06459142, -0.14544845],
                    [0.1576179, 0.09251871, -0.23684597],
                    [0.69779694, 0.04807342, 0.2815741],
                    [0.14056315, 0.09597486, -0.0076066],
                    [-0.39908218, -0.06192343, 0.10733142],
                    [0.54727954, -0.17750567, 0.19178301],
                    [-0.00580153, -0.02132819, -0.13689402],
                    [0.50251305, -0.40993088, 0.24313287],
                    [0.802118, 0.13514376, 0.33638433],
                    [0.25189212, 0.1890528, 0.02089245],
                    [-0.48022667, 0.05438873, 0.06795251],
                    [-0.35274172, 0.26828426, -0.12936133],
                    [-0.49821386, 0.31230676, -0.11585703],
                    [-0.35592726, 0.11659911, -0.14608204],
                    [-0.58235157, 0.18511184, -0.12243588],
                    [0.3902137, 0.1224643, 0.01401945],
                    [-0.48925862, 0.06798078, -0.08603149],
                    [0.4101538, 0.01609841, 0.12334888],
                    [0.548297, -0.05216893, 0.1131132],
                    [-0.66591036, -0.40535703, -0.2896782],
                    [-0.4229554, -0.35540357, -0.2856994],
                    [-0.8275982, -0.2398047, -0.20007816],
                    [-0.34162116, -0.14243487, -0.19347116],
                    [-0.74874175, -0.02638481, -0.10771039],
                    [0.23932706, 0.1505536, -0.23084542],
                    [0.12480536, 0.06012018, -0.32588354],
                    [0.6483111, -0.01114349, 0.3459606],
                    [0.09013392, 0.06082617, 0.07210302],
                    [-0.43313748, -0.14495769, 0.05812184],
                    [-0.4042339, -0.07634643, 0.20824675],
                    [0.607546, -0.18532366, 0.27214822],
                    [-0.03684331, -0.05222198, -0.22709979],
                    [-0.05634984, -0.0501423, -0.05471128],
                    [0.44185555, -0.49193463, 0.20481461],
                    [0.60735893, -0.43929234, 0.24247025],
                    [0.47099647, -0.38605097, 0.34453267],
                    [0.7674253, 0.18162435, 0.4288271],
                    [0.8920841, 0.07704247, 0.35746026],
                    [0.8278184, 0.21416262, 0.2653536],
                    [0.23842983, 0.23181428, 0.1204975],
                    [0.24940616, 0.27178112, -0.05025344],
                    [-0.43801162, 0.1454013, 0.11184394],
                    [-0.5812581, 0.04236865, 0.10867327],
                    [-0.3043195, 0.31623146, -0.21505453],
                    [-0.29726067, 0.2952155, -0.03888037],
                    [-0.5265889, 0.38169438, -0.19565056],
                    [-0.51310253, 0.36256832, -0.01972566],
                    [-0.26984298, 0.0696943, -0.09778006],
                    [-0.35253912, 0.09308639, -0.2530187],
                    [-0.6196562, 0.17241816, -0.22468816],
                    [-0.6688282, 0.19180627, -0.05559914],
                    [0.4040311, 0.07671524, -0.08406249],
                    [0.46496728, 0.20086823, 0.02602954],
                    [0.39932564, 0.06216513, 0.22173649],
                    [0.33160236, -0.05863262, 0.11246528],
                    [0.5667641, -0.0782447, 0.00859369],
                ]
            ]
        ),
        units=unit.nanometer,
    )

    charges = PartialCharges(
        value=np.array(
            [
                [
                    [-0.63724726],
                    [-0.59185505],
                    [0.8215345],
                    [0.6737049],
                    [1.0231042],
                    [-0.13568029],
                    [-0.1261484],
                    [-0.11253503],
                    [-0.21952946],
                    [-0.19249053],
                    [0.06502347],
                    [-0.9252221],
                    [-0.56190723],
                    [-0.6400435],
                    [-0.96135616],
                    [-0.63288206],
                    [-0.95366013],
                    [-0.6824144],
                    [-0.17004295],
                    [0.00631138],
                    [0.0766362],
                    [-0.21106213],
                    [-0.19152217],
                    [-0.2250725],
                    [-0.2490422],
                    [-0.31882274],
                    [0.01574999],
                    [-0.25544018],
                    [-0.01889882],
                    [0.14373958],
                    [0.13563025],
                    [0.14401634],
                    [0.12618725],
                    [0.15054023],
                    [0.4461465],
                    [0.44775078],
                    [0.34217808],
                    [0.39638954],
                    [0.3908998],
                    [0.37163648],
                    [0.35155675],
                    [0.44855404],
                    [0.44908404],
                    [0.21671738],
                    [0.19125023],
                    [0.18140866],
                    [0.11754125],
                    [0.12133403],
                    [0.15507519],
                    [0.13962482],
                    [0.10906038],
                    [0.09654428],
                    [0.06360894],
                    [0.10333423],
                    [0.0907615],
                    [0.11341174],
                    [0.10244449],
                    [0.09175525],
                    [0.11169444],
                    [0.13099036],
                    [0.11420555],
                    [0.13457368],
                    [0.17605565],
                    [0.11712421],
                    [0.15005884],
                    [0.15786003],
                ]
            ]
        ),
        units=unit.elementary_charge,
    )

    atomic_numbers = AtomicNumbers(
        value=np.array(
            [
                [8],
                [8],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [6],
                [7],
                [7],
                [7],
                [7],
                [7],
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
                [1],
            ]
        )
    )

    scf_dipole_moment = DipoleMomentPerSystem(
        value=np.array([[0.11519327, 0.04117512, 0.0367095]]),
        units=unit.elementary_charge * unit.nanometer,
    )
    scf_dipole_magnitude = DipoleMomentScalarPerSystem(
        value=np.linalg.norm(scf_dipole_moment.value).reshape(1, 1),
        units=unit.elementary_charge * unit.nanometer,
    )
    dataset_curation = TestCuration(
        dataset_name="test_dataset", local_cache_dir=str(prep_temp_dir)
    )

    com = dataset_curation._calc_center_of_mass(
        atomic_numbers.value.reshape(-1, 1), positions.value[0]
    )
    dipole_moment_comp = dataset_curation.compute_dipole_moment(
        atomic_numbers=atomic_numbers, positions=positions, partial_charges=charges
    )
    dipole_moment_scaled_comp = dataset_curation.compute_dipole_moment(
        atomic_numbers=atomic_numbers,
        positions=positions,
        partial_charges=charges,
        dipole_moment_scalar=scf_dipole_magnitude,
    )

    # scf dipole: [[0.11519327, 0.04117512, 0.0367095]]
    # computed from partial charges: [[0.09112792, 0.05734182, 0.05859077]]
    # reasonably close

    assert np.all(scf_dipole_moment.value - dipole_moment_comp.value < 0.05)

    assert np.allclose(
        np.linalg.norm(dipole_moment_scaled_comp.value).reshape(1, 1),
        scf_dipole_magnitude.value,
    )

    atomic_numbers = AtomicNumbers(value=np.array([[8], [1], [1]]))
    positions = Positions(
        value=np.array([[[8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]]]),
        units=unit.nanometer,
    )
    charges = PartialCharges(
        value=np.array([[[-2], [1], [1]]]),
        units=unit.elementary_charge,
    )

    dipole_moment_comp = dataset_curation.compute_dipole_moment(
        atomic_numbers=atomic_numbers, positions=positions, partial_charges=charges
    )
    assert np.allclose(dipole_moment_comp.value, np.array([[0.0, 0.0, 0.0]]))

    charges = PartialCharges(
        value=np.array([[[-1], [1], [2]]]),
        units=unit.elementary_charge,
    )
    dipole_moment_comp = dataset_curation.compute_dipole_moment(
        atomic_numbers=atomic_numbers, positions=positions, partial_charges=charges
    )
    assert np.allclose(dipole_moment_comp.value, np.array([[-1.0, 0.0, 0.0]]))

    positions = Positions(
        value=np.array(
            [
                [[8, 0.0, 0.0], [9.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
                [[0, 0, 2], [0, 0, 3], [0, 0, 0]],
            ]
        ),
        units=unit.nanometer,
    )
    charges = PartialCharges(
        value=np.array([[[-2], [1], [1]], [[-1], [1], [1]]]),
        units=unit.elementary_charge,
    )
    dipole_moment_comp = dataset_curation.compute_dipole_moment(
        atomic_numbers=atomic_numbers, positions=positions, partial_charges=charges
    )
    assert np.allclose(dipole_moment_comp.value[0], np.array([0, 0, 0]))

    assert np.allclose(dipole_moment_comp.value[1], [0.0, 0.0, -0.9441], atol=1e-3)


def test_base_convert_element_string_to_atomic_number(prep_temp_dir):
    curated_dataset = setup_test_dataset("test_dataset_1a", str(prep_temp_dir))

    output = curated_dataset._convert_element_list_to_atomic_numbers(["C", "H"])
    assert np.all(output == np.array([6, 1]))


def test_base_operations(prep_temp_dir):
    output_dir = f"{prep_temp_dir}/test_base_operations"
    curated_dataset = setup_test_dataset("test_dataset_1b", output_dir)

    assert curated_dataset.dataset_name == "test_dataset_1b"
    assert curated_dataset.local_cache_dir == output_dir
    assert curated_dataset.total_records() == 5
    assert curated_dataset.total_configs() == 15

    # test writing the dataset
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test.hdf5", output_file_dir=output_dir
    )

    assert n_record == 5
    assert n_configs == 15
    assert os.path.exists(f"{output_dir}/test.hdf5")

    # test writing a subset of the dataset

    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_subset.hdf5",
        output_file_dir=output_dir,
        total_configurations=5,
    )
    assert n_record == 2
    assert n_configs == 5
    assert os.path.exists(f"{output_dir}/test_subset.hdf5")

    # test max_conformers per record setting
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_max_conformers.hdf5",
        output_file_dir=output_dir,
        max_configurations_per_record=1,
    )
    assert n_record == 5
    assert n_configs == 5
    assert os.path.exists(f"{output_dir}/test_max_conformers.hdf5")

    # test max_conformers_per_record and total_conformers
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_max_conformers_total_conf.hdf5",
        output_file_dir=output_dir,
        max_configurations_per_record=2,
        total_configurations=6,
    )
    assert n_record == 3
    assert n_configs == 6
    assert os.path.exists(f"{output_dir}/test_max_conformers_total_conf.hdf5")

    # test max records
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_max_records.hdf5",
        output_file_dir=output_dir,
        total_records=2,
    )
    assert n_record == 2
    assert n_configs == 6
    assert os.path.exists(f"{output_dir}/test_max_records.hdf5")

    # test restricting the species
    # since I updated the first species each time I add a record, the first record will have 6 and 1, the second 7 and 1, etc.
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=[6, 1],
    )
    assert n_record == 1
    assert n_configs == 3
    assert os.path.exists(f"{output_dir}/test_species.hdf5")

    # this should have 2 records
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species2.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=[6, 7, 1],
    )
    assert n_record == 2
    assert n_configs == 6
    assert os.path.exists(f"{output_dir}/test_species2.hdf5")

    # test combing with other restrictions
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species3.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=[6, 7, 1],
        total_records=1,
    )
    assert n_record == 1
    assert n_configs == 3
    assert os.path.exists(f"{output_dir}/test_species3.hdf5")

    # test combining with other restrictions some more
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species4.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=[6, 7, 1],
        max_configurations_per_record=1,
    )
    assert n_record == 2
    assert n_configs == 2
    assert os.path.exists(f"{output_dir}/test_species4.hdf5")

    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species5.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=[6, 7, 8, 1],
        max_configurations_per_record=2,
        total_configurations=5,
    )
    assert n_record == 3
    assert n_configs == 5
    assert os.path.exists(f"{output_dir}/test_species5.hdf5")

    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species5.hdf5",
        output_file_dir=output_dir,
        atomic_species_to_limit=["C", "N", "O", "H"],
        max_configurations_per_record=2,
        total_configurations=5,
    )
    assert n_record == 3
    assert n_configs == 5
    assert os.path.exists(f"{output_dir}/test_species5.hdf5")

    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_species5.hdf5",
        output_file_dir=output_dir,
        final_configuration_only=True,
    )
    assert n_record == 5
    assert n_configs == 5

    # test to see if we can remove high energy configurations
    # anything greater than 2.5 should exclude the last record
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_energy.hdf5",
        output_file_dir=output_dir,
        max_force=2.5 * unit.kilojoule_per_mole / unit.nanometer,
    )
    assert n_record == 5
    assert n_configs == 10

    # test to see if we can remove high energy configurations with a different key
    # anything greater than 2.5 should exclude the first and the last record
    n_record, n_configs = curated_dataset.to_hdf5(
        hdf5_file_name="test_energy.hdf5",
        output_file_dir=output_dir,
        max_force=2.5 * unit.kilojoule_per_mole / unit.nanometer,
        max_force_key="dft_total_force",
    )
    assert n_record == 5
    assert n_configs == 5

    # we can't define total_Records and total_configurations at the same time
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            total_records=5,
            total_configurations=5,
        )

    # we need to make sure that we have defined units with max_force
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            max_force=2.5,
        )

    # we need to make sure those units are actually force units
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            max_force=2.5 * unit.kilojoule_per_mole,
        )

    # make sure we have passed a list of atomic species we might want to limit
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            atomic_species_to_limit=6,
        )

    # exclude everything so that we have an empty dataset resulting from trimming
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            atomic_species_to_limit=[100],
        )

    # give a bad input to atomic_species_to_limit; it can only be a list of ints or strings
    with pytest.raises(ValueError):
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            atomic_species_to_limit=[True, False, True, False],
        )

    # make the original dataset empty
    with pytest.raises(ValueError):
        empty_dataset = SourceDataset(name="empty_dataset")
        curated_dataset.dataset = empty_dataset
        n_record, n_configs = curated_dataset.to_hdf5(
            hdf5_file_name="test_energy.hdf5",
            output_file_dir=output_dir,
            max_force=2.5 * unit.kilojoule_per_mole / unit.nanometer,
        )


def test_load_from_local_db(prep_temp_dir):
    output_dir = f"{str(prep_temp_dir)}/test_load_from_local_db"
    curated_dataset = setup_test_dataset("test_ds_1", local_cache_dir=output_dir)
    assert os.path.exists(f"{output_dir}/test_ds_1.sqlite")

    assert curated_dataset.total_records() == 5
    assert curated_dataset.total_configs() == 15

    class TestCuration(DatasetCuration):
        def _init_dataset_parameters(self):
            pass

    curated_dataset2 = TestCuration(
        dataset_name="test_ds_2", local_cache_dir=str(prep_temp_dir)
    )
    # since there has nothing been added to the dataset this will fail
    assert curated_dataset2.total_records() == 0
    assert curated_dataset2.total_configs() == 0

    curated_dataset2.load_from_db(
        local_db_dir=output_dir, local_db_name="test_ds_1.sqlite"
    )
    assert curated_dataset2.total_records() == 5
    assert curated_dataset2.total_configs() == 15

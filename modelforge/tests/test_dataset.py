import os
import sys

import numpy as np
import pytest
import torch

from modelforge.dataset.dataset import DatasetFactory, TorchDataset, BatchData
from modelforge.dataset import _ImplementedDatasets

from modelforge.utils.prop import PropertyNames


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    import uuid

    filename = str(uuid.uuid4())

    fn = tmp_path_factory.mktemp(f"test_dataset_temp")
    return fn


def test_dataset_imported():
    """Sample test, will always pass so long as import statement worked."""

    assert "modelforge.dataset" in sys.modules


def test_dataset_basic_operations():
    atomic_subsystem_counts = np.array([3, 4])
    n_confs = np.array([2, 1])
    total_atoms_series = (atomic_subsystem_counts * n_confs).sum()
    total_atoms_single = atomic_subsystem_counts.sum()
    total_confs = n_confs.sum()
    total_records = len(atomic_subsystem_counts)
    geom_shape = (total_atoms_series, 3)
    atomic_numbers_shape = (total_atoms_single, 1)
    internal_energy_shape = (total_confs, 1)
    input_data = {
        "geometry": np.arange(geom_shape[0] * geom_shape[1]).reshape(geom_shape),
        "atomic_numbers": np.arange(atomic_numbers_shape[0]).reshape(
            atomic_numbers_shape
        ),
        "internal_energy_at_0K": np.arange(internal_energy_shape[0]).reshape(
            internal_energy_shape
        ),
        "atomic_subsystem_counts": atomic_subsystem_counts,
        "n_confs": n_confs,
        "charges": torch.randint(-1, 2, torch.Size([total_confs, 1])).numpy(),
    }

    property_names = PropertyNames(
        atomic_numbers="atomic_numbers",
        positions="geometry",
        E="internal_energy_at_0K",
        total_charge="charges",
    )
    dataset = TorchDataset(input_data, property_names)
    assert len(dataset) == total_confs
    assert dataset.record_len() == total_records

    atomic_numbers_true = []
    geom_true = []
    energy_true = []
    series_mol_idxs = []
    atom_start_idx_series = 0
    atom_start_idx_single = 0
    conf_idx = 0
    for rec in range(total_records):
        series_mol_idxs_for_rec = []
        atom_end_idx_single = atom_start_idx_single + atomic_subsystem_counts[rec]
        for conf in range(n_confs[rec]):
            atom_end_idx_series = atom_start_idx_series + atomic_subsystem_counts[rec]
            energy_true.append(input_data["internal_energy_at_0K"][conf_idx])
            geom_true.append(
                input_data["geometry"][atom_start_idx_series:atom_end_idx_series]
            )
            atomic_numbers_true.append(
                input_data["atomic_numbers"][
                    atom_start_idx_single:atom_end_idx_single
                ].flatten()
            )
            series_mol_idxs_for_rec.append(conf_idx)
            atom_start_idx_series = atom_end_idx_series
            conf_idx += 1
        atom_start_idx_single = atom_end_idx_single
        series_mol_idxs.append(series_mol_idxs_for_rec)

    for conf_idx in range(len(dataset)):
        conf_data = dataset[conf_idx]
        pos1 = geom_true[conf_idx]
        pos2 = conf_data.nnp_input.positions
        assert np.array_equal(pos2, pos1)
        assert np.array_equal(
            conf_data.nnp_input.atomic_numbers, atomic_numbers_true[conf_idx]
        )
        assert np.array_equal(
            conf_data.metadata.per_system_energy, energy_true[conf_idx]
        )

    for rec_idx in range(dataset.record_len()):
        assert np.array_equal(
            dataset.get_series_mol_idxs(rec_idx), series_mol_idxs[rec_idx]
        )


@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_get_properties(dataset_name, single_batch_with_batchsize, prep_temp_dir):

    batch = single_batch_with_batchsize(
        batch_size=16, dataset_name=dataset_name, local_cache_dir=str(prep_temp_dir)
    )
    a = 7


@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_different_properties_of_interest(dataset_name, dataset_factory, prep_temp_dir):
    local_cache_dir = str(prep_temp_dir) + "/data_test"

    data = _ImplementedDatasets.get_dataset_class(
        dataset_name,
    )(version_select="nc_1000_v0", local_cache_dir=local_cache_dir)
    if dataset_name == "QM9":
        assert data.properties_of_interest == [
            "geometry",
            "atomic_numbers",
            "internal_energy_at_0K",
            "charges",
            "dipole_moment",
        ]
        # spot check the processing of the yaml file
        assert data.gz_data_file["length"] == 1697917
        assert data.gz_data_file["md5"] == "dc8ada0d808d02c699daf2000aff1fe9"
        assert data.gz_data_file["name"] == "qm9_dataset_v0_nc_1000.hdf5.gz"
        assert data.hdf5_data_file["md5"] == "305a0602860f181fafa75f7c7e3e6de4"
        assert data.hdf5_data_file["name"] == "qm9_dataset_v0_nc_1000.hdf5"
        assert (
            data.processed_data_file["name"] == "qm9_dataset_v0_nc_1000_processed.npz"
        )

        data.properties_of_interest = [
            "internal_energy_at_0K",
            "geometry",
            "atomic_numbers",
        ]
        assert data.properties_of_interest == [
            "internal_energy_at_0K",
            "geometry",
            "atomic_numbers",
        ]

    elif dataset_name == "ANI1x" or dataset_name == "ANI2x":
        assert data.properties_of_interest == [
            "geometry",
            "atomic_numbers",
            "wb97x_dz.energy",
            "wb97x_dz.forces",
            "dipole_moment",
        ]

        data.properties_of_interest = [
            "internal_energy_at_0K",
            "geometry",
            "atomic_numbers",
            "wb97x_dz.energy",
            "wb97x_dz.forces",
        ]
        assert data.properties_of_interest == [
            "internal_energy_at_0K",
            "geometry",
            "atomic_numbers",
            "wb97x_dz.energy",
            "wb97x_dz.forces",
        ]
    elif dataset_name == "SPICE2":
        assert data.properties_of_interest == [
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "mbis_charges",
            "scf_dipole",
        ]

        data.properties_of_interest = [
            "dft_total_energy",
            "geometry",
            "atomic_numbers",
            "mbis_charges",
        ]
        assert data.properties_of_interest == [
            "dft_total_energy",
            "geometry",
            "atomic_numbers",
            "mbis_charges",
        ]
    elif dataset_name == "PhAlkEthOH":
        assert data.properties_of_interest == [
            "geometry",
            "atomic_numbers",
            "dft_total_energy",
            "dft_total_force",
            "total_charge",
            "dipole_moment",
        ]

        data.properties_of_interest = [
            "dft_total_energy",
            "geometry",
            "atomic_numbers",
        ]
        assert data.properties_of_interest == [
            "dft_total_energy",
            "geometry",
            "atomic_numbers",
        ]

    dataset = dataset_factory(
        dataset_name=dataset_name, local_cache_dir=local_cache_dir
    )

    raw_data_item = dataset[0]
    assert isinstance(raw_data_item, BatchData)
    assert len(raw_data_item.__dataclass_fields__) == 2
    assert len(raw_data_item.nnp_input.__slots__) == 8  # 8 properties are returned
    assert len(raw_data_item.metadata.__slots__) == 6  # 6 properties are returned


@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_file_existence_after_initialization(
    dataset_name, dataset_factory, prep_temp_dir
):
    """Test if files are created after dataset initialization."""
    import contextlib

    local_cache_dir = str(prep_temp_dir) + "/data_test"

    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        local_cache_dir=local_cache_dir, version_select="nc_1000_v0"
    )

    with contextlib.suppress(FileNotFoundError):
        os.remove(f"{local_cache_dir}/{data.gz_data_file['name']}")
        os.remove(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
        os.remove(f"{local_cache_dir}/{data.processed_data_file['name']}")

    dataset = dataset_factory(
        dataset_name=dataset_name,
        local_cache_dir=local_cache_dir,
    )

    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")


def test_caching(prep_temp_dir):
    import contextlib

    local_cache_dir = str(prep_temp_dir) + "/data_test"
    from modelforge.dataset.qm9 import QM9Dataset

    data = QM9Dataset(version_select="nc_1000_v0", local_cache_dir=local_cache_dir)

    # first test that no file exists
    with contextlib.suppress(FileNotFoundError):
        os.remove(f"{local_cache_dir}/{data.gz_data_file['name']}")
        os.remove(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
        os.remove(f"{local_cache_dir}/{data.processed_data_file['name']}")
    assert not os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")
    # the _file_validation method also checks the path in addition to the checksum
    assert (
        data._file_validation(
            data.gz_data_file["name"], local_cache_dir, data.gz_data_file["md5"]
        )
        == False
    )

    data._download()
    # check that the file exists
    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")
    # check that the file is there and has the right checksum
    assert (
        data._file_validation(
            data.gz_data_file["name"], local_cache_dir, data.gz_data_file["md5"]
        )
        == True
    )

    # give a random checksum to see this is false
    assert (
        data._file_validation(
            data.gz_data_file["name"], local_cache_dir, "madeupcheckusm"
        )
        == False
    )
    # make sure that if we run again we don't fail
    data._download()
    # remove the file and check that it is downloaded again
    os.remove(f"{local_cache_dir}/{data.gz_data_file['name']}")
    data._download()

    # check that the file is unzipped
    data._ungzip_hdf5()
    assert os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    assert (
        data._file_validation(
            data.hdf5_data_file["name"], local_cache_dir, data.hdf5_data_file["md5"]
        )
        == True
    )
    data._from_hdf5()

    data._to_file_cache()

    # npz files saved with different versions of python lead to different checksums
    # we will skip checking the checksums for these files, only seeing if they exist
    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")
    assert (
        data._file_validation(
            data.processed_data_file["name"],
            local_cache_dir,
            None,
        )
        == True
    )

    data._from_file_cache()


def test_metadata_validation(prep_temp_dir):
    """When we generate an .npz file, we also write out metadata in a .json file
    which is used to validate if we can use .npz file, or we need to
    regenerate it."""

    local_cache_dir = str(prep_temp_dir) + "/data_test"

    from modelforge.dataset.qm9 import QM9Dataset

    data = QM9Dataset(version_select="nc_1000_v0", local_cache_dir=local_cache_dir)

    a = ["energy", "force", "atomic_numbers"]
    b = ["energy", "atomic_numbers", "force"]
    assert data._check_lists(a, b) == True

    a = ["energy", "force"]

    assert data._check_lists(a, b) == False

    a = ["energy", "force", "atomic_numbers", "charges"]

    assert data._check_lists(a, b) == False

    # we do not have a metadata files so this will fail
    assert data._metadata_validation("qm9_test.json", local_cache_dir) == False

    metadata = {
        "data_keys": [
            "atomic_numbers",
            "internal_energy_at_0K",
            "geometry",
            "charges",
            "dipole_moment",
        ],
        "hdf5_checksum": "305a0602860f181fafa75f7c7e3e6de4",
        "hdf5_gz_checkusm": "dc8ada0d808d02c699daf2000aff1fe9",
        "date_generated": "2024-04-11 14:05:14.297305",
    }

    import json

    # create local_cache_dir if not already present
    import os

    os.makedirs(local_cache_dir, exist_ok=True)

    with open(
        f"{local_cache_dir}/qm9_test.json",
        "w+",
    ) as f:
        json.dump(metadata, f)

    assert data._metadata_validation("qm9_test.json", local_cache_dir) == True

    metadata["hdf5_checksum"] = "wrong_checksum"
    with open(
        f"{local_cache_dir}/qm9_test.json",
        "w",
    ) as f:
        json.dump(metadata, f)
    assert data._metadata_validation("qm9_test.json", local_cache_dir) == False


@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_different_scenarios_of_file_availability(
    dataset_name, prep_temp_dir, dataset_factory
):
    """Test the behavior when raw and processed dataset files are removed."""
    local_cache_dir = str(prep_temp_dir) + "/test_diff_scenario"

    # this will download the .gz, the .hdf5 and the .npz files
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    # we initialize this so that we have the correct parameters to compare against
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        version_select="nc_1000_v0", local_cache_dir=local_cache_dir
    )

    # first check if we remove the npz file, rerunning it will regenerate it
    os.remove(f"{local_cache_dir}/{data.processed_data_file['name']}")
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)

    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")

    # now remove metadata file, rerunning will regenerate the npz file
    os.remove(
        f"{local_cache_dir}/{data.processed_data_file['name'].replace('npz', 'json')}"
    )
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    assert os.path.exists(
        f"{local_cache_dir}/{data.processed_data_file['name'].replace('npz', 'json')}"
    )

    # now remove the  npz and hdf5 files, rerunning will generate it

    os.remove(f"{local_cache_dir}/{data.processed_data_file['name']}")
    os.remove(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)

    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")

    # now remove the gz file; rerunning should NOT download, it will use the npz
    os.remove(f"{local_cache_dir}/{data.gz_data_file['name']}")

    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    assert not os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")

    # now let us remove the hdf5 file, it should use the npz file
    os.remove(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    assert not os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")

    # now if we remove the npz, it will redownload the gz file and unzip it, then process it
    os.remove(f"{local_cache_dir}/{data.processed_data_file['name']}")
    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")

    # now we will remove the gz file, and set force_download to True
    # this should now regenerate the gz file, even though others are present

    dataset_factory(dataset_name=dataset_name, local_cache_dir=local_cache_dir)
    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.hdf5_data_file['name']}")
    assert os.path.exists(f"{local_cache_dir}/{data.processed_data_file['name']}")

    # now we will remove the gz file and run it again
    os.remove(f"{local_cache_dir}/{data.gz_data_file['name']}")
    dataset_factory(
        dataset_name=dataset_name, local_cache_dir=local_cache_dir, force_download=True
    )
    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")


@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_data_item_format_of_datamodule(
    dataset_name, datamodule_factory, prep_temp_dir
):
    """Test the format of individual data items in the dataset."""
    from typing import Dict

    local_cache_dir = str(prep_temp_dir) + "/data_test"

    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        local_cache_dir=local_cache_dir,
    )

    raw_data_item = dm.torch_dataset[0]
    assert isinstance(raw_data_item, BatchData)
    assert isinstance(raw_data_item.nnp_input.atomic_numbers, torch.Tensor)
    assert isinstance(raw_data_item.nnp_input.positions, torch.Tensor)
    assert isinstance(raw_data_item.metadata.per_system_energy, torch.Tensor)

    assert (
        raw_data_item.nnp_input.atomic_numbers.shape[0]
        == raw_data_item.nnp_input.positions.shape[0]
    )


from modelforge.potential import _Implemented_NNPs


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_dataset_neighborlist(
    potential_name, single_batch_with_batchsize, prep_temp_dir
):
    """Test the neighborlist."""

    batch = single_batch_with_batchsize(64, "QM9", str(prep_temp_dir))
    nnp_input = batch.nnp_input

    # test that the neighborlist is correctly generated
    from modelforge.tests.helper_functions import setup_potential_for_test

    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="ani2x",
        simulation_environment="PyTorch",
        use_training_mode_neighborlist=True,
    )
    model(nnp_input)

    pair_list = nnp_input.pair_list
    # pairlist is in ascending order in row 0
    assert torch.all(pair_list[0, 1:] >= pair_list[0, :-1])

    # test the pairlist for message passing networks (with redundant atom pairs)
    # first molecule is methane, check if bonds are correct
    methane_bonds = pair_list[:, :20]

    assert (
        torch.any(
            torch.eq(
                methane_bonds,
                torch.tensor(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            1,
                            2,
                            2,
                            2,
                            2,
                            3,
                            3,
                            3,
                            3,
                            4,
                            4,
                            4,
                            4,
                        ],
                        [
                            1,
                            2,
                            3,
                            4,
                            0,
                            2,
                            3,
                            4,
                            0,
                            1,
                            3,
                            4,
                            0,
                            1,
                            2,
                            4,
                            0,
                            1,
                            2,
                            3,
                        ],
                    ]
                ),
            )
            == False
        ).item()
        == False
    )
    # second molecule is ammonium, check if bonds are correct
    ammonium_bonds = pair_list[:, 20:30]
    assert (
        torch.any(
            torch.eq(
                ammonium_bonds,
                torch.tensor(
                    [
                        [5, 5, 5, 6, 6, 6, 7, 7, 7, 8],
                        [6, 7, 8, 5, 7, 8, 5, 6, 8, 5],
                    ]
                ),
            )
            == False
        ).item()
        == False
    )


@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_dataset_generation(dataset_name, datamodule_factory, prep_temp_dir):
    """Test the splitting of the dataset."""

    dataset = datamodule_factory(
        dataset_name=dataset_name, local_cache_dir=str(prep_temp_dir)
    )
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()
    try:
        dataset.test_dataloader()
    except AttributeError:
        # this isn't set when dataset is in 'fit' mode
        pass

    # the dataloader automatically splits and batches the dataset
    # for the training set it batches the 800 training datapoints (of 1000 total) in 13 batches
    # all with 64 points until the last which has 32

    assert len(train_dataloader) == 13  # nr of batches
    batch_data = [v_ for v_ in train_dataloader]
    val_data = [v_ for v_ in val_dataloader]
    sum_batch = sum([len(b.metadata.atomic_subsystem_counts) for b in batch_data])
    sum_val = sum([len(b.metadata.atomic_subsystem_counts) for b in val_data])
    sum_test = sum([len(b.metadata.atomic_subsystem_counts) for b in test_dataloader])

    assert sum_batch == 800
    assert sum_val == 100
    assert sum_test == 100

    assert len(batch_data[0].metadata.atomic_subsystem_counts) == 64
    assert len(batch_data[1].metadata.atomic_subsystem_counts) == 64
    assert len(batch_data[-1].metadata.atomic_subsystem_counts) == 32


from modelforge.dataset.utils import (
    RandomRecordSplittingStrategy,
    RandomSplittingStrategy,
    FirstComeFirstServeSplittingStrategy,
)


@pytest.mark.parametrize(
    "splitting_strategy",
    [
        RandomSplittingStrategy,
        FirstComeFirstServeSplittingStrategy,
        RandomRecordSplittingStrategy,
    ],
)
@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_dataset_splitting(
    splitting_strategy,
    dataset_name,
    datamodule_factory,
    get_dataset_container_fix,
    prep_temp_dir,
):
    """Test random_split on the the dataset."""
    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=splitting_strategy(),
        version_select="nc_1000_v0",
        remove_self_energies=False,
        local_cache_dir=str(prep_temp_dir),
    )

    train_dataset, val_dataset, test_dataset = (
        dm.train_dataset,
        dm.val_dataset,
        dm.test_dataset,
    )

    energy = train_dataset[0].metadata.per_system_energy.item()
    dataset_to_test = get_dataset_container_fix(dataset_name)
    if splitting_strategy == RandomSplittingStrategy:
        assert np.isclose(energy, dataset_to_test.expected_E_random_split)
    elif splitting_strategy == FirstComeFirstServeSplittingStrategy:
        assert np.isclose(energy, dataset_to_test.expected_E_fcfs_split)

    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=splitting_strategy(split=[0.6, 0.3, 0.1]),
        version_select="nc_1000_v0",
        remove_self_energies=False,
        local_cache_dir=str(prep_temp_dir),
    )

    train_dataset2, val_dataset2, test_dataset2 = (
        dm.train_dataset,
        dm.val_dataset,
        dm.test_dataset,
    )

    if (
        splitting_strategy == RandomSplittingStrategy
        or splitting_strategy == FirstComeFirstServeSplittingStrategy
    ):
        total = len(train_dataset2) + len(val_dataset2) + len(test_dataset2)
        print(len(train_dataset2), len(val_dataset2), len(test_dataset2), total)
        assert np.isclose(len(train_dataset2) / total, 0.6, atol=0.01)
        assert np.isclose(len(val_dataset2) / total, 0.3, atol=0.01)
        assert np.isclose(len(test_dataset2) / total, 0.1, atol=0.01)
    elif splitting_strategy == RandomRecordSplittingStrategy:
        # for the random record splitting we need to have a larger tolerance
        # as we are not guaranteed to get the exact split since the number of conformers per record is not fixed
        total = len(train_dataset2) + len(val_dataset2) + len(test_dataset2)

        assert np.isclose(len(train_dataset2) / total, 0.6, atol=0.05)
        assert np.isclose(len(val_dataset2) / total, 0.3, atol=0.05)
        assert np.isclose(len(test_dataset2) / total, 0.1, atol=0.05)

    try:
        splitting_strategy(split=[0.2, 0.1, 0.1])
    except AssertionError as excinfo:
        print(f"AssertionError raised: {excinfo}")


@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_dataset_downloader(dataset_name, dataset_factory, prep_temp_dir):
    """
    Test the DatasetDownloader functionality.
    """
    local_cache_dir = str(prep_temp_dir) + "/data_test"

    dataset = dataset_factory(
        dataset_name=dataset_name, local_cache_dir=local_cache_dir
    )
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        local_cache_dir=local_cache_dir, version_select="nc_1000_v0"
    )
    assert os.path.exists(f"{local_cache_dir}/{data.gz_data_file['name']}")


@pytest.mark.parametrize("dataset_name", _ImplementedDatasets.get_all_dataset_names())
def test_numpy_dataset_assignment(dataset_name):
    """
    Test if the numpy_dataset attribute is correctly assigned after processing or loading.
    """
    from modelforge.dataset import _ImplementedDatasets

    factory = DatasetFactory()
    data = _ImplementedDatasets.get_dataset_class(dataset_name)(
        version_select="nc_1000_v0"
    )
    factory._load_or_process_data(data)

    assert hasattr(data, "numpy_data")
    assert isinstance(data.numpy_data, np.lib.npyio.NpzFile)


def test_energy_postprocessing(prep_temp_dir):
    # test that the mean and stddev of the dataset
    # are correct
    from modelforge.dataset.dataset import DataModule

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # -------------------------------#
    # Test that we can calculate the normalize energies correctly
    dm = DataModule(
        name="QM9",
        batch_size=10,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.val_dataloader()))
    unnormalized_E = batch.metadata.per_system_energy.numpy().flatten()
    import numpy as np

    # check that normalized energies are correct
    assert torch.allclose(
        batch.metadata.per_system_energy.squeeze(1),
        torch.tensor(
            [
                [
                    -5966.9515,
                    -6157.1063,
                    -5612.6762,
                    -5385.5678,
                    -4396.5738,
                    -5568.9688,
                    -4778.9399,
                    -6732.1988,
                    -5960.1068,
                    -6156.9383,
                ]
            ],
            dtype=torch.float64,
        ),
    )

    # check that we have saved the dataset statistics
    # correctly
    f = dm.dataset_statistic_filename
    import toml

    dataset_statistic = toml.load(f)
    from openff.units import unit

    assert np.isclose(
        unit.Quantity(
            dataset_statistic["training_dataset_statistics"]["per_atom_energy_mean"]
        ).m,
        -402.916561,
    )

    assert np.isclose(
        unit.Quantity(
            dataset_statistic["training_dataset_statistics"]["per_atom_energy_stddev"]
        ).m,
        25.013382078330697,
    )

    # check that the normalization is correct
    normalized_atomic_energies = (
        unnormalized_E / batch.metadata.atomic_subsystem_counts.numpy().flatten()
    )
    mean = np.average(normalized_atomic_energies)
    stddev = np.std(normalized_atomic_energies)

    # seams reasonable
    assert np.isclose(mean, -388.36276540521123)
    assert np.isclose(stddev, 19.372371857226035)


@pytest.mark.parametrize("dataset_name", ["QM9"])
def test_function_of_self_energy(dataset_name, datamodule_factory, prep_temp_dir):
    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # prepare reference value
    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        version_select="nc_1000_v0",
        remove_self_energies=False,
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )

    methane_energy_reference = float(dm.train_dataset[0].metadata.per_system_energy)
    assert np.isclose(methane_energy_reference, -106277.4161)

    # Scenario 1: dataset contains self energies
    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )
    # it is saved in the dataset statistics

    import toml

    f = dm.dataset_statistic_filename
    dataset_statistic = toml.load(f)
    self_energies = dataset_statistic["atomic_self_energies"]
    from openff.units import unit

    # 5 elements present in the QM9 dataset
    assert len(self_energies.keys()) == 5
    # H: -1313.4668615546
    assert np.isclose(unit.Quantity(self_energies["H"]).m, -1313.4668615546)
    # C: -99366.70745535441
    assert np.isclose(unit.Quantity(self_energies["C"]).m, -99366.70745535441)
    # N: -143309.9379722722
    assert np.isclose(unit.Quantity(self_energies["N"]).m, -143309.9379722722)
    # O: -197082.0671774158
    assert np.isclose(unit.Quantity(self_energies["O"]).m, -197082.0671774158)

    # Scenario 2: dataset may or may not contain self energies
    # but user wants to use least square regression to calculate the energies
    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        regression_ase=True,
        remove_self_energies=True,
        version_select="nc_1000_v0",
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )

    # it is saved in the dataset statistics
    import toml

    f = dm.dataset_statistic_filename
    dataset_statistic = toml.load(f)
    self_energies = dataset_statistic["atomic_self_energies"]

    # 5 elements present in the total QM9 dataset
    assert len(self_energies.keys()) == 5
    # value from DFT calculation
    # H: -1313.4668615546
    assert np.isclose(
        unit.Quantity(self_energies["H"]).m,
        -1577.0870687452618,
    )
    # value from DFT calculation
    # C: -99366.70745535441
    assert np.isclose(
        unit.Quantity(self_energies["C"]).m,
        -99977.40806211969,
    )
    # value from DFT calculation
    # N: -143309.9379722722
    assert np.isclose(
        unit.Quantity(self_energies["N"]).m,
        -143742.7416655554,
    )
    # value from DFT calculation
    # O: -197082.0671774158
    assert np.isclose(
        unit.Quantity(self_energies["O"]).m,
        -197492.33270235246,
    )

    dm = datamodule_factory(
        dataset_name=dataset_name,
        batch_size=512,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        regression_ase=True,
        remove_self_energies=True,
        version_select="nc_1000_v0",
        regenerate_dataset_statistic=True,
        local_cache_dir=str(prep_temp_dir),
    )
    # it is saved in the dataset statistics
    import toml

    f = dm.dataset_statistic_filename
    dataset_statistic = toml.load(f)
    self_energies = dataset_statistic["atomic_self_energies"]

    # Test that self energies are correctly removed
    for regression in [True, False]:
        dm = datamodule_factory(
            dataset_name=dataset_name,
            batch_size=512,
            splitting_strategy=FirstComeFirstServeSplittingStrategy(),
            regression_ase=regression,
            remove_self_energies=True,
            version_select="nc_1000_v0",
            regenerate_dataset_statistic=True,
            local_cache_dir=str(prep_temp_dir),
        )
        # Extract the first molecule (methane)
        # double check that it is methane
        methane_atomic_indices = dm.train_dataset[0].nnp_input.atomic_numbers
        # extract energy
        methane_energy_offset = dm.train_dataset[0].metadata.per_system_energy
        if regression is False:
            # checking that the offset energy is actually correct for methane
            assert torch.isclose(
                methane_energy_offset, torch.tensor([-1656.8412], dtype=torch.float64)
            )
        # extract the ase offset
        from modelforge.potential.processing import (
            AtomicSelfEnergies,
            load_atomic_self_energies,
        )

        self_energies = load_atomic_self_energies(dm.dataset_statistic_filename)

        self_energies = AtomicSelfEnergies(self_energies)

        methane_ase = sum(
            [
                self_energies[int(index)]
                for index in list(methane_atomic_indices.numpy())
            ]
        )
        # compare this to the energy without postprocessing
        assert np.isclose(methane_energy_reference, methane_energy_offset + methane_ase)


def test_shifting_center_of_mass_to_origin(prep_temp_dir):
    local_cache_dir = str(prep_temp_dir) + "/data_test"

    from modelforge.dataset.dataset import initialize_datamodule
    from openff.units.elements import MASSES

    import torch

    # first check a molecule not centered at the origin
    dm = initialize_datamodule(
        "QM9",
        version_select="latest_test",
        shift_center_of_mass_to_origin=False,
        local_cache_dir=local_cache_dir,
    )
    start_idx = dm.torch_dataset.single_atom_start_idxs_by_conf[0]
    end_idx = dm.torch_dataset.single_atom_end_idxs_by_conf[0]

    from openff.units.elements import MASSES

    pos = dm.torch_dataset.properties_of_interest["positions"][start_idx:end_idx]

    atomic_masses = torch.Tensor(
        [
            MASSES[atomic_number].m
            for atomic_number in dm.torch_dataset.properties_of_interest[
                "atomic_numbers"
            ][start_idx:end_idx].tolist()
        ]
    )
    molecule_mass = torch.sum(atomic_masses)

    # I'm using einsum, so let us check it manually

    x = 0
    y = 0
    z = 0
    for i in range(0, pos.shape[0]):
        x += atomic_masses[i] * pos[i][0]
        y += atomic_masses[i] * pos[i][1]
        z += atomic_masses[i] * pos[i][2]

    x = x / molecule_mass
    y = y / molecule_mass
    z = z / molecule_mass

    com = torch.Tensor([x, y, z])

    assert torch.allclose(com, torch.Tensor([-0.0013, 0.1086, 0.0008]), atol=1e-4)

    # make sure that we do shift to the origin; we can do the whole dataset

    dm = initialize_datamodule(
        "QM9",
        version_select="latest_test",
        shift_center_of_mass_to_origin=True,
        local_cache_dir=local_cache_dir,
    )

    for conf_id in range(0, len(dm.torch_dataset)):
        start_idx = dm.torch_dataset.single_atom_start_idxs_by_conf[conf_id]
        end_idx = dm.torch_dataset.single_atom_end_idxs_by_conf[conf_id]

        # grab the positions that should be shifted
        pos = dm.torch_dataset.properties_of_interest["positions"][start_idx:end_idx]

        atomic_masses = torch.Tensor(
            [
                MASSES[atomic_number].m
                for atomic_number in dm.torch_dataset.properties_of_interest[
                    "atomic_numbers"
                ][start_idx:end_idx].tolist()
            ]
        )
        molecule_mass = torch.sum(atomic_masses)

        x = 0
        y = 0
        z = 0
        for i in range(0, pos.shape[0]):
            x += atomic_masses[i] * pos[i][0]
            y += atomic_masses[i] * pos[i][1]
            z += atomic_masses[i] * pos[i][2]

        x = x / molecule_mass
        y = y / molecule_mass
        z = z / molecule_mass

        com = torch.Tensor([x, y, z])
        assert torch.allclose(com, torch.Tensor([0.0, 0.0, 0.0]), atol=1e-4)

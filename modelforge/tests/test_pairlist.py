import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_dataset_temp")
    return fn


def test_pairlist_logic():
    import torch

    # dummy data for illustration
    positions = torch.tensor(
        [
            [0.4933, 0.4460, 0.5762],
            [0.2340, 0.2053, 0.5025],
            [0.6566, 0.1263, 0.8792],
            [0.1656, 0.0338, 0.6708],
            [0.5696, 0.4790, 0.9622],
            [0.3499, 0.4241, 0.8818],
            [0.8400, 0.9389, 0.1888],
            [0.4983, 0.0793, 0.8639],
            [0.6605, 0.7567, 0.1938],
            [0.7725, 0.9758, 0.7063],
        ]
    )
    molecule_indices = torch.tensor(
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
    )  # molecule index for each atom

    # generate index grid
    n = len(molecule_indices)
    i_indices, j_indices = torch.triu_indices(n, n, 1)

    # filter pairs to only keep those belonging to the same molecule
    same_molecule_mask = molecule_indices[i_indices] == molecule_indices[j_indices]

    # Apply mask to get final pair indices
    i_final_pairs = i_indices[same_molecule_mask]
    j_final_pairs = j_indices[same_molecule_mask]

    # Concatenate to form final (2, n_pairs) tensor
    final_pair_indices = torch.stack((i_final_pairs, j_final_pairs))

    assert torch.allclose(
        final_pair_indices,
        torch.tensor([[0, 0, 1, 3, 5, 5, 6, 8], [1, 2, 2, 4, 6, 7, 7, 9]]),
    )

    # Create pair_coordinates tensor
    pair_coordinates = positions[final_pair_indices.T]
    pair_coordinates = pair_coordinates.view(-1, 2, 3)

    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )
    # Calculate distances
    distances = (pair_coordinates[:, 0, :] - pair_coordinates[:, 1, :]).norm(
        p=2, dim=-1
    )

    # Define a cutoff
    cutoff = 1.0

    # Find pairs within the cutoff
    in_cutoff = (distances <= cutoff).nonzero().squeeze()

    # Get the atom indices within the cutoff
    atom_pairs_withing_cutoff = final_pair_indices[:, in_cutoff]
    assert torch.allclose(
        atom_pairs_withing_cutoff,
        torch.tensor([[0, 0, 1, 3, 5, 5, 8], [1, 2, 2, 4, 6, 7, 9]]),
    )


def test_pairlist():
    import torch
    from collections import namedtuple
    from modelforge.potential.neighbors import NeighborListForTraining, Pairlist

    TestInput = namedtuple(
        "TestInput", ["positions", "atomic_subsystem_indices", "pair_list"]
    )

    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1])
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )
    from openff.units import unit

    cutoff = unit.Quantity(5.0, unit.nanometer).to(unit.nanometer).m
    nlist = NeighborListForTraining(cutoff, only_unique_pairs=True)

    r = nlist(TestInput(positions, atomic_subsystem_indices, None))
    pair_indices = r.pair_indices

    # pairlist describes the pairs of interacting atoms within a batch
    # that means for the pairlist provided below:
    # pair1: pairlist[0][0] and pairlist[1][0], i.e. (0,1)
    # pair2: pairlist[0][1] and pairlist[1][1], i.e. (0,2)
    # pair3: pairlist[0][2] and pairlist[1][2], i.e. (1,2)

    assert torch.allclose(
        pair_indices, torch.tensor([[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]])
    )
    # NOTE: pairs are defined on axis=1 and not axis=0
    assert torch.allclose(
        r.r_ij,
        torch.tensor(
            [
                [1.0, 1.0, 1.0],  # pair1, [1.0, 1.0, 1.0] - [0.0, 0.0, 0.0]
                [2.0, 2.0, 2.0],  # pair2, [2.0, 2.0, 2.0] - [0.0, 0.0, 0.0]
                [1.0, 1.0, 1.0],  # pair3, [3.0, 3.0, 3.0] - [0.0, 0.0, 0.0]
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )

    # test with cutoff
    cutoff = unit.Quantity(2.0, unit.nanometer).to(unit.nanometer).m
    nlist = NeighborListForTraining(cutoff, only_unique_pairs=True)
    r = nlist(TestInput(positions, atomic_subsystem_indices, None))
    pair_indices = r.pair_indices

    assert torch.equal(pair_indices, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))
    # pairs that are excluded through cutoff: (0,2) and (3,5)
    assert torch.equal(
        r.r_ij,
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )

    assert torch.allclose(
        r.d_ij, torch.tensor([1.7321, 1.7321, 1.7321, 1.7321]), atol=1e-3
    )

    # -------------------------------- #
    # test with complete pairlist
    cutoff = unit.Quantity(2.0, unit.nanometer).to(unit.nanometer).m
    neigborlist = NeighborListForTraining(cutoff, only_unique_pairs=False)
    r = neigborlist(TestInput(positions, atomic_subsystem_indices, None))
    pair_indices = r.pair_indices

    assert torch.equal(
        pair_indices, torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    )

    # -------------------------------- #
    # make sure that Pairlist and Neighborlist behave the same for large cutoffs
    cutoff = unit.Quantity(10.0, unit.nanometer).to(unit.nanometer).m
    only_unique_pairs = False
    neighborlist = NeighborListForTraining(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(TestInput(positions, atomic_subsystem_indices, None))
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # -------------------------------- #
    # make sure that they are the same also for non-redundant pairs
    cutoff = unit.Quantity(10.0, unit.nanometer).to(unit.nanometer).m
    only_unique_pairs = True
    neighborlist = NeighborListForTraining(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(TestInput(positions, atomic_subsystem_indices, None))
    neighbor_indices = r.pair_indices

    assert torch.equal(pair_indices, neighbor_indices)

    # -------------------------------- #
    # this should fail
    cutoff = unit.Quantity(2.0, unit.nanometer).to(unit.nanometer).m
    only_unique_pairs = True
    neighborlist = NeighborListForTraining(cutoff, only_unique_pairs=only_unique_pairs)
    pairlist = Pairlist(only_unique_pairs=only_unique_pairs)
    r = pairlist(positions, atomic_subsystem_indices)
    pair_indices = r.pair_indices
    r = neighborlist(TestInput(positions, atomic_subsystem_indices, None))
    neighbor_indices = r.pair_indices

    assert not pair_indices.shape == neighbor_indices.shape


def test_neighborlists_for_dimer():
    import torch
    from modelforge.utils.prop import NNPInput

    atomic_numbers = torch.tensor([17, 17], dtype=torch.int32)
    positions = torch.tensor(
        [[0.0000e00, 0.0000e00, 1.0118e-01], [1.2391e-17, 0.0000e00, -1.0118e-01]]
    )
    atomic_subsystem_indices = torch.tensor([0, 0], dtype=torch.int32)
    per_system_total_charge = torch.tensor([0], dtype=torch.int32)
    pair_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    test_system = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=positions,
        atomic_subsystem_indices=atomic_subsystem_indices,
        per_system_total_charge=per_system_total_charge,
        pair_list=pair_list,
    )

    from modelforge.potential.neighbors import (
        NeighborListForTraining,
        NeighborlistForInference,
        OrthogonalDisplacementFunction,
    )

    nlist_inf = NeighborlistForInference(
        cutoff=0.51,
        only_unique_pairs=True,
        displacement_function=OrthogonalDisplacementFunction(),
    )

    nlist_inf_output = nlist_inf.forward(test_system)
    assert nlist_inf_output.pair_indices.shape == (2, 1)
    assert nlist_inf_output.r_ij.shape == (1, 3)
    assert nlist_inf_output.d_ij.shape == (1, 1)

    nlist_train = NeighborListForTraining(cutoff=0.51, only_unique_pairs=True)

    nlist_train_output = nlist_train.forward(test_system)

    assert nlist_train_output.pair_indices.shape == (2, 1)
    assert nlist_train_output.r_ij.shape == (1, 3)
    assert nlist_train_output.d_ij.shape == (1, 1)

    nlist_inf = NeighborlistForInference(
        cutoff=0.51,
        only_unique_pairs=False,
        displacement_function=OrthogonalDisplacementFunction(),
    )

    nlist_inf_output = nlist_inf.forward(test_system)
    assert nlist_inf_output.pair_indices.shape == (2, 2)
    assert nlist_inf_output.r_ij.shape == (2, 3)
    assert nlist_inf_output.d_ij.shape == (2, 1)

    nlist_train = NeighborListForTraining(cutoff=0.51, only_unique_pairs=False)
    nlist_train_output = nlist_train.forward(test_system)
    assert nlist_train_output.pair_indices.shape == (2, 2)
    assert nlist_train_output.r_ij.shape == (2, 3)
    assert nlist_train_output.d_ij.shape == (2, 1)


def test_pairlist_precomputation():
    import numpy as np
    import torch

    from modelforge.potential.neighbors import Pairlist

    atomic_subsystem_indices = torch.tensor([0, 0, 0])

    pairlist = Pairlist()

    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 6)
    assert nr_pairs[0] == 6

    # 3 molecules, 3 atoms each
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 18)
    assert np.all(nr_pairs == [6, 6, 6])

    # 3 molecules, 3,4, and 5 atoms each
    atomic_subsystem_indices = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    pairs, nr_pairs = pairlist.construct_initial_pairlist_using_numpy(
        atomic_subsystem_indices.to("cpu")
    )

    assert pairs.shape == (2, 38)
    assert np.all(nr_pairs == [6, 12, 20])


def test_pairlist_on_dataset(datamodule_factory, prep_temp_dir, dataset_temp_dir):
    # Set up a dataset
    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    local_cache_dir = str(prep_temp_dir) + "/test_pairlist"
    dataset_cache_dir = str(dataset_temp_dir)

    dataset = datamodule_factory(
        dataset_name="QM9",
        batch_size=1,
        local_cache_dir=local_cache_dir,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        dataset_cache_dir=dataset_cache_dir,
    )

    # -------------------------------#
    # -------------------------------#
    # get methane input
    batch = next(iter(dataset.train_dataloader(shuffle=False))).nnp_input
    import torch

    # make sure that the pairlist of methane is correct (single molecule)
    assert torch.equal(
        batch.pair_list,
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
            ]
        ),
    )

    # test that the pairlist of 2 molecules is correct (which can then be expected also to be true for N molecules)
    dataset = datamodule_factory(
        dataset_name="QM9",
        batch_size=2,
        local_cache_dir=local_cache_dir,
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        dataset_cache_dir=dataset_cache_dir,
    )

    # -------------------------------#
    # -------------------------------#
    # get methane input
    batch = next(iter(dataset.train_dataloader(shuffle=False))).nnp_input

    assert torch.equal(
        batch.pair_list,
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
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
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
                    6,
                    7,
                    8,
                    5,
                    7,
                    8,
                    5,
                    6,
                    8,
                    5,
                    6,
                    7,
                ],
            ]
        ),
    )

    # check that the pairlist maximum value for i is the number of atoms in the batch
    assert (
        int(batch.pair_list[0][-1].item()) + 1 == 8 + 1 == len(batch.atomic_numbers)
    )  # +1 because of 0-based indexing


def test_displacement_function():
    """Test that OrthogonalDisplacementFunction behaves as expected, including toggling periodicity"""
    import torch

    from modelforge.potential.neighbors import OrthogonalDisplacementFunction

    displacement_function = OrthogonalDisplacementFunction()

    box_vectors = torch.tensor(
        [[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=torch.float32
    )

    coords1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0, 0, 0],
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [8.5, 8.5, 8.5],
        ],
        dtype=torch.float32,
    )

    coords2 = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1.0, 1.0, 1.0],
            [0, 0, 0],
            [8.5, 8.5, 8.5],
            [3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    r_ij, d_ij = displacement_function(coords1, coords1, box_vectors, is_periodic=True)

    assert torch.allclose(r_ij, torch.zeros_like(r_ij))
    assert torch.allclose(d_ij, torch.zeros_like(d_ij))

    r_ij, d_ij = displacement_function(coords1, coords2, box_vectors, is_periodic=True)

    assert torch.allclose(
        r_ij,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                [4.5, 4.5, 4.5],
                [-4.5, -4.5, -4.5],
            ],
            dtype=r_ij.dtype,
        ),
    )

    assert torch.allclose(
        d_ij,
        torch.tensor(
            [[1.0], [1.0], [1.0], [1.7321], [1.7321], [7.7942], [7.7942]],
            dtype=d_ij.dtype,
        ),
        atol=1e-4,
    )
    # make sure the function works if the box is not periodic
    displacement_function = OrthogonalDisplacementFunction()
    r_ij, d_ij = displacement_function(coords1, coords1, box_vectors, is_periodic=False)

    assert torch.allclose(r_ij, torch.zeros_like(r_ij))
    assert torch.allclose(d_ij, torch.zeros_like(d_ij))

    r_ij, d_ij = displacement_function(coords1, coords2, box_vectors, is_periodic=False)

    # since the
    assert torch.allclose(r_ij, coords1 - coords2)
    assert torch.allclose(d_ij, torch.norm(r_ij, dim=1, keepdim=True, p=2))


def test_inference_neighborlist_building():
    """Test that NeighborlistBruteNsq and NeighborlistVerletNsq behave identically when building the neighborlist"""
    from modelforge.potential.neighbors import (
        NeighborlistForInference,
        OrthogonalDisplacementFunction,
    )
    import torch

    from modelforge.dataset.dataset import NNPInput

    displacement_function = OrthogonalDisplacementFunction()

    positions = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [3.0, 0, 0], [8, 0, 0]], dtype=torch.float32
    )

    data = NNPInput(
        atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        positions=positions,
        atomic_subsystem_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        per_system_total_charge=torch.tensor([0.0], dtype=torch.float32),
        box_vectors=torch.tensor(
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=torch.float32
        ),
        is_periodic=True,
    )
    # test brute force neighborlist
    nlist = NeighborlistForInference(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=False
    )
    nlist._set_strategy("brute_nsq")
    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 12

    nlist_verlet = NeighborlistForInference(
        cutoff=5.0,
        displacement_function=displacement_function,
        only_unique_pairs=False,
    )
    nlist_verlet._set_strategy("verlet_nsq", skin=0.5)

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    nlist = NeighborlistForInference(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=True
    )

    nlist._set_strategy("brute_nsq")
    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 6

    nlist_verlet = NeighborlistForInference(
        cutoff=5.0,
        displacement_function=displacement_function,
        only_unique_pairs=True,
    )
    nlist_verlet._set_strategy("verlet_nsq", skin=0.5)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    nlist = NeighborlistForInference(
        cutoff=3.5, displacement_function=displacement_function, only_unique_pairs=False
    )
    nlist._set_strategy("brute_nsq")
    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 10

    assert torch.all(d_ij <= 3.5)

    assert torch.all(
        pairs
        == torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 3, 2, 3], [1, 2, 3, 2, 3, 0, 0, 0, 1, 1]]
        )
    )

    assert torch.allclose(
        r_ij,
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [1.0, -0.0, -0.0],
                [3.0, -0.0, -0.0],
                [-2.0, -0.0, -0.0],
                [2.0, -0.0, -0.0],
                [-3.0, -0.0, -0.0],
            ]
        ),
    )

    assert torch.allclose(
        d_ij,
        torch.tensor(
            [[1.0], [3.0], [2.0], [2.0], [3.0], [1.0], [3.0], [2.0], [2.0], [3.0]]
        ),
    )

    nlist_verlet = NeighborlistForInference(
        cutoff=3.5,
        displacement_function=displacement_function,
        only_unique_pairs=False,
    )

    nlist_verlet._set_strategy("verlet_nsq", skin=0.5)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    displacement_function = OrthogonalDisplacementFunction()

    nlist = NeighborlistForInference(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=False
    )
    nlist._set_strategy("brute_nsq")
    data.is_periodic = False

    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 8
    assert torch.all(d_ij <= 5.0)

    nlist_verlet = NeighborlistForInference(
        cutoff=5.0,
        displacement_function=displacement_function,
        only_unique_pairs=False,
    )
    nlist_verlet._set_strategy("verlet_nsq", skin=0.5)

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # test updates to verlet list

    positions = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [3.0, 0, 0], [8, 0, 0]], dtype=torch.float32
    )


def test_verlet_inference():
    """Test to ensure that the verlet neighborlist properly updates by comparing to brute force neighborlist"""
    from modelforge.potential.neighbors import (
        NeighborlistForInference,
        OrthogonalDisplacementFunction,
    )
    import torch

    from modelforge.dataset.dataset import NNPInput

    def return_data(positions, box_length=10, is_periodic=True):
        return NNPInput(
            atomic_numbers=torch.ones(positions.shape[0], dtype=torch.int64),
            positions=positions,
            atomic_subsystem_indices=torch.zeros(positions.shape[0], dtype=torch.int64),
            per_system_total_charge=torch.tensor([0.0], dtype=torch.float32),
            box_vectors=torch.tensor(
                [[box_length, 0, 0], [0, box_length, 0], [0, 0, box_length]],
                dtype=torch.float32,
            ),
            is_periodic=is_periodic,
        )

    positions = torch.tensor(
        [[2.0, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    displacement_function = OrthogonalDisplacementFunction()
    nlist_verlet = NeighborlistForInference(
        cutoff=1.5,
        displacement_function=displacement_function,
        only_unique_pairs=True,
    )
    nlist_verlet._set_strategy("verlet_nsq", skin=0.5)

    nlist_brute = NeighborlistForInference(
        cutoff=1.5,
        displacement_function=displacement_function,
        only_unique_pairs=True,
    )
    nlist_brute._set_strategy("brute_nsq")

    pairs, d_ij, r_ij = nlist_brute(data)

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 1
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # move one particle father away, but still interacting
    positions = torch.tensor(
        [[2.2, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    # since we didn't move far enough to trigger a rebuild of the verlet list, the results should be the same
    assert nlist_verlet.builds == 1
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # move one particle father away, but still interacting, but enough to trigger a rebuild,
    # since rebuild 0.5*skin = 0.25
    positions = torch.tensor(
        [[2.3, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 2
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # move one particle farther away so it no longer interacts; but less than 0.5*skin = 0.25 since last rebuild,
    # so no rebuilding will occur
    positions = torch.tensor(
        [[2.51, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 2
    assert pairs.shape[1] == 1
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # move the particle back such that it is interacting, but less than half the skin, so no rebuild
    positions = torch.tensor(
        [[2.45, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 2
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # force a rebuild by changing box_vectors
    positions = torch.tensor(
        [[2.45, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions, 9)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 3
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # force rebuild by changing number of particles; but let's add a particle that doesn't interact
    positions = torch.tensor(
        [[2.45, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0], [4, 0, 0]], dtype=torch.float32
    )
    data = return_data(positions, 9)

    pairs, d_ij, r_ij = nlist_brute(data)
    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)

    assert nlist_verlet.builds == 4
    assert pairs.shape[1] == 2
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

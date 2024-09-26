import pytest
import torch

from modelforge.potential.neighbors import OrthogonalDisplacementFunction


def test_displacement_function():
    """Test that OrthogonalDisplacementFunction behaves as expected, including toggling periodicity"""
    from modelforge.potential.neighbors import OrthogonalDisplacementFunction

    displacement_function = OrthogonalDisplacementFunction(periodic=True)

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
    r_ij, d_ij = displacement_function(coords1, coords1, box_vectors)

    assert torch.allclose(r_ij, torch.zeros_like(r_ij))
    assert torch.allclose(d_ij, torch.zeros_like(d_ij))

    r_ij, d_ij = displacement_function(coords1, coords2, box_vectors)

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
    displacement_function = OrthogonalDisplacementFunction(periodic=False)
    r_ij, d_ij = displacement_function(coords1, coords1, box_vectors)

    assert torch.allclose(r_ij, torch.zeros_like(r_ij))
    assert torch.allclose(d_ij, torch.zeros_like(d_ij))

    r_ij, d_ij = displacement_function(coords1, coords2, box_vectors)

    # since the
    assert torch.allclose(r_ij, coords1 - coords2)
    assert torch.allclose(d_ij, torch.norm(r_ij, dim=1, keepdim=True, p=2))


def test_inference_neighborlist_building():
    """Test that NeighborlistBruteNsq and NeighborlistVerletNsq behave identically when building the neighborlist"""
    from modelforge.potential.neighbors import (
        NeighborlistBruteNsq,
        NeighborlistVerletNsq,
        OrthogonalDisplacementFunction,
    )
    from modelforge.dataset.dataset import NNPInput

    displacement_function = OrthogonalDisplacementFunction(periodic=True)

    positions = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [3.0, 0, 0], [8, 0, 0]], dtype=torch.float32
    )

    data = NNPInput(
        atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        positions=positions,
        atomic_subsystem_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        total_charge=torch.tensor([0.0], dtype=torch.float32),
        box_vectors=torch.tensor(
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=torch.float32
        ),
    )
    # test to
    nlist = NeighborlistBruteNsq(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=False
    )
    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 12

    nlist_verlet = NeighborlistVerletNsq(
        cutoff=5.0,
        displacement_function=displacement_function,
        skin=0.5,
        only_unique_pairs=False,
    )

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    nlist = NeighborlistBruteNsq(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=True
    )

    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 6

    nlist_verlet = NeighborlistVerletNsq(
        cutoff=5.0,
        displacement_function=displacement_function,
        skin=0.5,
        only_unique_pairs=True,
    )

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    nlist = NeighborlistBruteNsq(
        cutoff=3.5, displacement_function=displacement_function, only_unique_pairs=False
    )

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

    nlist_verlet = NeighborlistVerletNsq(
        cutoff=3.5,
        displacement_function=displacement_function,
        skin=0.5,
        only_unique_pairs=False,
    )

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    displacement_function = OrthogonalDisplacementFunction(periodic=False)

    nlist = NeighborlistBruteNsq(
        cutoff=5.0, displacement_function=displacement_function, only_unique_pairs=False
    )

    pairs, d_ij, r_ij = nlist(data)

    assert pairs.shape[1] == 8
    assert torch.all(d_ij <= 5.0)

    nlist_verlet = NeighborlistVerletNsq(
        cutoff=5.0,
        displacement_function=displacement_function,
        skin=0.5,
        only_unique_pairs=False,
    )

    pairs_v, d_ij_v, r_ij_v = nlist_verlet(data)
    assert pairs_v.shape[1] == pairs.shape[1]
    assert torch.all(pairs_v == pairs)
    assert torch.allclose(d_ij_v, d_ij)
    assert torch.allclose(r_ij_v, r_ij)

    # test updates to verlet list

    positions = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [3.0, 0, 0], [8, 0, 0]], dtype=torch.float32
    )

    data = NNPInput(
        atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.int64),
        positions=positions,
        atomic_subsystem_indices=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        total_charge=torch.tensor([0.0], dtype=torch.float32),
        box_vectors=torch.tensor(
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=torch.float32
        ),
    )

    displacement_function = OrthogonalDisplacementFunction(periodic=True)
    nlist = NeighborlistBruteNsq(
        cutoff=3.0, displacement_function=displacement_function, only_unique_pairs=True
    )


def test_verlet_inference():
    """Test to ensure that the verlet neighborlist properly updates by comparing to brute force neighborlist"""
    from modelforge.potential.neighbors import (
        NeighborlistBruteNsq,
        NeighborlistVerletNsq,
        OrthogonalDisplacementFunction,
    )
    from modelforge.dataset.dataset import NNPInput

    def return_data(positions, box_length=10):
        return NNPInput(
            atomic_numbers=torch.ones(positions.shape[0], dtype=torch.int64),
            positions=positions,
            atomic_subsystem_indices=torch.zeros(positions.shape[0], dtype=torch.int64),
            total_charge=torch.tensor([0.0], dtype=torch.float32),
            box_vectors=torch.tensor(
                [[box_length, 0, 0], [0, box_length, 0], [0, 0, box_length]],
                dtype=torch.float32,
            ),
        )

    positions = torch.tensor(
        [[2.0, 0, 0], [1.0, 0, 0], [0.0, 0.0, 0]], dtype=torch.float32
    )
    data = return_data(positions)

    displacement_function = OrthogonalDisplacementFunction(periodic=True)
    nlist_verlet = NeighborlistVerletNsq(
        cutoff=1.5,
        displacement_function=displacement_function,
        skin=0.5,
        only_unique_pairs=True,
    )

    nlist_brute = NeighborlistBruteNsq(
        cutoff=1.5,
        displacement_function=displacement_function,
        only_unique_pairs=True,
    )

    print("first check")
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

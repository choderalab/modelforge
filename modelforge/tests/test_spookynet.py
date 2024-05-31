from modelforge.potential.spookynet import SpookyNet
from spookynet import SpookyNet as RefSpookyNet
import torch

import pytest


def test_spookynet_init():
    """Test initialization of the SpookyNet model."""

    spookynet = SpookyNet()
    assert spookynet is not None, "SpookyNet model should be initialized."


from openff.units import unit


@pytest.mark.parametrize(
    "model_parameter",
    (
            [64, 50, 20, unit.Quantity(5.0, unit.angstrom), 2],
            [32, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
            [128, 120, 64, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_spookynet_forward(single_batch_with_batchsize_64, model_parameter):
    """
    Test the forward pass of the SpookyNet model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        number_of_gaussians,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    spookynet = SpookyNet(
        number_of_atom_features=nr_atom_basis,
        max_Z=max_atomic_number,
        number_of_radial_basis_functions=number_of_gaussians,
        cutoff=cutoff,
        number_of_interaction_modules=nr_interaction_blocks,
    )
    energy = spookynet(single_batch_with_batchsize_64.nnp_input).E
    nr_of_mols = single_batch_with_batchsize_64.nnp_input.atomic_subsystem_indices.unique().shape[
        0
    ]

    assert (
            len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def make_random_pairlist(nr_atoms, nr_pairs, include_self_pairs):
    if include_self_pairs:
        nr_pairs_choose = nr_pairs - nr_atoms
        assert nr_pairs_choose >= 0, """Number of pairs must be greater than or equal to the number of atoms if "
            include_self_pairs is True."""

    else:
        nr_pairs_choose = nr_pairs

    all_pairs = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    self_pairs = all_pairs.T[0] == all_pairs.T[1]
    non_self_pairs = all_pairs[~self_pairs]
    perm = torch.randperm(non_self_pairs.size(0))
    idx = perm[:nr_pairs_choose]
    pairlist = non_self_pairs[idx]
    if include_self_pairs:
        pairlist = torch.cat(
            [pairlist, all_pairs[self_pairs]], dim=0
        )

    return pairlist.T


def test_atomic_properties_static():
    ref_spookynet = RefSpookyNet()

    nr_atoms = 5
    geometry_basis = 3
    nr_pairs = 7
    idx_i, idx_j = make_random_pairlist(nr_atoms, nr_pairs, False)

    Z = torch.randint(1, 100, (nr_atoms,))
    R = torch.rand((nr_atoms, geometry_basis))
    print(ref_spookynet._atomic_properties_static(Z, R, idx_i, idx_j))

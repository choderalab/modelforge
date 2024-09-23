import os
from sys import platform

import jax.numpy as jnp
import jax.random
import numpy as onp
import pytest
import sake as reference_sake
import torch

from modelforge.potential.sake import SAKEInteraction
from modelforge.tests.helper_functions import setup_potential_for_test

ON_MAC = platform == "darwin"


def test_init():
    """Test initialization of the SAKE neural network potential."""

    sake = setup_potential_for_test("sake", "training")

    assert sake is not None, "SAKE model should be initialized."


from openff.units import unit


def test_forward(single_batch_with_batchsize):
    """
    Test the forward pass of the SAKE model.
    """
    # get methane input
    batch = single_batch_with_batchsize(batch_size=64, dataset_name="QM9")
    methane = batch.nnp_input

    sake = setup_potential_for_test("sake", "training")
    energy = sake(methane)["per_molecule_energy"]
    nr_of_mols = methane.atomic_subsystem_indices.unique().shape[0]

    assert (
        len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_interaction_forward():
    nr_atoms = 41
    nr_atom_basis = 47
    geometry_basis = 3
    sake_block = SAKEInteraction(
        nr_atom_basis=nr_atom_basis,
        nr_edge_basis=37,
        nr_edge_basis_hidden=5,
        nr_atom_basis_hidden=7,
        nr_atom_basis_spatial_hidden=13,
        nr_atom_basis_spatial=17,
        nr_atom_basis_velocity=19,
        nr_coefficients=23,
        nr_heads=29,
        activation=torch.nn.ReLU(),
        maximum_interaction_radius=(5.0 * unit.angstrom).to(unit.nanometer).m,
        number_of_radial_basis_functions=53,
        epsilon=1e-5,
        scale_factor=(1.0 * unit.nanometer).m,
    )
    h = torch.randn(nr_atoms, nr_atom_basis)
    x = torch.randn(nr_atoms, geometry_basis)
    v = torch.randn(nr_atoms, geometry_basis)
    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    nr_pairs = 43
    edge_mask = onp.random.choice(len(pairlist), nr_pairs, replace=False)
    pairlist = pairlist[edge_mask].T
    sake_block(h, x, v, pairlist)


@pytest.mark.parametrize("eq_atol", [3e-1])
@pytest.mark.parametrize("h_atol", [8e-2])
def test_layer_equivariance(h_atol, eq_atol, single_batch_with_batchsize):
    from dataclasses import replace

    import torch

    # Model parameters
    torch.manual_seed(1884)

    # define a rotation matrix in 3D that rotates by 90 degrees around the
    # z-axis (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sake = setup_potential_for_test("sake", "training")

    # get methane input
    batch = single_batch_with_batchsize(batch_size=64, dataset_name="QM9")

    nnp_input = batch.nnp_input
    perturbed_nnp_input = replace(nnp_input)
    perturbed_nnp_input.positions = torch.matmul(nnp_input.positions, rotation_matrix)

    # prepare reference and perturbed inputs
    neighborlist = sake.neighborlist(nnp_input)
    reference_v_torch = torch.randn_like(nnp_input.positions)

    perturbed_v_torch = torch.matmul(reference_v_torch, rotation_matrix)

    emedding = torch.nn.Embedding(101, 11)
    atomic_embedding = emedding(nnp_input.atomic_numbers)

    (
        reference_h_out_torch,
        reference_x_out_torch,
        reference_v_out_torch,
    ) = sake.core_network.interaction_modules[0](
        atomic_embedding,
        nnp_input.positions,
        reference_v_torch,
        neighborlist.pair_indices,
    )
    (
        perturbed_h_out_torch,
        perturbed_x_out_torch,
        perturbed_v_out_torch,
    ) = sake.core_network.interaction_modules[0](
        atomic_embedding,
        perturbed_nnp_input.positions,
        perturbed_v_torch,
        neighborlist.pair_indices,
    )

    # x and v are equivariant, h is invariant
    assert torch.allclose(reference_h_out_torch, perturbed_h_out_torch, atol=h_atol)
    assert torch.allclose(
        torch.matmul(reference_x_out_torch, rotation_matrix),
        perturbed_x_out_torch,
        atol=eq_atol,
    )
    assert torch.allclose(
        torch.matmul(reference_v_out_torch, rotation_matrix),
        perturbed_v_out_torch,
        atol=eq_atol,
    )


def make_reference_equivalent_sake_interaction(out_features, hidden_features, nr_heads):
    radial_max_distance = unit.Quantity(5.0, unit.angstrom)
    # Define the modelforge layer
    mf_sake_block = SAKEInteraction(
        nr_atom_basis=out_features,
        nr_edge_basis=hidden_features,
        nr_edge_basis_hidden=hidden_features,
        nr_atom_basis_hidden=hidden_features,
        nr_atom_basis_spatial_hidden=hidden_features,
        nr_atom_basis_spatial=hidden_features,
        nr_atom_basis_velocity=hidden_features,
        nr_coefficients=(nr_heads * hidden_features),
        nr_heads=nr_heads,
        activation=torch.nn.SiLU(),
        maximum_interaction_radius=radial_max_distance.to(unit.nanometer).m,
        number_of_radial_basis_functions=50,
        epsilon=1e-5,
        scale_factor=unit.Quantity(1.0, unit.nanometer).to(unit.nanometer).m,
    )

    # Define the reference layer
    ref_sake_interaction = reference_sake.layers.DenseSAKELayer(
        out_features=out_features,
        hidden_features=hidden_features,
        n_heads=nr_heads,
        cutoff=None,
    )

    return mf_sake_block, ref_sake_interaction


def make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs):
    all_pairs = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    self_pairs = all_pairs.T[0] == all_pairs.T[1]
    non_self_pairs = all_pairs[~self_pairs]
    non_self_pairs_jax = jnp.array(onp.array(non_self_pairs))
    if include_self_pairs:
        nr_pairs_choose = nr_pairs - nr_atoms
        assert nr_pairs_choose >= 0, (
            "Number of pairs must be greater than or equal to the number of atoms if "
            "include_self_pairs is True."
        )
    else:
        nr_pairs_choose = nr_pairs
    pairlist_jax = jax.random.choice(
        key, non_self_pairs_jax, (nr_pairs_choose,), replace=False
    ).T
    if include_self_pairs:
        pairlist_jax = jnp.concatenate(
            [pairlist_jax, jnp.array(onp.array(all_pairs[self_pairs].T))], axis=1
        )
    pairlist = torch.tensor(onp.array(pairlist_jax), dtype=torch.int64)
    mask = jnp.zeros((nr_atoms, nr_atoms))
    for i in range(nr_pairs):
        mask = mask.at[pairlist_jax[0, i], pairlist_jax[1, i]].set(1)
    return pairlist, mask


def test_radial_symmetry_function_against_reference():
    from sake.utils import ExpNormalSmearing as RefExpNormalSmearing

    from modelforge.potential.utils import PhysNetRadialBasisFunction

    nr_atoms = 1
    number_of_radial_basis_functions = 10
    cutoff_upper = unit.Quantity(6.0, unit.nanometer)
    cutoff_lower = unit.Quantity(2.0, unit.nanometer)

    radial_symmetry_function_module = PhysNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=cutoff_upper.to(unit.nanometer).m,
        min_distance=cutoff_lower.to(unit.nanometer).m,
        dtype=torch.float32,
    )
    ref_radial_basis_module = RefExpNormalSmearing(
        num_rbf=number_of_radial_basis_functions,
        cutoff_upper=cutoff_upper.to(unit.nanometer).m,
        cutoff_lower=cutoff_lower.to(unit.nanometer).m,
    )
    key = jax.random.PRNGKey(1884)

    # Generate random input data in JAX
    d_ij_jax = jax.random.uniform(key, (nr_atoms, nr_atoms, 1))
    d_ij = torch.from_numpy(onp.array(d_ij_jax)).reshape((nr_atoms**2, 1))

    mf_rbf = radial_symmetry_function_module(d_ij)
    variables = ref_radial_basis_module.init(key, d_ij_jax)

    assert torch.allclose(
        torch.from_numpy(onp.array(variables["params"]["means"])),
        radial_symmetry_function_module.radial_basis_centers.detach().T,
        atol=1e-1,
        rtol=1e-1,
    )
    assert torch.allclose(
        torch.from_numpy(onp.array(variables["params"]["betas"])) ** -0.5,
        radial_symmetry_function_module.radial_scale_factor.detach().T,
        atol=1e-2,
        rtol=1e-2,
    )

    ref_rbf = ref_radial_basis_module.apply(variables, d_ij_jax)

    assert torch.allclose(
        mf_rbf,
        torch.from_numpy(onp.array(ref_rbf)).reshape(
            nr_atoms**2, number_of_radial_basis_functions
        ),
    )


@pytest.mark.skipif(ON_MAC, reason="Test fails on macOS")
@pytest.mark.parametrize("include_self_pairs", [True, False])
@pytest.mark.parametrize("v_is_none", [True, False])
def test_sake_layer_against_reference(include_self_pairs, v_is_none):
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_atom_basis = out_features
    nr_pairs = 17
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

    pairlist, mask = make_equivalent_pairlist_mask(
        key, nr_atoms, nr_pairs, include_self_pairs=include_self_pairs
    )

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(
        out_features, hidden_features, nr_heads
    )
    # Generate random input data in JAX
    h_key, x_key, v_key, init_key = jax.random.split(key, 4)
    h_jax = jax.random.normal(h_key, (nr_atoms, nr_atom_basis))
    x_jax = jax.random.normal(x_key, (nr_atoms, geometry_basis))
    if v_is_none:
        v_jax = None
        v = torch.zeros((nr_atoms, geometry_basis))
    else:
        v_jax = jax.random.normal(v_key, (nr_atoms, geometry_basis))
        v = torch.from_numpy(onp.array(v_jax))

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = torch.from_numpy(onp.array(h_jax))
    x = torch.from_numpy(onp.array(x_jax))

    variables = ref_sake_interaction.init(init_key, h_jax, x_jax, v_jax, mask)
    layer = variables["params"]

    assert torch.allclose(
        torch.from_numpy(onp.array(layer["edge_model"]["kernel"]["betas"]) ** -0.5),
        mf_sake_block.radial_symmetry_function_module.radial_scale_factor.detach().T,
    )
    assert torch.allclose(
        torch.from_numpy(onp.array(layer["edge_model"]["kernel"]["means"])),
        mf_sake_block.radial_symmetry_function_module.radial_basis_centers.detach().T,
    )
    layer["edge_model"]["mlp_in"]["bias"] = (
        mf_sake_block.edge_mlp_in.bias.detach().numpy().T
    )
    layer["edge_model"]["mlp_in"]["kernel"] = (
        mf_sake_block.edge_mlp_in.weight.detach().numpy().T
    )
    layer["edge_model"]["mlp_out"]["layers_0"]["bias"] = (
        mf_sake_block.edge_mlp_out[0].bias.detach().numpy().T
    )
    layer["edge_model"]["mlp_out"]["layers_0"]["kernel"] = (
        mf_sake_block.edge_mlp_out[0].weight.detach().numpy().T
    )
    layer["edge_model"]["mlp_out"]["layers_2"]["bias"] = (
        mf_sake_block.edge_mlp_out[1].bias.detach().numpy().T
    )
    layer["edge_model"]["mlp_out"]["layers_2"]["kernel"] = (
        mf_sake_block.edge_mlp_out[1].weight.detach().numpy().T
    )
    layer["node_mlp"]["layers_0"]["bias"] = (
        mf_sake_block.node_mlp[0].bias.detach().numpy().T
    )
    layer["node_mlp"]["layers_0"]["kernel"] = (
        mf_sake_block.node_mlp[0].weight.detach().numpy().T
    )
    layer["node_mlp"]["layers_2"]["bias"] = (
        mf_sake_block.node_mlp[1].bias.detach().numpy().T
    )
    layer["node_mlp"]["layers_2"]["kernel"] = (
        mf_sake_block.node_mlp[1].weight.detach().numpy().T
    )
    layer["post_norm_mlp"]["layers_0"]["bias"] = (
        mf_sake_block.post_norm_mlp[0].bias.detach().numpy().T
    )
    layer["post_norm_mlp"]["layers_0"]["kernel"] = (
        mf_sake_block.post_norm_mlp[0].weight.detach().numpy().T
    )
    layer["post_norm_mlp"]["layers_2"]["bias"] = (
        mf_sake_block.post_norm_mlp[1].bias.detach().numpy().T
    )
    layer["post_norm_mlp"]["layers_2"]["kernel"] = (
        mf_sake_block.post_norm_mlp[1].weight.detach().numpy().T
    )
    layer["semantic_attention_mlp"]["layers_0"]["bias"] = (
        mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    )
    layer["semantic_attention_mlp"]["layers_0"]["kernel"] = (
        mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    )

    if not v_is_none:
        layer["velocity_mlp"]["layers_0"]["kernel"] = (
            mf_sake_block.velocity_mlp[0].weight.detach().numpy().T
        )
        layer["velocity_mlp"]["layers_0"]["bias"] = (
            mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
        )
        layer["velocity_mlp"]["layers_2"]["kernel"] = (
            mf_sake_block.velocity_mlp[1].weight.detach().numpy().T
        )
    layer["v_mixing"]["kernel"] = mf_sake_block.v_mixing_mlp.weight.detach().numpy().T
    layer["x_mixing"]["layers_0"]["kernel"] = (
        mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    )

    mf_h, mf_x, mf_v = mf_sake_block(h, x, v, pairlist)

    ref_h, ref_x, ref_v = ref_sake_interaction.apply(
        variables, h_jax, x_jax, v_jax, mask
    )

    ref_h_is_nan = torch.from_numpy(onp.isnan(ref_h))
    ref_x_is_nan = torch.from_numpy(onp.isnan(ref_x))
    ref_v_is_nan = torch.from_numpy(onp.isnan(ref_v))

    assert torch.allclose(
        torch.nan_to_num(mf_h, nan=0.0) * ~ref_h_is_nan,
        torch.nan_to_num(torch.from_numpy(onp.array(ref_h)), nan=0.0),
    )
    assert torch.allclose(
        torch.nan_to_num(mf_x, nan=0.0) * ~ref_x_is_nan,
        torch.nan_to_num(torch.from_numpy(onp.array(ref_x)), nan=0.0),
    )
    assert torch.allclose(
        torch.nan_to_num(mf_v, nan=0.0) * ~ref_v_is_nan,
        torch.nan_to_num(torch.from_numpy(onp.array(ref_v)), nan=0.0),
    )


import pytest


def test_model_invariance(single_batch_with_batchsize):
    from dataclasses import replace

    sake = setup_potential_for_test("sake", "training")
    # get methane input
    batch = single_batch_with_batchsize(batch_size=1, dataset_name="QM9")
    methane = batch.nnp_input

    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    perturbed_methane_input = replace(methane)
    perturbed_methane_input.positions = torch.matmul(methane.positions, rotation_matrix)

    reference_out = sake(methane)
    perturbed_out = sake(perturbed_methane_input)

    assert torch.allclose(
        reference_out["per_molecule_energy"], perturbed_out["per_molecule_energy"]
    )

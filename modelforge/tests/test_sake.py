import os

import jax.random
import jax.numpy as jnp
import pytest
import torch
import numpy as onp
from jax import Array
import flax

from modelforge.potential.sake import SAKE, SAKEInteraction
import sake as reference_sake

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_SAKE_init():
    """Test initialization of the SAKE neural network potential."""

    sake = SAKE()
    assert sake is not None, "SAKE model should be initialized."


from openff.units import unit


def test_sake_forward():
    """
    Test the forward pass of the SAKE model.
    """
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # Set up dataset
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    sake = SAKE()
    print(list(p.dtype for p in sake.parameters()))
    energy = sake(methane).E
    nr_of_mols = methane.atomic_subsystem_indices.unique().shape[0]

    assert (
            len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_sake_interaction_forward():
    from modelforge.potential.sake import ExpNormalSmearing
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
        radial_basis_module=ExpNormalSmearing(0.0, 5.0, n_rbf=50),
        epsilon=1e-5
    )
    h = torch.randn(nr_atoms, nr_atom_basis)
    x = torch.randn(nr_atoms, geometry_basis)
    v = torch.randn(nr_atoms, geometry_basis)
    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    nr_pairs = 43
    edge_mask = onp.random.choice(len(pairlist), nr_pairs, replace=False)
    pairlist = pairlist[edge_mask].T
    sake_block(h, x, v, pairlist)


@pytest.mark.parametrize("atol", [1e0])
def test_sake_interaction_equivariance(atol):
    import torch
    from modelforge.potential.sake import SAKE
    from dataclasses import replace

    # Model parameters
    nr_atom_basis = 11
    hidden_features = 7
    nr_heads = 2
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sake = SAKE(number_of_atom_features=nr_atom_basis)  # only for preparing inputs

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    # Set up dataset
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input
    perturbed_methane_input = replace(methane)
    perturbed_methane_input.positions = torch.matmul(
        methane.positions, rotation_matrix
    )

    # prepare reference and perturbed inputs
    reference_prepared_input = sake.prepare_inputs(methane, only_unique_pairs=False)
    reference_v_torch = torch.randn_like(reference_prepared_input.positions)

    perturbed_prepared_input = sake.prepare_inputs(perturbed_methane_input)
    perturbed_v_torch = torch.matmul(reference_v_torch, rotation_matrix)


    reference_h_out_torch, reference_x_out_torch, reference_v_out_torch = sake.interaction_modules[0](
        reference_prepared_input.atomic_embedding,
        reference_prepared_input.positions,
        reference_v_torch,
        reference_prepared_input.pair_indices
    )
    perturbed_h_out_torch, perturbed_x_out_torch, perturbed_v_out_torch = sake.interaction_modules[0](
        perturbed_prepared_input.atomic_embedding,
        perturbed_prepared_input.positions,
        perturbed_v_torch,
        perturbed_prepared_input.pair_indices
    )

    # x and v are equivariant, h is invariant
    assert torch.allclose(reference_h_out_torch, perturbed_h_out_torch, atol=atol)
    assert torch.allclose(torch.matmul(reference_x_out_torch, rotation_matrix), perturbed_x_out_torch, atol=atol)
    assert torch.allclose(torch.matmul(reference_v_out_torch, rotation_matrix), perturbed_v_out_torch, atol=atol)


def make_reference_equivalent_sake_interaction(out_features, hidden_features, nr_heads):
    from modelforge.potential.sake import ExpNormalSmearing

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
        radial_basis_module=ExpNormalSmearing(0.0, 5.0, n_rbf=50),
        epsilon=1e-5
    )

    # Define the reference layer
    ref_sake_interaction = reference_sake.layers.DenseSAKELayer(out_features=out_features,
                                                                hidden_features=hidden_features,
                                                                n_heads=nr_heads,
                                                                cutoff=None,
                                                                )

    return mf_sake_block, ref_sake_interaction


def test_x_minus_xt_against_reference():
    from sake.functional import get_x_minus_xt, get_x_minus_xt_norm
    nr_atoms = 13
    geometry_basis = 3
    key = jax.random.PRNGKey(1884)

    x_jax = jax.random.normal(key, (nr_atoms, geometry_basis))
    x_minus_xt = get_x_minus_xt(x_jax)

    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    pairlist = pairlist.T
    idx_i, idx_j = pairlist

    x = torch.from_numpy(onp.array(x_jax))
    r_ij = x[idx_j] - x[idx_i]

    assert torch.allclose(torch.from_numpy(onp.array(x_minus_xt.reshape(nr_atoms ** 2, geometry_basis, order='C'))),
                          r_ij)

    x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)
    d_ij = torch.sqrt((r_ij ** 2).sum(dim=1) + 1e-5)

    # Fortran and C ordering give the same result since x_minus_xt_norm is symmetric
    assert jnp.array_equal(x_minus_xt_norm, jnp.transpose(x_minus_xt_norm, (1, 0, 2)))
    assert torch.allclose(torch.from_numpy(onp.array(x_minus_xt_norm.reshape(nr_atoms ** 2, order='F'))), d_ij)
    assert torch.allclose(torch.from_numpy(onp.array(x_minus_xt_norm.reshape(nr_atoms ** 2, order='C'))), d_ij)


def make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs):
    all_pairs = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    self_pairs = all_pairs.T[0] == all_pairs.T[1]
    non_self_pairs = all_pairs[~self_pairs]
    non_self_pairs_jax = jnp.array(onp.array(non_self_pairs))
    if include_self_pairs:
        nr_pairs_choose = nr_pairs - nr_atoms
        assert nr_pairs_choose >= 0, "Number of pairs must be greater than or equal to the number of atoms if " \
                                     "include_self_pairs is True."
    else:
        nr_pairs_choose = nr_pairs
    pairlist_jax = jax.random.choice(key, non_self_pairs_jax, (nr_pairs_choose,), replace=False).T
    if include_self_pairs:
        pairlist_jax = jnp.concatenate([pairlist_jax, jnp.array(onp.array(all_pairs[self_pairs].T))], axis=1)
    pairlist = torch.tensor(onp.array(pairlist_jax), dtype=torch.int64)
    mask = jnp.zeros((nr_atoms, nr_atoms))
    for i in range(nr_pairs):
        mask = mask.at[pairlist_jax[0, i], pairlist_jax[1, i]].set(1)
    return pairlist, mask


def test_make_equivalent_pairlists():
    nr_atoms = 4
    nr_pairs = 5
    key = jax.random.PRNGKey(1884)
    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=False)
    assert pairlist.shape == (2, nr_pairs)
    assert mask.shape == (nr_atoms, nr_atoms)

    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=True)
    assert pairlist.shape == (2, nr_pairs)
    assert mask.shape == (nr_atoms, nr_atoms)


@pytest.mark.parametrize("include_self_pairs", [True, False])
def test_update_edge_against_reference(include_self_pairs):
    from modelforge.potential import CosineCutoff
    from modelforge.potential.sake import ExpNormalSmearing

    from sake.layers import ContinuousFilterConvolutionWithConcatenation
    from sake.functional import get_h_cat_ht, get_x_minus_xt, get_x_minus_xt_norm
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    nr_heads = 5
    nr_atom_basis = out_features
    nr_edge_basis = hidden_features
    geometry_basis = 3
    num_rbf = 31
    epsilon = 1e-5
    nr_pairs = 17

    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

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
        radial_basis_module=ExpNormalSmearing(0.0, 5.0, n_rbf=num_rbf),
        epsilon=epsilon
    )
    ref_sake_edge_model = ContinuousFilterConvolutionWithConcatenation(out_features=nr_edge_basis,
                                                                       kernel_features=num_rbf)

    # Generate random input data in JAX
    h_key, x_key, pairlist_key, init_key = jax.random.split(key, 4)
    h_jax = jax.random.normal(h_key, (nr_atoms, nr_atom_basis))
    x_jax = jax.random.normal(x_key, (nr_atoms, geometry_basis))

    h_cat_ht = get_h_cat_ht(h_jax)
    x_minus_xt = get_x_minus_xt(x_jax)
    x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)

    pairlist, mask = make_equivalent_pairlist_mask(pairlist_key, nr_atoms, nr_pairs,
                                                   include_self_pairs=include_self_pairs)
    idx_i, idx_j = pairlist

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = torch.from_numpy(onp.array(h_jax))
    x = torch.from_numpy(onp.array(x_jax))

    r_ij = x[idx_j] - x[idx_i]
    d_ij = torch.sqrt((r_ij ** 2).sum(dim=1) + epsilon)

    compare_equivalent_edge_features(x_minus_xt, r_ij, mask, pairlist)
    compare_equivalent_edge_features(x_minus_xt_norm, d_ij, mask, pairlist)

    variables = ref_sake_edge_model.init(init_key, h_cat_ht, x_minus_xt_norm)

    variables["params"]["mlp_in"]["kernel"] = mf_sake_block.edge_mlp_in.weight.detach().numpy().T
    variables["params"]["mlp_in"]["bias"] = mf_sake_block.edge_mlp_in.bias.detach().numpy().T
    variables["params"]["mlp_out"]["layers_0"]["kernel"] = mf_sake_block.edge_mlp_out[0].weight.detach().numpy().T
    variables["params"]["mlp_out"]["layers_0"]["bias"] = mf_sake_block.edge_mlp_out[0].bias.detach().numpy().T
    variables["params"]["mlp_out"]["layers_2"]["kernel"] = mf_sake_block.edge_mlp_out[1].weight.detach().numpy().T
    variables["params"]["mlp_out"]["layers_2"]["bias"] = mf_sake_block.edge_mlp_out[1].bias.detach().numpy().T
    variables['params']["kernel"]["means"] = mf_sake_block.radial_basis_module.means.detach().numpy().T
    variables['params']["kernel"]["betas"] = mf_sake_block.radial_basis_module.betas.detach().numpy().T

    ref_edge = ref_sake_edge_model.apply(variables, h_cat_ht, x_minus_xt_norm)
    mf_edge = mf_sake_block.update_edge(h[idx_j], h[idx_i], d_ij)

    compare_equivalent_edge_features(ref_edge, mf_edge, mask, pairlist)


def make_equivalent_edge_features(key, nr_features, nr_atoms, pairlist):
    nr_pairs = pairlist.shape[1]
    jax_array = jax.random.normal(key, (nr_atoms, nr_atoms, nr_features))
    torch_array = torch.zeros((nr_pairs, nr_features))
    for i in range(nr_pairs):
        torch_array[i] = torch.from_numpy(onp.array(jax_array[pairlist[0, i].item(), pairlist[1, i].item()]))
    return jax_array, torch_array


def compare_equivalent_edge_features(jax_array, torch_array, mask, pairlist, atol=1e-8):
    nr_pairs = pairlist.shape[1]
    for i in range(nr_pairs):
        if mask[pairlist[0, i].item(), pairlist[1, i].item()] == 0:
            # not checking masked values
            pass
        else:
            # print("torch_array", torch_array[i].shape, torch_array[i])
            # print("jax_array", jax_array[pairlist[0, i].item(), pairlist[1, i].item()].shape, jax_array[
            #     pairlist[0, i].item(), pairlist[1, i].item()])
            assert torch.allclose(torch_array[i], torch.from_numpy(
                onp.array(jax_array[pairlist[0, i].item(), pairlist[1, i].item()])), atol=atol)


def test_exp_normal_smearing_against_reference():
    from modelforge.potential.sake import ExpNormalSmearing
    from sake.utils import ExpNormalSmearing as RefExpNormalSmearing

    nr_atoms = 13
    nr_rbf = 11

    radial_basis_module = ExpNormalSmearing(0.0, 5.0, n_rbf=nr_rbf)
    ref_radial_basis_module = RefExpNormalSmearing(num_rbf=nr_rbf)
    key = jax.random.PRNGKey(1884)

    # Generate random input data in JAX
    d_ij_jax = jax.random.normal(key, (nr_atoms, nr_atoms, 1))
    d_ij = torch.from_numpy(onp.array(d_ij_jax)).reshape(nr_atoms ** 2)

    mf_rbf = radial_basis_module(d_ij)
    variables = ref_radial_basis_module.init(key, d_ij_jax)

    ref_rbf = ref_radial_basis_module.apply(variables, d_ij_jax)

    assert torch.allclose(mf_rbf, torch.from_numpy(onp.array(ref_rbf)).reshape(nr_atoms ** 2, nr_rbf))


@pytest.mark.parametrize("include_self_pairs", [True, False])
def test_combined_attention_against_reference(include_self_pairs):
    nr_atoms = 5
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 13
    nr_pairs = 17
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)
    nr_edge_basis = hidden_features

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    pairlist_key, input_key = jax.random.split(key, 2)
    pairlist, mask = make_equivalent_pairlist_mask(pairlist_key, nr_atoms, nr_pairs,
                                                   include_self_pairs=include_self_pairs)
    idx_i, idx_j = pairlist

    h_key, x_key = jax.random.split(input_key, 2)
    h_e_mtx, h_ij_edge = make_equivalent_edge_features(h_key, nr_edge_basis, nr_atoms, pairlist)
    x_minus_xt, x_minus_xt_torch = make_equivalent_edge_features(x_key, geometry_basis, nr_atoms, pairlist)

    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)
    d_ij = torch.norm(x_minus_xt_torch, dim=-1)

    mf_h_ij_semantic = mf_sake_block.get_semantic_attention(h_ij_edge, idx_i, idx_j, d_ij, nr_atoms)

    variables = ref_sake_interaction.init(key, x_minus_xt_norm, h_e_mtx, mask,
                                          method=ref_sake_interaction.combined_attention)
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    ref_combined_attention = \
        ref_sake_interaction.apply(variables, x_minus_xt_norm, h_e_mtx, mask,
                                   method=ref_sake_interaction.combined_attention)[
            2]
    ref_h_ij_semantic = jnp.reshape(
        jnp.expand_dims(h_e_mtx, axis=-1) * jnp.expand_dims(ref_combined_attention, axis=-2),
        (nr_atoms, nr_atoms, nr_heads * nr_edge_basis))

    compare_equivalent_edge_features(ref_h_ij_semantic, mf_h_ij_semantic, mask, pairlist)


@pytest.mark.parametrize("include_self_pairs", [True, False])
def test_spatial_attention_against_reference(include_self_pairs):
    nr_atoms = 7
    out_features = 2
    hidden_features = 2
    geometry_basis = 3
    nr_heads = 5
    nr_pairs = 13
    nr_coefficients = nr_heads * hidden_features
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    pairlist_key, input_key = jax.random.split(key, 2)
    pairlist, mask = make_equivalent_pairlist_mask(pairlist_key, nr_atoms, nr_pairs,
                                                   include_self_pairs=include_self_pairs)
    idx_i, idx_j = pairlist

    h_key, x_key = jax.random.split(input_key, 2)
    h_e_att, h_ij_semantic = make_equivalent_edge_features(h_key, nr_coefficients, nr_atoms, pairlist)
    x_minus_xt, r_ij = make_equivalent_edge_features(x_key, geometry_basis, nr_atoms, pairlist)

    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)
    d_ij = torch.sqrt((r_ij ** 2).sum(dim=1))
    dir_ij = r_ij / (d_ij.unsqueeze(-1) + 1e-5)

    mf_combinations = mf_sake_block.get_combinations(h_ij_semantic, dir_ij)
    mf_result = mf_sake_block.get_spatial_attention(mf_combinations, idx_i, nr_atoms)

    variables = ref_sake_interaction.init(key, h_e_att, x_minus_xt, x_minus_xt_norm, mask,
                                          method=ref_sake_interaction.spatial_attention)

    variables["params"]["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
        0].weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
        1].weight.detach().numpy().T
    ref_spatial, ref_combinations = ref_sake_interaction.apply(variables, h_e_att, x_minus_xt, x_minus_xt_norm, mask,
                                                               method=ref_sake_interaction.spatial_attention)
    compare_equivalent_edge_features(ref_combinations, mf_combinations, mask, pairlist)
    assert torch.allclose(mf_result, torch.from_numpy(onp.array(ref_spatial)))


@pytest.mark.parametrize("include_self_pairs", [True, False])
@pytest.mark.parametrize("v_is_none", [True, False])
def test_sake_layer_against_reference(include_self_pairs, v_is_none):
    print("include_self_pairs:", include_self_pairs)
    print("v_is_none:", v_is_none)
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_atom_basis = out_features
    nr_pairs = 17
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=include_self_pairs)

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
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

    print("v:", v)
    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = torch.from_numpy(onp.array(h_jax))
    x = torch.from_numpy(onp.array(x_jax))

    variables = ref_sake_interaction.init(init_key, h_jax, x_jax, v_jax, mask)

    variables["params"]["edge_model"]["mlp_in"]["kernel"] = mf_sake_block.edge_mlp_in.weight.detach().numpy().T
    variables["params"]["edge_model"]["mlp_in"]["bias"] = mf_sake_block.edge_mlp_in.bias.detach().numpy().T
    variables["params"]["edge_model"]["mlp_out"]["layers_0"]["kernel"] = mf_sake_block.edge_mlp_out[
        0].weight.detach().numpy().T
    variables["params"]["edge_model"]["mlp_out"]["layers_0"]["bias"] = mf_sake_block.edge_mlp_out[
        0].bias.detach().numpy().T
    variables["params"]["edge_model"]["mlp_out"]["layers_2"]["kernel"] = mf_sake_block.edge_mlp_out[
        1].weight.detach().numpy().T
    variables["params"]["edge_model"]["mlp_out"]["layers_2"]["bias"] = mf_sake_block.edge_mlp_out[
        1].bias.detach().numpy().T
    variables['params']["edge_model"]["kernel"]["means"] = mf_sake_block.radial_basis_module.means.detach().numpy().T
    variables['params']["edge_model"]["kernel"]["betas"] = mf_sake_block.radial_basis_module.betas.detach().numpy().T
    variables["params"]["node_mlp"]["layers_0"]["kernel"] = mf_sake_block.node_mlp[0].weight.detach().numpy().T
    variables["params"]["node_mlp"]["layers_2"]["kernel"] = mf_sake_block.node_mlp[1].weight.detach().numpy().T
    if not v_is_none:
        variables["params"]["velocity_mlp"]["layers_0"]["kernel"] = mf_sake_block.velocity_mlp[
            0].weight.detach().numpy().T
        variables["params"]["velocity_mlp"]["layers_0"]["bias"] = mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
        variables["params"]["velocity_mlp"]["layers_2"]["kernel"] = mf_sake_block.velocity_mlp[
            1].weight.detach().numpy().T
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    variables["params"]["v_mixing"]["kernel"] = mf_sake_block.v_mixing_mlp.weight.detach().numpy().T
    variables["params"]["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
        0].weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
        1].weight.detach().numpy().T

    mf_h, mf_x, mf_v = mf_sake_block(h, x, v, pairlist)

    ref_h, ref_x, ref_v = ref_sake_interaction.apply(variables, h_jax, x_jax, v_jax, mask)

    ref_h_is_nan = torch.from_numpy(onp.isnan(ref_h))
    ref_x_is_nan = torch.from_numpy(onp.isnan(ref_x))
    ref_v_is_nan = torch.from_numpy(onp.isnan(ref_v))

    assert torch.allclose(torch.nan_to_num(mf_h, nan=0.0) * ~ref_h_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array(ref_h)), nan=0.0))
    assert torch.allclose(torch.nan_to_num(mf_x, nan=0.0) * ~ref_x_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array(ref_x)), nan=0.0))
    assert torch.allclose(torch.nan_to_num(mf_v, nan=0.0) * ~ref_v_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array(ref_v)), nan=0.0))


def test_sake_model_against_reference():
    nr_heads = 5
    nr_atom_basis = 11
    max_Z = 13
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)
    nr_interaction_blocks = 3
    cutoff = 5.0 * unit.nanometer

    mf_sake = SAKE(
        max_Z=max_Z,
        number_of_atom_features=nr_atom_basis,
        number_of_interaction_modules=nr_interaction_blocks,
        number_of_spatial_attention_heads=nr_heads,
        cutoff=cutoff,
        number_of_radial_basis_functions=50,
        epsilon=1e-8
    )

    ref_sake = reference_sake.models.DenseSAKEModel(
        hidden_features=nr_atom_basis,
        out_features=1,
        depth=nr_interaction_blocks,
        n_heads=nr_heads,
        cutoff=None
    )

    from modelforge.dataset import QM9Dataset
    from modelforge.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input
    prepared_methane = mf_sake.prepare_inputs(methane)

    print(prepared_methane)
    mask = jnp.zeros((prepared_methane.number_of_atoms, prepared_methane.number_of_atoms))
    for i in range(prepared_methane.pair_indices.shape[1]):
        mask = mask.at[prepared_methane.pair_indices[0, i].item(), prepared_methane.pair_indices[1, i].item()].set(1)

    h = prepared_methane.atomic_embedding.detach().numpy()
    x = prepared_methane.positions.detach().numpy()
    variables = ref_sake.init(key, h, x, mask=mask)

    variables["params"]["embedding_in"]["kernel"] = mf_sake.embedding_in.weight.detach().numpy().T
    variables["params"]["embedding_in"]["bias"] = mf_sake.embedding_in.bias.detach().numpy().T
    variables["params"]["embedding_out"]["layers_0"]["kernel"] = mf_sake.energy_layer[0].weight.detach().numpy().T
    variables["params"]["embedding_out"]["layers_0"]["bias"] = mf_sake.energy_layer[0].bias.detach().numpy().T
    variables["params"]["embedding_out"]["layers_2"]["kernel"] = mf_sake.energy_layer[2].weight.detach().numpy().T
    variables["params"]["embedding_out"]["layers_2"]["bias"] = mf_sake.energy_layer[2].bias.detach().numpy().T
    layers = ((layer_name, variables["params"][layer_name]) for layer_name in variables["params"].keys() if
              layer_name.startswith("d"))
    for (layer_name, layer), mf_sake_block in zip(layers, mf_sake.interaction_modules.children()):
        layer["edge_model"]["mlp_in"]["kernel"] = mf_sake_block.edge_mlp_in.weight.detach().numpy().T
        layer["edge_model"]["mlp_in"]["bias"] = mf_sake_block.edge_mlp_in.bias.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_0"]["kernel"] = mf_sake_block.edge_mlp_out[
            0].weight.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_0"]["bias"] = mf_sake_block.edge_mlp_out[
            0].bias.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_2"]["kernel"] = mf_sake_block.edge_mlp_out[
            1].weight.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_2"]["bias"] = mf_sake_block.edge_mlp_out[
            1].bias.detach().numpy().T
        layer["edge_model"]["kernel"]["means"] = mf_sake_block.radial_basis_module.means.detach().numpy().T
        layer["edge_model"]["kernel"]["betas"] = mf_sake_block.radial_basis_module.betas.detach().numpy().T
        layer["node_mlp"]["layers_0"]["kernel"] = mf_sake_block.node_mlp[0].weight.detach().numpy().T
        layer["node_mlp"]["layers_2"]["kernel"] = mf_sake_block.node_mlp[1].weight.detach().numpy().T
        if layer_name != "d0":
            layer["velocity_mlp"]["layers_0"]["kernel"] = mf_sake_block.velocity_mlp[0].weight.detach().numpy().T
            layer["velocity_mlp"]["layers_0"]["bias"] = mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
            layer["velocity_mlp"]["layers_2"]["kernel"] = mf_sake_block.velocity_mlp[1].weight.detach().numpy().T
        layer["semantic_attention_mlp"]["layers_0"][
            "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
        layer["semantic_attention_mlp"]["layers_0"][
            "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
        layer["v_mixing"]["kernel"] = mf_sake_block.v_mixing_mlp.weight.detach().numpy().T
        layer["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
        layer["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
            0].weight.detach().numpy().T
        layer["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
            1].weight.detach().numpy().T

    ref_out = ref_sake.apply(variables, h, x, mask=mask)[0].sum(-2)
    mf_out = mf_sake(methane)

    # assert torch.allclose(mf_out.E, torch.from_numpy(onp.array(ref_out[0])))

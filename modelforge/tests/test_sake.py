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
from .helper_functions import (
    SIMPLIFIED_INPUT_DATA,
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "https://www.reddit.com/true"


def test_SAKE_init():
    """Test initialization of the SAKE neural network potential."""

    sake = SAKE()
    assert sake is not None, "SAKE model should be initialized."


from openff.units import unit


@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
def test_sake_forward(input_data):
    """
    Test the forward pass of the SAKE model.
    """
    sake = SAKE()
    nnp_input = input_data.nnp_input
    energy = sake(nnp_input).E
    nr_of_mols = nnp_input.atomic_subsystem_indices.unique().shape[0]

    assert (
            len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_sake_interaction_forward():
    from modelforge.potential import CosineCutoff
    from modelforge.potential import GaussianRBF
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
        radial_basis_module=GaussianRBF(n_rbf=31, cutoff=5.0 * unit.nanometer),
        cutoff_module=CosineCutoff(5.0 * unit.nanometer),
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


def test_sake_interaction_equivariance():
    import torch
    from .helper_functions import generate_methane_input
    from modelforge.potential.sake import SAKE
    from dataclasses import replace

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sake = SAKE()
    methane_input = generate_methane_input().nnp_input
    perturbed_methane_input = replace(methane_input)
    perturbed_methane_input.positions = torch.matmul(
        methane_input.positions, rotation_matrix
    )

    # prepare reference and perturbed inputs
    reference_prepared_input = sake.prepare_inputs(methane_input, only_unique_pairs=False)
    reference_v = torch.randn_like(reference_prepared_input.positions)

    perturbed_prepared_input = sake.prepare_inputs(perturbed_methane_input)
    perturbed_v = torch.matmul(reference_v, rotation_matrix)

    # Test that the interaction block is equivariant
    sake_interaction = sake.interaction_modules[0]

    reference_out = sake_interaction(
        reference_prepared_input.atomic_embedding,
        reference_prepared_input.positions,
        reference_v,
        reference_prepared_input.pair_indices
    )

    perturbed_out = sake_interaction(
        perturbed_prepared_input.atomic_embedding,
        perturbed_prepared_input.positions,
        perturbed_v,
        perturbed_prepared_input.pair_indices
    )

    perturbed_h, perturbed_x, perturbed_v = perturbed_out
    reference_h, reference_x, reference_v = reference_out

    # x and v are equivariant, h is invariant
    assert torch.allclose(reference_h, perturbed_h)
    assert torch.allclose(torch.matmul(reference_x, rotation_matrix), perturbed_x)
    assert torch.allclose(torch.matmul(reference_v, rotation_matrix), perturbed_v)


class CosineCutoffJAX(flax.linen.Module):
    """ JAX version of the cosine cutoff module. """
    cutoff_units: unit.Quantity

    def setup(self):
        """
        Behler-style cosine cutoff module.

        Parameters:
        ----------
        cutoff_units: unit.Quantity
            The cutoff distance.

        """
        super().__init__()
        self.cutoff = self.cutoff_units.to(unit.nanometer).m

    def __call__(self, input: Array) -> Array:
        return _cosine_cutoff(input, self.cutoff)


def _cosine_cutoff(d_ij: Array, cutoff: float) -> Array:
    """
    Compute the cosine cutoff for a distance tensor using JAX.
    NOTE: the cutoff function doesn't care about units as long as they are consisten,

    Parameters
    ----------
    d_ij : Tensor
        Pairwise distance tensor. Shape: [n_pairs, distance]
    cutoff : float
        The cutoff distance.

    Returns
    -------
    Tensor
        The cosine cutoff tensor. Shape: [..., N]
    """
    # Compute values of cutoff function
    input_cut = 0.5 * (jnp.cos(d_ij * jnp.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (d_ij < cutoff).astype(jnp.float_)
    return input_cut


def make_reference_equivalent_sake_interaction(out_features, hidden_features, nr_heads):
    from modelforge.potential import CosineCutoff
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
        radial_basis_module=ExpNormalSmearing(n_rbf=50),
        cutoff_module=CosineCutoff(5.0 * unit.nanometer),
        epsilon=1e-5
    )

    # Define the reference layer
    ref_sake_interaction = reference_sake.layers.DenseSAKELayer(out_features=out_features,
                                                                hidden_features=hidden_features,
                                                                n_heads=nr_heads,
                                                                cutoff=CosineCutoffJAX(5.0 * unit.nanometer),
                                                                )

    return mf_sake_block, ref_sake_interaction


def test_cutoff_against_reference():
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_pairs = nr_atoms ** 2
    key = jax.random.PRNGKey(1884)

    x_minus_xt = jax.random.normal(key, (nr_atoms, nr_atoms, geometry_basis))
    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)
    d_ij = torch.from_numpy(onp.array(x_minus_xt_norm)).reshape(nr_pairs, )

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    mf_cutoff = mf_sake_block.cutoff_module(d_ij)
    ref_cutoff = ref_sake_interaction.cutoff.apply({}, x_minus_xt_norm)
    assert torch.allclose(mf_cutoff, torch.from_numpy(onp.array(ref_cutoff).reshape(nr_pairs, )), atol=1e-4)


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
                          r_ij, atol=1e-4)

    x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)
    d_ij = torch.sqrt((r_ij ** 2).sum(dim=1) + 1e-5)

    # Fortran and C ordering give the same result since x_minus_xt_norm is symmetric
    assert jnp.array_equal(x_minus_xt_norm, jnp.transpose(x_minus_xt_norm, (1, 0, 2)))
    assert torch.allclose(torch.from_numpy(onp.array(x_minus_xt_norm.reshape(nr_atoms ** 2, order='F'))), d_ij)
    assert torch.allclose(torch.from_numpy(onp.array(x_minus_xt_norm.reshape(nr_atoms ** 2, order='C'))), d_ij)


def make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=True):
    all_pairs = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    self_pairs = all_pairs.T[0] == all_pairs.T[1]
    if not include_self_pairs:
        all_pairs = all_pairs[~self_pairs]
    all_pairs_jax = jnp.array(onp.array(all_pairs))
    pairlist_jax = jax.random.choice(key, all_pairs_jax, (nr_pairs,), replace=False).T
    pairlist = torch.tensor(onp.array(pairlist_jax), dtype=torch.int64)
    mask = jnp.zeros((nr_atoms, nr_atoms))
    for i in range(nr_pairs):
        mask = mask.at[pairlist_jax[0, i], pairlist_jax[1, i]].set(1)
    return pairlist, mask


def test_make_equivalent_pairlists():
    nr_atoms = 4
    nr_pairs = 3
    key = jax.random.PRNGKey(1884)
    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=False)
    assert pairlist.shape == (2, nr_pairs)
    assert mask.shape == (nr_atoms, nr_atoms)

    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=True)
    assert pairlist.shape == (2, nr_pairs)
    assert mask.shape == (nr_atoms, nr_atoms)


def test_update_edge_against_reference():
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

    key = jax.random.PRNGKey(1884)

    mf_sake_block = SAKEInteraction(
        nr_atom_basis=out_features,
        nr_edge_basis=hidden_features,
        nr_edge_basis_hidden=hidden_features,
        nr_atom_basis_hidden=hidden_features,
        nr_atom_basis_spatial_hidden=hidden_features,
        nr_atom_basis_spatial=hidden_features,
        nr_atom_basis_velocity=hidden_features,
        nr_coefficients=(
                nr_heads * hidden_features),
        nr_heads=nr_heads,
        activation=torch.nn.SiLU(),
        radial_basis_module=ExpNormalSmearing(
            n_rbf=num_rbf),
        cutoff_module=CosineCutoff(
            5.0 * unit.nanometer),
        epsilon=epsilon
    )
    ref_sake_edge_model = ContinuousFilterConvolutionWithConcatenation(out_features=nr_edge_basis,
                                                                       kernel_features=num_rbf)

    # Generate random input data in JAX
    h_jax = jax.random.normal(key, (nr_atoms, nr_atom_basis))
    x_jax = jax.random.normal(key, (nr_atoms, geometry_basis))

    h_cat_ht = get_h_cat_ht(h_jax)
    x_minus_xt = get_x_minus_xt(x_jax)
    x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt)

    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    pairlist = pairlist.T
    idx_i, idx_j = pairlist
    nr_pairs = nr_atoms ** 2

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = torch.from_numpy(onp.array(h_jax))
    x = torch.from_numpy(onp.array(x_jax))

    r_ij = x[idx_j] - x[idx_i]
    d_ij = torch.sqrt((r_ij ** 2).sum(dim=1) + epsilon)

    variables = ref_sake_edge_model.init(key, h_cat_ht, x_minus_xt_norm)

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

    # NOTE: if h[idx_j] is the first argument and h[idx_i] is the second argument, it matches C-style ordering.
    # if h[idx_i] is the first argument and h[idx_j] is the second argument, it matches Fortran-style ordering.
    assert torch.allclose(mf_edge, torch.from_numpy(onp.array(ref_edge).reshape(nr_pairs, -1, order='C')), atol=1e-7)

    mf_edge = mf_sake_block.update_edge(h[idx_i], h[idx_j], d_ij)
    assert torch.allclose(mf_edge, torch.from_numpy(onp.array(ref_edge).reshape(nr_pairs, -1, order='F')), atol=1e-7)


def make_equivalent_arrays(nr_features, nr_atoms, pairlist):
    nr_pairs = pairlist.shape[1]
    jax_array = jnp.arange(nr_atoms ** 2 * nr_features, dtype=float).reshape(nr_atoms, nr_atoms, nr_features)
    torch_array = torch.zeros((nr_pairs, nr_features))
    for i in range(nr_pairs):
        torch_array[i] = torch.from_numpy(onp.array(jax_array[pairlist[0, i].item(), pairlist[1, i].item()]))
    return jax_array, torch_array


def compare_equivalent_arrays(jax_array, torch_array, mask, pairlist):
    nr_pairs = pairlist.shape[1]
    for i in range(nr_pairs):
        if mask[pairlist[0, i].item(), pairlist[1, i].item()] == 0:
            # not checking masked values
            pass
        else:
            assert torch.allclose(torch_array[i], torch.from_numpy(
                onp.array(jax_array[pairlist[0, i].item(), pairlist[1, i].item()])))


def test_semantic_attention_against_reference():
    from modelforge.potential.utils import scatter_softmax

    nr_atoms = 7
    out_features = 11
    hidden_features = 6
    nr_heads = 4
    nr_pairs = 5
    key = jax.random.PRNGKey(1884)
    nr_edge_basis = hidden_features

    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=False)
    idx_i, idx_j = pairlist
    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)

    h_e_mtx, h_ij_edge = make_equivalent_arrays(nr_edge_basis, nr_atoms, pairlist)

    h_ij_att_weights = mf_sake_block.semantic_attention_mlp(h_ij_edge)
    expanded_idx_i = idx_i.view(-1, 1).expand_as(h_ij_att_weights)
    h_ij_att_before_cutoff = scatter_softmax(h_ij_att_weights, expanded_idx_i, dim=0, dim_size=nr_atoms,
                                             device=h_ij_edge.device)

    variables = ref_sake_interaction.init(key, h_e_mtx, mask,
                                          method=ref_sake_interaction.semantic_attention)
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    ref_semantic_attention = \
        ref_sake_interaction.apply(variables, h_e_mtx, mask, method=ref_sake_interaction.semantic_attention)

    # NOTE: if there is no sender(?) for a particular receiver(?), the attention weights will not be zero.
    compare_equivalent_arrays(ref_semantic_attention, h_ij_att_before_cutoff, mask, pairlist)


def test_exp_normal_smearing_against_reference():
    from modelforge.potential.sake import ExpNormalSmearing
    from sake.utils import ExpNormalSmearing as RefExpNormalSmearing

    nr_atoms = 13
    nr_rbf = 11

    radial_basis_module = ExpNormalSmearing(n_rbf=nr_rbf)
    ref_radial_basis_module = RefExpNormalSmearing(num_rbf=nr_rbf)
    key = jax.random.PRNGKey(1884)

    # Generate random input data in JAX
    d_ij_jax = jax.random.normal(key, (nr_atoms, nr_atoms, 1))
    d_ij = torch.from_numpy(onp.array(d_ij_jax)).reshape(nr_atoms ** 2)

    mf_rbf = radial_basis_module(d_ij)
    variables = ref_radial_basis_module.init(key, d_ij_jax)

    ref_rbf = ref_radial_basis_module.apply(variables, d_ij_jax)

    assert torch.allclose(mf_rbf, torch.from_numpy(onp.array(ref_rbf)).reshape(nr_atoms ** 2, nr_rbf))


def test_combined_attention_against_reference():
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_pairs = 17
    key = jax.random.PRNGKey(1884)
    nr_edge_basis = hidden_features

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=False)
    idx_i, idx_j = pairlist

    h_e_mtx, h_ij_edge = make_equivalent_arrays(nr_edge_basis, nr_atoms, pairlist)
    x_minus_xt, x_minus_xt_norm = make_equivalent_arrays(geometry_basis, nr_atoms, pairlist)

    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)
    d_ij = torch.norm(h_ij_edge, dim=-1)

    mf_h_ij_semantic = mf_sake_block.get_semantic_attention(h_ij_edge, idx_i, d_ij, nr_atoms)

    variables = ref_sake_interaction.init(key, x_minus_xt_norm, h_e_mtx, mask,
                                          method=ref_sake_interaction.combined_attention)
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    variables["params"]["semantic_attention_mlp"]["layers_0"][
        "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    ref_combined_attention = \
        ref_sake_interaction.apply(variables, x_minus_xt_norm, h_e_mtx, mask, method=ref_sake_interaction.combined_attention)[
            2]
    ref_h_ij_semantic = jnp.reshape(
        jnp.expand_dims(h_e_mtx, axis=-1) * jnp.expand_dims(ref_combined_attention, axis=-2),
        (nr_atoms, nr_atoms, nr_heads * nr_edge_basis))

    compare_equivalent_arrays(ref_h_ij_semantic, mf_h_ij_semantic, mask, pairlist)


def test_spatial_attention_against_reference():
    # Define the parameters for the test
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)

    nr_coefficients = nr_heads * hidden_features

    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    nr_pairs = nr_atoms ** 2
    pairlist = pairlist.T
    idx_i, idx_j = pairlist

    key = jax.random.PRNGKey(1884)

    # Generate random input data in JAX
    h_e_att = jax.random.normal(key, (nr_atoms, nr_atoms, nr_coefficients))
    x_minus_xt = jax.random.normal(key, (nr_atoms, nr_atoms, geometry_basis))
    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h_ij_semantic = torch.from_numpy(onp.array(h_e_att)).reshape(nr_pairs, hidden_features * nr_heads)
    dir_ij = torch.from_numpy(onp.array(x_minus_xt / (x_minus_xt_norm + 1e-5))).reshape(nr_pairs, geometry_basis)

    mf_combinations = mf_sake_block.get_combinations(h_ij_semantic, dir_ij)
    mf_result = mf_sake_block.get_spatial_attention(mf_combinations, idx_i, nr_atoms)

    variables = ref_sake_interaction.init(key, h_e_att, x_minus_xt, x_minus_xt_norm,
                                          method=ref_sake_interaction.spatial_attention)

    variables["params"]["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
        0].weight.detach().numpy().T
    variables["params"]["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
        1].weight.detach().numpy().T
    ref_spatial, ref_combinations = ref_sake_interaction.apply(variables, h_e_att, x_minus_xt, x_minus_xt_norm,
                                                               method=ref_sake_interaction.spatial_attention)
    assert (torch.allclose(mf_combinations,
                           torch.from_numpy(onp.array(ref_combinations)).reshape(nr_pairs, nr_heads * hidden_features,
                                                                                 geometry_basis), atol=1e-6))
    assert (torch.allclose(mf_result, torch.from_numpy(onp.array(ref_spatial)), atol=1e-6))


def test_sake_layer_against_reference():
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_atom_basis = out_features
    key = jax.random.PRNGKey(1884)

    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    pairlist = pairlist.T

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    # Generate random input data in JAX
    h_jax = jax.random.normal(key, (nr_atoms, nr_atom_basis))
    x_jax = jax.random.normal(key, (nr_atoms, geometry_basis))
    v_jax = jax.random.normal(key, (nr_atoms, geometry_basis))

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = torch.from_numpy(onp.array(h_jax))
    x = torch.from_numpy(onp.array(x_jax))
    v = torch.from_numpy(onp.array(v_jax))

    variables = ref_sake_interaction.init(key, h_jax, x_jax, v_jax)
    print(variables["params"].keys())

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
    variables["params"]["velocity_mlp"]["layers_0"]["kernel"] = mf_sake_block.velocity_mlp[0].weight.detach().numpy().T
    variables["params"]["velocity_mlp"]["layers_0"]["bias"] = mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
    variables["params"]["velocity_mlp"]["layers_2"]["kernel"] = mf_sake_block.velocity_mlp[1].weight.detach().numpy().T
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

    ref_h, ref_x, ref_v = ref_sake_interaction.apply(variables, h_jax, x_jax, v_jax)

    print(mf_h, torch.from_numpy(onp.array(ref_h)))
    assert torch.allclose(mf_h, torch.from_numpy(onp.array(ref_h)), atol=1e-4)
    assert torch.allclose(mf_x, torch.from_numpy(onp.array(ref_x)), atol=1e-4)
    assert torch.allclose(mf_v, torch.from_numpy(onp.array(ref_v)), atol=1e-4)
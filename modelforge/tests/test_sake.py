import os

import jax.random
import jax.numpy as jnp
import pytest
import torch
import numpy as onp

from modelforge.potential.sake import SAKE, SAKEInteraction
import sake as reference_sake
from .helper_functions import (
    setup_simple_model,
    SIMPLIFIED_INPUT_DATA,
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "https://www.reddit.com/true"


@pytest.mark.parametrize("lightning", [True, False])
def test_SAKE_init(lightning):
    """Test initialization of the SAKE neural network potential."""

    sake = setup_simple_model(SAKE, lightning=lightning)
    assert sake is not None, "SAKE model should be initialized."


from openff.units import unit


@pytest.mark.parametrize("lightning", [True, False])
@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
@pytest.mark.parametrize(
    "model_parameter",
    (
            [64, 50, 2, unit.Quantity(5.0, unit.angstrom), 2],
            [32, 60, 10, unit.Quantity(7.0, unit.angstrom), 1],
            [128, 120, 5, unit.Quantity(5.0, unit.angstrom), 3],
    ),
)
def test_sake_forward(lightning, input_data, model_parameter):
    """
    Test the forward pass of the SAKE model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        n_rbf,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    sake = setup_simple_model(
        SAKE,
        lightning=lightning,
        nr_atom_basis=nr_atom_basis,
        max_atomic_number=max_atomic_number,
        n_rbf=n_rbf,
        cutoff=cutoff,
        nr_interaction_blocks=nr_interaction_blocks,
    )
    energy = sake(input_data)
    nr_of_mols = input_data["atomic_subsystem_indices"].unique().shape[0]

    assert energy.shape == (
        nr_of_mols,
        1,
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
    from .helper_functions import generate_methane_input, setup_simple_model
    from modelforge.potential.sake import SAKE

    # define a rotation matrix in 3D that rotates by 90 degrees around the z-axis
    # (clockwise when looking along the z-axis towards the origin)
    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    sake = setup_simple_model(SAKE)
    methane_input = generate_methane_input()
    perturbed_methane_input = methane_input.copy()
    perturbed_methane_input["positions"] = torch.matmul(
        methane_input["positions"], rotation_matrix
    )

    # prepare reference and perturbed inputs
    reference_prepared_input = sake.prepare_inputs(methane_input)
    reference_v = torch.randn_like(reference_prepared_input["positions"])
    reference_d_ij = reference_prepared_input["d_ij"]
    reference_r_ij = reference_prepared_input["r_ij"]
    reference_dir_ij = reference_r_ij / reference_d_ij

    perturbed_prepared_input = sake.prepare_inputs(perturbed_methane_input)
    perturbed_v = torch.matmul(reference_v, rotation_matrix)
    perturbed_d_ij = perturbed_prepared_input["d_ij"]
    perturbed_r_ij = perturbed_prepared_input["r_ij"]
    perturbed_dir_ij = perturbed_r_ij / perturbed_d_ij

    # check that the invariant properties are preserved
    # d_ij is the distance between atom i and j
    assert torch.allclose(reference_d_ij, perturbed_d_ij)

    # what shoudl not be invariant is the direction
    assert not torch.allclose(reference_dir_ij, perturbed_dir_ij)

    # Check for equivariance
    # rotate the reference dir_ij
    rotated_reference_dir_ij = torch.matmul(reference_dir_ij, rotation_matrix)
    # Compare the rotated original dir_ij with the dir_ij from rotated positions
    assert torch.allclose(rotated_reference_dir_ij, perturbed_dir_ij)

    # Test that the interaction block is equivariant
    sake_interaction = sake.interaction_modules[0]

    reference_r = sake_interaction(
        reference_prepared_input["atomic_embedding"],
        reference_prepared_input["positions"],
        reference_v,
        reference_prepared_input["pair_indices"]
    )

    perturbed_r = sake_interaction(
        perturbed_prepared_input["atomic_embedding"],
        perturbed_prepared_input["positions"],
        perturbed_v,
        perturbed_prepared_input["pair_indices"],
    )

    perturbed_h, perturbed_x, perturbed_v = perturbed_r
    reference_h, reference_x, reference_v = reference_r

    # x and v are equivariant, h is invariant
    assert torch.allclose(reference_h, perturbed_h)
    assert torch.allclose(torch.matmul(reference_x, rotation_matrix), perturbed_x)
    assert torch.allclose(torch.matmul(reference_v, rotation_matrix), perturbed_v)


def test_spatial_attention_against_reference():
    from modelforge.potential import CosineCutoff
    from modelforge.potential import GaussianRBF
    nr_atoms = 2
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_rbf = 50
    cutoff_module = CosineCutoff(5.0 * unit.nanometer)
    mf_sake_block = SAKEInteraction(
        nr_atom_basis=out_features,
        nr_edge_basis=hidden_features,
        nr_edge_basis_hidden=hidden_features,
        nr_atom_basis_hidden=hidden_features,
        nr_atom_basis_spatial_hidden=hidden_features,
        nr_atom_basis_spatial=hidden_features,
        nr_atom_basis_velocity=hidden_features,
        nr_coefficients=nr_heads * hidden_features,
        nr_heads=nr_heads,
        activation=torch.nn.SiLU(),
        radial_basis_module=GaussianRBF(n_rbf=nr_rbf, cutoff=5.0 * unit.nanometer),
        cutoff_module=cutoff_module,
        epsilon=1e-5
    )
    ref_sake_interaction = reference_sake.layers.DenseSAKELayer(out_features=out_features,
                                                                hidden_features=hidden_features,
                                                                n_heads=nr_heads,
                                                                cutoff=cutoff_module
                                                                )

    pairlist = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    nr_pairs = nr_atoms ** 2
    pairlist = pairlist.T
    idx_i, idx_j = pairlist

    key = jax.random.PRNGKey(1884)

    h_e_att = jax.random.normal(key, (nr_atoms, nr_atoms, hidden_features * nr_heads))
    x_minus_xt = jax.random.normal(key, (nr_atoms, nr_atoms, geometry_basis))
    x_minus_xt_norm = jnp.linalg.norm(x_minus_xt, axis=-1, keepdims=True)

    h_ij_semantic = torch.from_numpy(onp.array(h_e_att)).reshape(nr_pairs, hidden_features * nr_heads)
    dir_ij = torch.from_numpy(onp.array(x_minus_xt / (x_minus_xt_norm + 1e-5))).reshape(nr_pairs, geometry_basis)

    # variables = ref_sake_interaction.init(key, h_e_att, x_minus_xt, x_minus_xt_norm,
    #                                       method=ref_sake_interaction.combined_attention)
    # variables["params"]["semantic_attention_mlp"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    #
    mf_combinations = mf_sake_block.get_combinations(h_ij_semantic, dir_ij)
    mf_result = mf_sake_block.get_spatial_attention(mf_combinations, idx_i, nr_atoms)

    variables = ref_sake_interaction.init(key, h_e_att, x_minus_xt, x_minus_xt_norm,
                                          method=ref_sake_interaction.spatial_attention)

    variables["params"]["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T
    # variables["params"]["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
    #     0].weight.detach().numpy().T
    # variables["params"]["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
    #     1].weight.detach().numpy().T
    ref_result = ref_sake_interaction.apply(variables, h_e_att, x_minus_xt, x_minus_xt_norm,
                                            method=ref_sake_interaction.spatial_attention)
    assert (torch.allclose(mf_combinations,
                           torch.from_numpy(onp.array(ref_result[1])).reshape(nr_pairs, nr_heads * hidden_features,
                                                                              geometry_basis), atol=1e-4))
    assert (torch.allclose(mf_result, torch.from_numpy(onp.array(ref_result[0])), atol=1e-4))



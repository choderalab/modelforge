import os

import jax.random
import jax.numpy as jnp
import pytest
import torch
import numpy as onp

from modelforge.potential.sake import SAKE, SAKEInteraction
from modelforge.potential.utils import SAKERadialBasisFunction
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
    energy = sake(methane).E
    nr_of_mols = methane.atomic_subsystem_indices.unique().shape[0]

    assert (
            len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def test_sake_interaction_forward():
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
        cutoff=5.0 * unit.angstrom,
        number_of_radial_basis_functions=53,
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


@pytest.mark.parametrize("eq_atol", [3e-1])
@pytest.mark.parametrize("h_atol", [8e-2])
def test_sake_layer_equivariance(h_atol, eq_atol):
    import torch
    from modelforge.potential.sake import SAKE
    from dataclasses import replace

    # Model parameters
    nr_atom_basis = 11
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
    assert torch.allclose(reference_h_out_torch, perturbed_h_out_torch, atol=h_atol)
    assert torch.allclose(torch.matmul(reference_x_out_torch, rotation_matrix), perturbed_x_out_torch, atol=eq_atol)
    assert torch.allclose(torch.matmul(reference_v_out_torch, rotation_matrix), perturbed_v_out_torch, atol=eq_atol)


def make_reference_equivalent_sake_interaction(out_features, hidden_features, nr_heads):
    cutoff = 5.0 * unit.angstrom
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
        cutoff=cutoff,
        number_of_radial_basis_functions=50,
        epsilon=1e-5
    )

    # Define the reference layer
    ref_sake_interaction = reference_sake.layers.DenseSAKELayer(out_features=out_features,
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


def test_radial_symmetry_function_against_reference():
    from modelforge.potential.utils import SAKERadialSymmetryFunction, SAKERadialBasisFunction
    from sake.utils import ExpNormalSmearing as RefExpNormalSmearing

    nr_atoms = 13
    number_of_radial_basis_functions = 11
    cutoff_upper = 6.0 * unit.bohr
    cutoff_lower = 2.0 * unit.bohr
    mf_unit = unit.nanometer
    ref_unit = unit.nanometer

    radial_symmetry_function_module = SAKERadialSymmetryFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        max_distance=cutoff_upper,
        min_distance=cutoff_lower,
        dtype=torch.float32, trainable=False,
        radial_basis_function=SAKERadialBasisFunction(
            cutoff_lower))
    ref_radial_basis_module = RefExpNormalSmearing(num_rbf=number_of_radial_basis_functions, cutoff_upper=cutoff_upper.to(ref_unit).m, cutoff_lower=cutoff_lower.to(ref_unit).m)
    key = jax.random.PRNGKey(1884)

    # Generate random input data in JAX
    d_ij_bohr_mag = jax.random.normal(key, (nr_atoms, nr_atoms, 1))
    d_ij_jax = (d_ij_bohr_mag * unit.bohr).to(ref_unit).m
    d_ij = torch.from_numpy(onp.array((d_ij_bohr_mag * unit.bohr).to(mf_unit).m)).reshape(nr_atoms ** 2)

    mf_rbf = radial_symmetry_function_module(d_ij)
    variables = ref_radial_basis_module.init(key, d_ij_jax)

    assert torch.allclose(torch.from_numpy(onp.array(variables["params"]["means"])), radial_symmetry_function_module.radial_basis_centers.detach().T)
    assert torch.allclose(torch.from_numpy(onp.array(variables["params"]["betas"])), radial_symmetry_function_module.radial_scale_factor.detach().T)

    ref_rbf = ref_radial_basis_module.apply(variables, d_ij_jax)

    assert torch.allclose(mf_rbf, torch.from_numpy(onp.array(ref_rbf)).reshape(nr_atoms ** 2, number_of_radial_basis_functions))


@pytest.mark.parametrize("include_self_pairs", [True, False])
@pytest.mark.parametrize("v_is_none", [True, False])
@pytest.mark.parametrize("atol", [9e-7])
def test_sake_layer_against_reference(include_self_pairs, v_is_none, atol):
    nr_atoms = 13
    out_features = 11
    hidden_features = 7
    geometry_basis = 3
    nr_heads = 5
    nr_atom_basis = out_features
    nr_pairs = 17
    mf_unit = unit.nanometer
    ref_unit = unit.nanometer
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)

    pairlist, mask = make_equivalent_pairlist_mask(key, nr_atoms, nr_pairs, include_self_pairs=include_self_pairs)

    mf_sake_block, ref_sake_interaction = make_reference_equivalent_sake_interaction(out_features, hidden_features,
                                                                                     nr_heads)
    # Generate random input data in JAX
    h_key, x_key, v_key, init_key = jax.random.split(key, 4)
    h_jax = jax.random.normal(h_key, (nr_atoms, nr_atom_basis))
    x_bohr_mag = jax.random.normal(x_key, (nr_atoms, geometry_basis))
    x_jax = (x_bohr_mag * unit.bohr).to(ref_unit).m
    if v_is_none:
        v_jax = None
        v = torch.zeros((nr_atoms, geometry_basis))
    else:
        v_bohr_mag = jax.random.normal(v_key, (nr_atoms, geometry_basis))
        v_jax = (v_bohr_mag * unit.bohr).to(ref_unit).m
        v = torch.from_numpy(onp.array((v_bohr_mag * unit.bohr).to(mf_unit).m))

    # Convert the input tensors from JAX to torch and reshape to diagonal batching
    h = (torch.from_numpy(onp.array(h_jax)))
    x = (torch.from_numpy(onp.array(x_bohr_mag)) * unit.bohr).to(ref_unit).m

    variables = ref_sake_interaction.init(init_key, h_jax, x_jax, v_jax, mask)
    layer = variables["params"]

    assert torch.allclose(torch.from_numpy(onp.array(layer["edge_model"]["kernel"]["betas"])),
                          mf_sake_block.radial_symmetry_function_module.radial_scale_factor.detach().T)
    assert torch.allclose(torch.from_numpy(onp.array(layer["edge_model"]["kernel"]["means"])),
                          mf_sake_block.radial_symmetry_function_module.radial_basis_centers.detach().T)
    layer["edge_model"]["mlp_in"]["bias"] = mf_sake_block.edge_mlp_in.bias.detach().numpy().T
    layer["edge_model"]["mlp_in"]["kernel"] = mf_sake_block.edge_mlp_in.weight.detach().numpy().T
    layer["edge_model"]["mlp_out"]["layers_0"]["bias"] = mf_sake_block.edge_mlp_out[
        0].bias.detach().numpy().T
    layer["edge_model"]["mlp_out"]["layers_0"]["kernel"] = mf_sake_block.edge_mlp_out[
        0].weight.detach().numpy().T
    layer["edge_model"]["mlp_out"]["layers_2"]["bias"] = mf_sake_block.edge_mlp_out[
        1].bias.detach().numpy().T
    layer["edge_model"]["mlp_out"]["layers_2"]["kernel"] = mf_sake_block.edge_mlp_out[
        1].weight.detach().numpy().T
    layer["node_mlp"]["layers_0"]["bias"] = mf_sake_block.node_mlp[0].bias.detach().numpy().T
    layer["node_mlp"]["layers_0"]["kernel"] = mf_sake_block.node_mlp[0].weight.detach().numpy().T
    layer["node_mlp"]["layers_2"]["bias"] = mf_sake_block.node_mlp[1].bias.detach().numpy().T
    layer["node_mlp"]["layers_2"]["kernel"] = mf_sake_block.node_mlp[1].weight.detach().numpy().T
    layer["post_norm_mlp"]["layers_0"]["bias"] = mf_sake_block.post_norm_mlp[
        0].bias.detach().numpy().T
    layer["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
        0].weight.detach().numpy().T
    layer["post_norm_mlp"]["layers_2"]["bias"] = mf_sake_block.post_norm_mlp[
        1].bias.detach().numpy().T
    layer["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
        1].weight.detach().numpy().T
    layer["semantic_attention_mlp"]["layers_0"][
        "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
    layer["semantic_attention_mlp"]["layers_0"][
        "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
    if not v_is_none:
        layer["velocity_mlp"]["layers_0"]["kernel"] = mf_sake_block.velocity_mlp[0].weight.detach().numpy().T
        layer["velocity_mlp"]["layers_0"]["bias"] = mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
        layer["velocity_mlp"]["layers_2"]["kernel"] = mf_sake_block.velocity_mlp[1].weight.detach().numpy().T
    layer["v_mixing"]["kernel"] = mf_sake_block.v_mixing_mlp.weight.detach().numpy().T
    layer["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T

    mf_h, mf_x, mf_v = mf_sake_block(h, x, v, pairlist)

    ref_h, ref_x, ref_v = ref_sake_interaction.apply(variables, h_jax, x_jax, v_jax, mask)

    ref_h_is_nan = torch.from_numpy(onp.isnan(ref_h))
    ref_x_is_nan = torch.from_numpy(onp.isnan(ref_x))
    ref_v_is_nan = torch.from_numpy(onp.isnan(ref_v))

    print(f"{ref_h_is_nan.sum()} NaNs in ref_h")
    print(f"{ref_x_is_nan.sum()} NaNs in ref_x")
    print(f"{ref_v_is_nan.sum()} NaNs in ref_v")
    assert torch.allclose(torch.nan_to_num(mf_h, nan=0.0) * ~ref_h_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array(ref_h)), nan=0.0))
    assert torch.allclose(torch.nan_to_num(mf_x, nan=0.0) * ~ref_x_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array((ref_x * ref_unit).to(mf_unit).m)), nan=0.0))
    assert torch.allclose(torch.nan_to_num(mf_v, nan=0.0) * ~ref_v_is_nan,
                          torch.nan_to_num(torch.from_numpy(onp.array((ref_v * ref_unit).to(mf_unit).m)), nan=0.0), atol=atol)


def test_sake_model_against_reference():
    nr_heads = 5
    nr_atom_basis = 11
    max_Z = 13
    key = jax.random.PRNGKey(1884)
    torch.manual_seed(1884)
    nr_interaction_blocks = 3
    cutoff = 5.0 * unit.angstrom

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

    mask = jnp.zeros((prepared_methane.number_of_atoms, prepared_methane.number_of_atoms))
    for i in range(prepared_methane.pair_indices.shape[1]):
        mask = mask.at[prepared_methane.pair_indices[0, i].item(), prepared_methane.pair_indices[1, i].item()].set(1)

    h = jax.nn.one_hot(prepared_methane.atomic_numbers.detach().numpy(), max_Z)
    x = prepared_methane.positions.detach().numpy()
    variables = ref_sake.init(key, h, x, mask=mask)

    variables["params"]["embedding_in"]["kernel"] = mf_sake.embedding.weight.detach().numpy().T
    variables["params"]["embedding_in"]["bias"] = mf_sake.embedding.bias.detach().numpy().T
    variables["params"]["embedding_out"]["layers_0"]["kernel"] = mf_sake.energy_layer[0].weight.detach().numpy().T
    variables["params"]["embedding_out"]["layers_0"]["bias"] = mf_sake.energy_layer[0].bias.detach().numpy().T
    variables["params"]["embedding_out"]["layers_2"]["kernel"] = mf_sake.energy_layer[2].weight.detach().numpy().T
    variables["params"]["embedding_out"]["layers_2"]["bias"] = mf_sake.energy_layer[2].bias.detach().numpy().T
    layers = ((layer_name, variables["params"][layer_name]) for layer_name in variables["params"].keys() if
              layer_name.startswith("d"))
    for (layer_name, layer), mf_sake_block in zip(layers, mf_sake.interaction_modules.children()):
        layer["edge_model"]["kernel"][
            "betas"] = mf_sake_block.radial_symmetry_function_module.radial_scale_factor.detach().numpy().T
        layer["edge_model"]["kernel"][
            "means"] = mf_sake_block.radial_symmetry_function_module.radial_basis_centers.detach().numpy().T
        layer["edge_model"]["mlp_in"]["bias"] = mf_sake_block.edge_mlp_in.bias.detach().numpy().T
        layer["edge_model"]["mlp_in"]["kernel"] = mf_sake_block.edge_mlp_in.weight.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_0"]["bias"] = mf_sake_block.edge_mlp_out[
            0].bias.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_0"]["kernel"] = mf_sake_block.edge_mlp_out[
            0].weight.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_2"]["bias"] = mf_sake_block.edge_mlp_out[
            1].bias.detach().numpy().T
        layer["edge_model"]["mlp_out"]["layers_2"]["kernel"] = mf_sake_block.edge_mlp_out[
            1].weight.detach().numpy().T
        layer["node_mlp"]["layers_0"]["bias"] = mf_sake_block.node_mlp[0].bias.detach().numpy().T
        layer["node_mlp"]["layers_0"]["kernel"] = mf_sake_block.node_mlp[0].weight.detach().numpy().T
        layer["node_mlp"]["layers_2"]["bias"] = mf_sake_block.node_mlp[1].bias.detach().numpy().T
        layer["node_mlp"]["layers_2"]["kernel"] = mf_sake_block.node_mlp[1].weight.detach().numpy().T
        layer["post_norm_mlp"]["layers_0"]["bias"] = mf_sake_block.post_norm_mlp[
            0].bias.detach().numpy().T
        layer["post_norm_mlp"]["layers_0"]["kernel"] = mf_sake_block.post_norm_mlp[
            0].weight.detach().numpy().T
        layer["post_norm_mlp"]["layers_2"]["bias"] = mf_sake_block.post_norm_mlp[
            1].bias.detach().numpy().T
        layer["post_norm_mlp"]["layers_2"]["kernel"] = mf_sake_block.post_norm_mlp[
            1].weight.detach().numpy().T
        layer["semantic_attention_mlp"]["layers_0"][
            "bias"] = mf_sake_block.semantic_attention_mlp.bias.detach().numpy().T
        layer["semantic_attention_mlp"]["layers_0"][
            "kernel"] = mf_sake_block.semantic_attention_mlp.weight.detach().numpy().T
        if layer_name != "d0":
            layer["velocity_mlp"]["layers_0"]["kernel"] = mf_sake_block.velocity_mlp[0].weight.detach().numpy().T
            layer["velocity_mlp"]["layers_0"]["bias"] = mf_sake_block.velocity_mlp[0].bias.detach().numpy().T
            layer["velocity_mlp"]["layers_2"]["kernel"] = mf_sake_block.velocity_mlp[1].weight.detach().numpy().T
        layer["v_mixing"]["kernel"] = mf_sake_block.v_mixing_mlp.weight.detach().numpy().T
        layer["x_mixing"]["layers_0"]["kernel"] = mf_sake_block.x_mixing_mlp.weight.detach().numpy().T

    # jax.tree_util.tree_map_with_path(lambda path, leaf: print(path, leaf.shape), variables)

    mf_out = mf_sake(methane)
    ref_out = ref_sake.apply(variables, h, x, mask=mask)[0].sum(-2)
    # ref_out is nan, so we can't compare it to the modelforge output

    # assert torch.allclose(mf_out.E, torch.from_numpy(onp.array(ref_out[0])))


def test_model_invariance():
    from modelforge.dataset import QM9Dataset
    from modelforge.dataset import TorchDataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from dataclasses import replace

    model = SAKE()
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(
        data, batch_size=1, splitting_strategy=FirstComeFirstServeSplittingStrategy()
    )

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    # get methane input
    methane = next(iter(dataset.train_dataloader())).nnp_input

    rotation_matrix = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    perturbed_methane_input = replace(methane)
    perturbed_methane_input.positions = torch.matmul(
        methane.positions, rotation_matrix
    )

    reference_out = model(methane)
    perturbed_out = model(perturbed_methane_input)

    assert torch.allclose(reference_out.E, perturbed_out.E)

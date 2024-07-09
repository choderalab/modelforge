def test_tensornet_init():
    import torch

    from modelforge.potential.tensornet import TensorNet

    seed = 0
    torch.manual_seed(seed)

    net = TensorNet()
    assert net is not None


def test_tensornet_forward():  # TODO
    import torch

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

    seed = 0
    torch.manual_seed(seed)

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    batch = next(iter(dataset.train_dataloader())).nnp_input
    from modelforge.potential.tensornet import TensorNet

    net = TensorNet()
    # net(batch)


def test_tensornet_input():
    import torch
    from openff.units import unit

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.tests.precalculated_values import prepare_values_for_test_tensornet_input

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_input.pt"
    # reference_data = None

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    mf_input = next(iter(dataset.train_dataloader())).nnp_input
    # modelforge TensorNet
    model = TensorNet(radial_max_distance=100 * unit.angstrom)  # no max distance for test purposes
    model.input_preparation._input_checks(mf_input)
    pairlist_output = model.input_preparation.prepare_inputs(mf_input)

    # torchmd-net TensorNet
    if reference_data:
        edge_index, edge_weight, edge_vec = torch.load(reference_data)
    else:
        edge_index, edge_weight, edge_vec = prepare_values_for_test_tensornet_input(
            mf_input,
            seed=seed,
        )

    # reshape and compare
    pair_indices = pairlist_output.pair_indices.t()
    edge_index = edge_index.t()
    for _, pair_index in enumerate(pair_indices):
        idx = ((edge_index == pair_index).sum(axis=1) == 2).nonzero()[0][0]  # select [True, True]
        print(pairlist_output.d_ij[_][0])
        print(edge_weight[idx])
        assert torch.allclose(pairlist_output.d_ij[_][0], edge_weight[idx])
        assert torch.allclose(pairlist_output.r_ij[_], -edge_vec[idx])


def test_tensornet_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function
    # to the output of the modelforge radial symmetry function
    # TODO: only 'expnorm' from TensorNet implemented

    import torch
    from openff.units import unit

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.utils import TensorNetRadialSymmetryFunction
    from modelforge.tests.precalculated_values import prepare_values_for_test_tensornet_compare_radial_symmetry_features

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_radial_symmetry_features.pt"
    # reference_data = None

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5  # NOTE: angstrom

    # TensorNet constants
    radial_cutoff = 5.0
    radial_start = 0.0  # cutoff_lower also affect cutoff function in torchmd-net
    radial_dist_divisions = 8

    rsf = TensorNetRadialSymmetryFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    mf_r = rsf(d_ij / 10)  # torch.Size([5, 1, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)
    # cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom, representation_unit=unit.angstrom)

    rcut_ij = cutoff_module(d_ij / 10)  # torch.Size([5, 1]) # NOTE: nanometer
    mf_r = mf_r * rcut_ij.unsqueeze(-1)

    if reference_data:
        tn_r = torch.load(reference_data)
    else:
        tn_r = prepare_values_for_test_tensornet_compare_radial_symmetry_features(
            d_ij,
            radial_start,
            radial_cutoff,
            radial_dist_divisions,
            trainable=False,
            seed=seed,
        )

    assert torch.allclose(mf_r, tn_r)


def test_tensornet_representation():
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.potential.tensornet import TensorNetRepresentation
    from modelforge.tests.precalculated_values import prepare_values_for_test_tensornet_representation

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_representation.pt"
    # reference_data = None

    hidden_channels = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1
    trainable_rbf = False
    max_z = 128
    dtype = torch.float32
    # representation_unit = unit.angstrom

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    mf_input = next(iter(dataset.train_dataloader())).nnp_input
    # modelforge TensorNet
    torch.manual_seed(seed)
    model = TensorNet()
    # model = TensorNet(representation_unit=representation_unit)
    model.input_preparation._input_checks(mf_input)
    pairlist_output = model.input_preparation.prepare_inputs(mf_input)

    ################ modelforge TensorNet ################
    torch.manual_seed(seed)
    tensornet_representation_module = TensorNetRepresentation(
        hidden_channels,
        num_rbf,
        act_class,
        cutoff_upper * unit.angstrom,
        cutoff_lower * unit.angstrom,
        trainable_rbf,
        max_z,
        dtype,
        # representation_unit,
    )
    # tensornet_representation_module = model.core_module.representation_module
    nnp_input = model.core_module._model_specific_input_preparation(mf_input, pairlist_output)
    mf_X = tensornet_representation_module(nnp_input)
    ################ modelforge TensorNet ################

    ################ torchmd-net TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_representation(
            nnp_input,
            hidden_channels,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            trainable_rbf,
            max_z,
            seed,
        )
    ################ torchmd-net TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


def test_tensornet_interaction():
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.potential.tensornet import TensorNetInteraction
    from modelforge.tests.precalculated_values import prepare_values_for_test_tensornet_interaction

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_interaction.pt"
    # reference_data = None

    hidden_channels = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1
    dtype = torch.float32
    # representation_unit = unit.angstrom

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    mf_input = next(iter(dataset.train_dataloader())).nnp_input
    # modelforge TensorNet
    torch.manual_seed(seed)
    model = TensorNet()
    # model = TensorNet(representation_unit=unit.angstrom)
    model.input_preparation._input_checks(mf_input)
    pairlist_output = model.input_preparation.prepare_inputs(mf_input)

    ################ modelforge TensorNet ################
    tensornet_representation_module = model.core_module.representation_module
    nnp_input = model.core_module._model_specific_input_preparation(mf_input, pairlist_output)
    X = tensornet_representation_module(nnp_input)

    radial_feature_vector = tensornet_representation_module.radial_symmetry_function(
        nnp_input.d_ij
    )
    # _d_ij_in_representation_unit = (nnp_input.d_ij * unit.nanometer).to(representation_unit).m
    # radial_feature_vector = tensornet_representation_module.radial_symmetry_function(
    #     _d_ij_in_representation_unit
    # )
    rcut_ij = tensornet_representation_module.cutoff_module(nnp_input.d_ij)
    radial_feature_vector = radial_feature_vector * rcut_ij.unsqueeze(-1)

    total_charge = torch.zeros_like(nnp_input.atomic_numbers)

    # interaction
    torch.manual_seed(seed)
    interaction_module = TensorNetInteraction(
        hidden_channels,
        num_rbf,
        act_class,
        cutoff_upper * unit.angstrom,
        "O(3)",
        dtype,
        # representation_unit,
    )
    mf_X = interaction_module(
        X,
        nnp_input.pair_indices,
        nnp_input.d_ij.squeeze(-1),
        radial_feature_vector.squeeze(1),
        total_charge,
    )
    ################ modelforge TensorNet ################

    ################ TensorNet ################

    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_interaction(
            X,
            nnp_input,
            radial_feature_vector,
            total_charge,
            hidden_channels,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            seed,
        )
    ################ TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # test_tensornet_input()

    test_tensornet_compare_radial_symmetry_features()

    # test_tensornet_representation()

def test_tensornet_init():
    from modelforge.potential.tensornet import TensorNet

    net = TensorNet()
    assert net is not None


def test_tensornet_forward():
    # Set up a dataset
    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

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


def test_model_input():
    import torch
    from openff.units import unit
    from torchmdnet.models.utils import OptimizedDistance

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet

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
    z, pos, batch = (
        mf_input.atomic_numbers,
        mf_input.positions,
        mf_input.atomic_subsystem_indices
    )
    distance_module = OptimizedDistance(
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=153,
        return_vecs=True,
        loop=False,
        check_errors=False,
        resize_to_fit=False,  # not self.static_shapes
        box=None,
        long_edge_index=False,
    )

    edge_index, edge_weight, edge_vec = distance_module(pos, batch, None)

    # reshape and compare
    pair_indices = pairlist_output.pair_indices.transpose(0, 1)
    edge_index = edge_index.transpose(0, 1)
    for _, pair_index in enumerate(pair_indices):
        idx = ((edge_index == pair_index).sum(axis=1) == 2).nonzero()[0][0]  # select [True, True]
        print(pairlist_output.d_ij[_][0])
        print(edge_weight[idx])
        assert torch.allclose(pairlist_output.d_ij[_][0], edge_weight[idx])
        assert torch.allclose(pairlist_output.r_ij[_], -edge_vec[idx])


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function
    # to the output of the modelforge radial symmetry function
    # TODO: only 'expnorm' from TensorNet implemented

    import torch
    from openff.units import unit
    from torchmdnet.models.utils import ExpNormalSmearing

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.utils import TensorNetRadialSymmetryFunction

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5  # NOTE: angstrom

    # TensorNet constants
    radial_cutoff = 5.0
    radial_start = 0.0
    radial_dist_divisions = 8

    rsf = TensorNetRadialSymmetryFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    r_mf = rsf(d_ij / 10)  # torch.Size([5, 1, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(radial_cutoff * unit.angstrom)

    rcut_ij = cutoff_module(d_ij / 10).unsqueeze(-1)  # torch.Size([5, 1, 1]) # NOTE: nanometer
    r_mf = r_mf * rcut_ij

    rsf_tn = ExpNormalSmearing(
        cutoff_lower=radial_start,
        cutoff_upper=radial_cutoff,
        num_rbf=radial_dist_divisions,
        trainable=False,
    )
    r_tn = rsf_tn(d_ij)

    assert torch.allclose(r_mf, r_tn)


def test_representation():
    import torch
    from openff.units import unit
    from torch import nn
    from torchmdnet.models.model import create_model, load_model
    from torchmdnet.models.tensornet import TensorEmbedding
    from torchmdnet.models.utils import ExpNormalSmearing, OptimizedDistance

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.tensornet import TensorNetNeuralNetworkData
    from modelforge.potential.tensornet import TensorNetRepresentation
    from modelforge.potential.utils import TensorNetRadialSymmetryFunction

    num_atoms = 5
    hidden_channels = 2
    num_rbf = 8
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.0
    trainable_rbf = False
    max_z = 128
    dtype = torch.float32

    import torch
    from openff.units import unit
    from torchmdnet.models.utils import OptimizedDistance

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet

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

    ################ TensorNet ################
    # TensorNet embedding modules setup
    tensor_embedding = TensorEmbedding(
        hidden_channels,
        num_rbf,
        act_class,
        cutoff_lower,
        cutoff_upper,
        trainable_rbf,
        max_z,
        dtype,
    )
    z, pos, batch = (
        mf_input.atomic_numbers,
        mf_input.positions,
        mf_input.atomic_subsystem_indices
    )
    distance_module = OptimizedDistance(
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_pairs=153,
        return_vecs=True,
        loop=False,
        check_errors=False,
        resize_to_fit=False,  # not self.static_shapes
        box=None,
        long_edge_index=False,
    )
    distance_expansion = ExpNormalSmearing(
        cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
    )

    # calculate embedding
    edge_index, edge_weight, edge_vec = distance_module(pos, batch, None)
    edge_attr = distance_expansion(edge_weight)
    X_tn = tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
    ################ TensorNet ################

    ################ modelforge TensorNet ################
    tensornet_representation_module = model.core_module.tensornet_representation_module
    nnp_input = model.core_module._model_specific_input_preparation(mf_input, pairlist_output)
    tensornet_representation_module(nnp_input)

    ################ modelforge TensorNet ################

    return 0


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    # test_compare_radial_symmetry_features()

    # test_compare_radial_symmetry_features

    test_representation()

    # test_model_input(test_model_input)

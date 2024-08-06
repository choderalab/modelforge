def test_init():
    """Test initialization of the TensorNet model."""
    from modelforge.potential.tensornet import TensorNet
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    assert tensornet is not None, "TensorNet model should be initialized."


def test_forward():  # TODO
    import torch

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    seed = 0
    torch.manual_seed(seed)

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=64,
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

    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    tensornet(batch)


def test_input():
    import torch
    from openff.units import unit

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_input,
    )
    from modelforge.tests.test_models import load_configs_into_pydantic_models

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
    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    tensornet.compute_interacting_pairs._input_checks(mf_input)
    pairlist_output = tensornet.compute_interacting_pairs.prepare_inputs(mf_input)

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
        idx = ((edge_index == pair_index).sum(axis=1) == 2).nonzero()[0][
            0
        ]  # select [True, True]
        assert torch.allclose(pairlist_output.d_ij[_][0], edge_weight[idx])
        assert torch.allclose(pairlist_output.r_ij[_], -edge_vec[idx])


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function
    # to the output of the modelforge radial symmetry function
    # TODO: only 'expnorm' from TensorNet implemented
    import torch
    from openff.units import unit

    from modelforge.potential.utils import CosineCutoff
    from modelforge.potential.utils import TensorNetRadialBasisFunction
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_compare_radial_symmetry_features,
    )

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_radial_symmetry_features.pt"
    # reference_data = None

    # generate a random list of distances, all < 5
    d_ij = torch.rand(5, 1) * 5  # NOTE: angstrom

    # TensorNet constants
    maximum_interaction_radius = 5.1
    minimum_interaction_radius = 0.0  # cutoff_lower also affect cutoff function in torchmd-net
    number_of_per_atom_features = 8

    rsf = TensorNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_per_atom_features,
        max_distance=maximum_interaction_radius * unit.angstrom,
        min_distance=minimum_interaction_radius * unit.angstrom,
        alpha=((maximum_interaction_radius - minimum_interaction_radius) / 5.0 * unit.angstrom),
    )
    mf_r = rsf(d_ij / 10)  # torch.Size([5, 8]) # NOTE: nanometer
    cutoff_module = CosineCutoff(maximum_interaction_radius * unit.angstrom)

    rcut_ij = cutoff_module(d_ij / 10)  # torch.Size([5, 1]) # NOTE: nanometer
    mf_r = (mf_r * rcut_ij).unsqueeze(1)

    if reference_data:
        tn_r = torch.load(reference_data)
    else:
        tn_r = prepare_values_for_test_tensornet_compare_radial_symmetry_features(
            d_ij,
            minimum_interaction_radius,
            maximum_interaction_radius,
            number_of_per_atom_features,
            trainable=False,
            seed=seed,
        )

    assert torch.allclose(mf_r, tn_r)


def test_representation():
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.potential.tensornet import TensorNetRepresentation
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_representation,
    )
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_representation.pt"
    # reference_data = None

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1
    trainable_rbf = False
    highest_atomic_number = 128

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
    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    tensornet.compute_interacting_pairs._input_checks(mf_input)
    pairlist_output = tensornet.compute_interacting_pairs.prepare_inputs(mf_input)

    ################ modelforge TensorNet ################
    torch.manual_seed(seed)
    tensornet_representation_module = TensorNetRepresentation(
        number_of_per_atom_features,
        num_rbf,
        act_class,
        cutoff_upper * unit.angstrom,
        cutoff_lower * unit.angstrom,
        trainable_rbf,
        highest_atomic_number,
    )
    nnp_input = tensornet.core_module._model_specific_input_preparation(
        mf_input, pairlist_output
    )
    mf_X, _ = tensornet_representation_module(nnp_input)
    ################ modelforge TensorNet ################

    ################ torchmd-net TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_representation(
            nnp_input,
            number_of_per_atom_features,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            trainable_rbf,
            highest_atomic_number,
            seed,
        )
    ################ torchmd-net TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


def test_interaction():
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.potential.tensornet import TensorNetInteraction
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_interaction,
    )
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    seed = 0
    torch.manual_seed(seed)

    reference_data = "modelforge/tests/data/tensornet_interaction.pt"
    # reference_data = None

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1

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
    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    tensornet.compute_interacting_pairs._input_checks(mf_input)
    pairlist_output = tensornet.compute_interacting_pairs.prepare_inputs(mf_input)

    ################ modelforge TensorNet ################
    tensornet_representation_module = tensornet.core_module.representation_module
    nnp_input = tensornet.core_module._model_specific_input_preparation(
        mf_input, pairlist_output
    )
    X, _ = tensornet_representation_module(nnp_input)

    radial_feature_vector = tensornet_representation_module.radial_symmetry_function(
        nnp_input.d_ij
    )
    rcut_ij = tensornet_representation_module.cutoff_module(nnp_input.d_ij)
    radial_feature_vector = (radial_feature_vector * rcut_ij).unsqueeze(1)

    atomic_charges = torch.zeros_like(nnp_input.atomic_numbers)

    # interaction
    torch.manual_seed(seed)
    interaction_module = TensorNetInteraction(
        number_of_per_atom_features,
        num_rbf,
        act_class,
        cutoff_upper * unit.angstrom,
        "O(3)",
    )
    mf_X = interaction_module(
        X,
        nnp_input.pair_indices,
        nnp_input.d_ij.squeeze(-1),
        radial_feature_vector.squeeze(1),
        atomic_charges,
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
            atomic_charges,
            number_of_per_atom_features,
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

    # test_forward()

    # test_input()

    # test_compare_radial_symmetry_features()

    # test_representation()

    test_interaction()

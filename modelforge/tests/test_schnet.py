import pytest
import torch
from modelforge.tests.precalculated_values import (
    load_precalculated_schnet_results,
    setup_single_methane_input,
)
from typing import Optional


def setup_schnet_model(potential_seed: Optional[int] = None):
    from modelforge.tests.test_potentials import load_configs_into_pydantic_models
    from modelforge.potential import NeuralNetworkPotentialFactory

    # read default parameters
    config = load_configs_into_pydantic_models("schnet", "qm9")
    # override defaults to match reference implementation in spk
    config[
        "potential"
    ].core_parameter.featurization.atomic_number.number_of_per_atom_features = 12
    config["potential"].core_parameter.number_of_radial_basis_functions = 5
    config["potential"].core_parameter.number_of_filters = 12

    model = NeuralNetworkPotentialFactory.generate_trainer(
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
        potential_seed=potential_seed,
    ).lightning_module.potential
    return model


def test_init():
    """Test initialization of the Schnet model."""
    schnet = setup_schnet_model()
    assert schnet is not None, "Schnet model should be initialized."


def test_compare_rbf():
    # compare schnetpack RadialSymmetryFunction with modelforge RadialSymmetryFunction
    from modelforge.potential import SchnetRadialBasisFunction
    from openff.units import unit

    # Initialize the RBFs
    number_of_gaussians = 10
    cutoff = unit.Quantity(5.2, unit.angstrom).to(unit.nanometer).m
    start = unit.Quantity(0.8, unit.angstrom).to(unit.nanometer).m

    rbf_module = SchnetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_gaussians,
        max_distance=cutoff,
        min_distance=start,
    )

    # define pariwise distances
    d_ij = torch.tensor([[1.0077], [4.2496], [2.8202], [3.4342], [9.2465]])

    # this has been calculated with schnetpack2.0
    reference_rbf_output = torch.tensor(
        [
            [
                [
                    9.1371e-01,
                    8.4755e-01,
                    2.8922e-01,
                    3.6308e-02,
                    1.6768e-03,
                    2.8488e-05,
                    1.7805e-07,
                    4.0939e-10,
                    3.4629e-13,
                    1.0776e-16,
                ]
            ],
            [
                [
                    1.5448e-11,
                    1.0867e-08,
                    2.8121e-06,
                    2.6772e-04,
                    9.3763e-03,
                    1.2081e-01,
                    5.7260e-01,
                    9.9843e-01,
                    6.4046e-01,
                    1.5114e-01,
                ]
            ],
            [
                [
                    1.9595e-04,
                    7.4063e-03,
                    1.0298e-01,
                    5.2678e-01,
                    9.9130e-01,
                    6.8625e-01,
                    1.7477e-01,
                    1.6374e-02,
                    5.6435e-04,
                    7.1557e-06,
                ]
            ],
            [
                [
                    4.9634e-07,
                    6.5867e-05,
                    3.2156e-03,
                    5.7752e-02,
                    3.8157e-01,
                    9.2744e-01,
                    8.2929e-01,
                    2.7279e-01,
                    3.3011e-02,
                    1.4696e-03,
                ]
            ],
            [
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    5.6052e-45,
                    5.2717e-39,
                    1.8660e-33,
                    2.4297e-28,
                    1.1639e-23,
                    2.0511e-19,
                    1.3297e-15,
                ]
            ],
        ]
    )

    assert torch.allclose(
        reference_rbf_output,
        rbf_module(d_ij / 10).unsqueeze(1),
        atol=1e-5,
    )  # NOTE: there is a shape mismatch between the two outputs


def test_compare_implementation_against_reference_implementation():
    # ---------------------------------------- #
    # test the implementation of the representation part of the PaiNN model
    # ---------------------------------------- #
    model = setup_schnet_model(1234).double()
    # ------------------------------------ #
    # reference values
    # generated with schnetpack2.0

    # ------------------------------------ #
    # set up the input for the spk Schnet model
    input = setup_single_methane_input()
    spk_input = input["spk_methane_input"]
    model_input = input["modelforge_methane_input"]

    # ---------------------------------------- #
    # test forward pass
    # ---------------------------------------- #
    # reset
    torch.manual_seed(1234)
    for i in range(3):
        model.core_network.interaction_modules[i].intput_to_feature.reset_parameters()
        for j in range(2):
            model.core_network.interaction_modules[i].feature_to_output[
                j
            ].reset_parameters()
            model.core_network.interaction_modules[i].filter_network[
                j
            ].reset_parameters()

    calculated_results = model.compute_core_network_output(model_input)
    reference_results = load_precalculated_schnet_results()
    assert (
        reference_results["scalar_representation"].shape
        == calculated_results["per_atom_scalar_representation"].shape
    )

    scalar_spk = reference_results["scalar_representation"]
    scalar_mf = calculated_results["per_atom_scalar_representation"]
    assert torch.allclose(scalar_spk, scalar_mf, atol=1e-4)

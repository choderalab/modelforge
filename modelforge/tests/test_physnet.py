def test_init():

    from modelforge.potential.physnet import PhysNet

    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"physnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    model = PhysNet(**potential_parameter)


def test_forward(single_batch_with_batchsize_64):
    import torch
    from modelforge.potential.physnet import PhysNet

    # read default parameters
    from modelforge.tests.test_models import load_configs

    # read default parameters
    config = load_configs(f"physnet_without_ase", "qm9")
    # Extract parameters
    potential_parameter = config["potential"].get("potential_parameter", {})

    # Extract parameters
    potential_parameter["number_of_modules"] = 1
    potential_parameter["number_of_interaction_residual"] = 1

    model = PhysNet(**potential_parameter)
    model = model.to(torch.float32)
    print(model)
    yhat = model(single_batch_with_batchsize_64.nnp_input.to(dtype=torch.float32))


def test_compare_representation():
    # This test compares the RBF calculation of the original
    # PhysNet implemntation against the SAKE/PhysNet implementation in modelforge
    # # NOTE: input in PhysNet is expected in angstrom, in contrast to modelforge which expects input in nanomter

    import numpy as np
    import torch
    from openff.units import unit

    # set up test parameters
    number_of_radial_basis_functions = K = 20
    _max_distance_in_nanometer = 0.5
    #############################
    # RBF comparision
    #############################
    # Initialize the rbf class
    from modelforge.potential.utils import PhysNetRadialSymmetryFunction

    mf_rbf = PhysNetRadialSymmetryFunction(
        number_of_radial_basis_functions,
        max_distance=_max_distance_in_nanometer * unit.nanometer,
    )

    # compare the rbf output
    #################
    # PhysNet implementation
    from .precalculated_values import provide_reference_for_test_physnet_test_rbf

    pn_rbf_output = provide_reference_for_test_physnet_test_rbf()
    D = np.array([[1.0394776], [3.375541]], dtype=np.float32)

    mf_rbf_output = mf_rbf(torch.tensor(D / 10).squeeze())
    assert np.allclose(np.flip(pn_rbf_output.squeeze(), axis=1), mf_rbf_output.numpy())

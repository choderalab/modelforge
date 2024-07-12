from modelforge.potential.spookynet import SpookyNet
from spookynet import SpookyNet as RefSpookyNet
from modelforge.tests.precalculated_values import (
    setup_single_methane_input,
)
import torch

import pytest


def test_init():
    """Test initialization of the SpookyNet model."""
    from modelforge.potential.spookynet import SpookyNet

    from modelforge.tests.test_models import load_configs

    # load default parameters
    config = load_configs(f"spookynet", "qm9")
    # initialize model
    spookynet = SpookyNet(
        **config["potential"]["core_parameter"],
        postprocessing_parameter=config["potential"]["postprocessing_parameter"],
    )
    assert spookynet is not None, "Schnet model should be initialized."


from openff.units import unit


def test_forward():
    # ---------------------------------------- #
    # test the implementation of the representation part of the PaiNN model
    # ---------------------------------------- #
    from modelforge.potential.spookynet import SpookyNet

    from modelforge.tests.test_models import load_configs

    # load default parameters
    config = load_configs(f"spookynet", "qm9")

    # override default parameters
    config["potential"]["core_parameter"]["number_of_atom_features"] = 12
    config["potential"]["core_parameter"]["number_of_radial_basis_functions"] = 5

    torch.manual_seed(1234)

    # initialize model
    spookynet = SpookyNet(
        **config["potential"]["core_parameter"],
        postprocessing_parameter=config["potential"]["postprocessing_parameter"],
    ).double()

    input = setup_single_methane_input()
    model_input = input["modelforge_methane_input"]


    spookynet.input_preparation._input_checks(model_input)

    pairlist_output = spookynet.input_preparation.prepare_inputs(model_input)
    print(f"{pairlist_output.d_ij.shape=}")
    prepared_input = spookynet.core_module._model_specific_input_preparation(
        model_input, pairlist_output
    )
    calculated_results = spookynet.core_module.forward(model_input, pairlist_output)


def test_spookynet_forward(single_batch_with_batchsize_64, model_parameter):
    """
    Test the forward pass of the SpookyNet model.
    """
    print(f"model_parameter: {model_parameter}")
    (
        nr_atom_basis,
        max_atomic_number,
        number_of_gaussians,
        cutoff,
        nr_interaction_blocks,
    ) = model_parameter
    spookynet = SpookyNet(
        number_of_atom_features=nr_atom_basis,
        max_Z=max_atomic_number,
        number_of_radial_basis_functions=number_of_gaussians,
        cutoff=cutoff,
        number_of_interaction_modules=nr_interaction_blocks,
    )
    energy = spookynet(single_batch_with_batchsize_64.nnp_input).E
    nr_of_mols = single_batch_with_batchsize_64.nnp_input.atomic_subsystem_indices.unique().shape[
        0
    ]

    assert (
            len(energy) == nr_of_mols
    )  # Assuming energy is calculated per sample in the batch


def make_random_pairlist(nr_atoms, nr_pairs, include_self_pairs):
    if include_self_pairs:
        assert nr_pairs <= nr_atoms ** 2, """Number of pairs requested is more than the number of possible pairs."""
        nr_pairs_choose = nr_pairs - nr_atoms

    else:
        assert nr_pairs <= nr_atoms ** 2 - nr_atoms, """Number of pairs requested is more than the number of possible 
        pairs."""
        nr_pairs_choose = nr_pairs
    assert nr_pairs_choose >= 0, """Number of pairs must be greater than or equal to the number of atoms if "
        include_self_pairs is True or greater than 0 if include_self_pairs is False."""
    all_pairs = torch.cartesian_prod(torch.arange(nr_atoms), torch.arange(nr_atoms))
    self_pairs = all_pairs.T[0] == all_pairs.T[1]
    non_self_pairs = all_pairs[~self_pairs]
    perm = torch.randperm(non_self_pairs.size(0))
    idx = perm[:nr_pairs_choose]
    pairlist = non_self_pairs[idx]
    if include_self_pairs:
        pairlist = torch.cat(
            [pairlist, all_pairs[self_pairs]], dim=0
        )

    return pairlist.T


def test_atomic_properties_static():
    ref_spookynet = RefSpookyNet()

    nr_atoms = 5
    geometry_basis = 3
    nr_pairs = 7
    idx_i, idx_j = make_random_pairlist(nr_atoms, nr_pairs, False)

    Z = torch.randint(1, 100, (nr_atoms,))
    R = torch.rand((nr_atoms, geometry_basis))
    print(ref_spookynet._atomic_properties_static(Z, R, idx_i, idx_j))


def test_spookynet_interaction_module_forward():
    from modelforge.potential.spookynet import SpookyNetInteractionModule
    N = 5
    P = 19
    num_features = 7
    number_of_radial_basis_functions = 5
    spookynet_interaction_module = SpookyNetInteractionModule(
        num_features=num_features,
        num_basis_functions=number_of_radial_basis_functions,
        num_residual_pre=3,
        num_residual_local_x=3,
        num_residual_local_s=3,
        num_residual_local_p=3,
        num_residual_local_d=3,
        num_residual_local=3,
        num_residual_nonlocal_q=19,
        num_residual_nonlocal_k=13,
        num_residual_nonlocal_v=17,
        num_residual_post=3,
        num_residual_output=3
    )

    x = torch.rand((N, num_features))
    f_ij = torch.rand((P, number_of_radial_basis_functions))
    f_ij_cutoff = torch.rand((P, 1))
    p_orbital_ij = torch.rand((P, 1))
    d_orbital_ij = torch.rand((P, 1))
    pairlist = make_random_pairlist(N, P, include_self_pairs=False)
    spookynet_interaction_module(x, pairlist, f_ij, f_ij_cutoff, p_orbital_ij, d_orbital_ij)


def test_spookynet_interaction_module_against_reference():
    from modelforge.potential.spookynet import SpookyNetInteractionModule as MfSpookyNetInteractionModule
    from spookynet.modules.interaction_module import InteractionModule as RefSpookyNetInteractionModule
    N = 5
    P = 19
    num_features = 7
    number_of_radial_basis_functions = 5
    num_residual_all = 3
    mf_spookynet_interaction_module = MfSpookyNetInteractionModule(
        num_features=num_features,
        num_basis_functions=number_of_radial_basis_functions,
        num_residual_pre=num_residual_all,
        num_residual_local_x=num_residual_all,
        num_residual_local_s=num_residual_all,
        num_residual_local_p=num_residual_all,
        num_residual_local_d=num_residual_all,
        num_residual_local=num_residual_all,
        num_residual_nonlocal_q=num_residual_all,
        num_residual_nonlocal_k=num_residual_all,
        num_residual_nonlocal_v=num_residual_all,
        num_residual_post=num_residual_all,
        num_residual_output=num_residual_all
    ).to(torch.double)

    ref_spookynet_interaction_module = RefSpookyNetInteractionModule(
        num_features=num_features,
        num_basis_functions=number_of_radial_basis_functions,
        num_residual_pre=num_residual_all,
        num_residual_local_x=num_residual_all,
        num_residual_local_s=num_residual_all,
        num_residual_local_p=num_residual_all,
        num_residual_local_d=num_residual_all,
        num_residual_local=num_residual_all,
        num_residual_nonlocal_q=num_residual_all,
        num_residual_nonlocal_k=num_residual_all,
        num_residual_nonlocal_v=num_residual_all,
        num_residual_post=num_residual_all,
        num_residual_output=num_residual_all
    ).to(torch.double)

    for (_, mf_param), (_, ref_param) in zip(mf_spookynet_interaction_module.named_parameters(),
                                             ref_spookynet_interaction_module.named_parameters()):
        mf_param.requires_grad = False
        mf_param[:] = ref_param

    assert len(list(mf_spookynet_interaction_module.resblock.named_parameters())) == len(
        list(ref_spookynet_interaction_module.resblock.named_parameters()))
    for (mf_name, mf_param), (ref_name, ref_param) in zip(mf_spookynet_interaction_module.resblock.named_parameters(),
                                                          ref_spookynet_interaction_module.resblock.named_parameters()):
        print(f"{mf_name=} {ref_name=}")
        if not torch.equal(mf_param, ref_param):
            print(f"{mf_param=} {ref_param=}")
        else:
            print("parameters are the same")

    x = torch.rand((N, num_features), dtype=torch.double)
    f_ij = torch.rand((P, number_of_radial_basis_functions), dtype=torch.double)
    f_ij_cutoff = torch.rand((P, 1), dtype=torch.double)
    p_orbital_ij = torch.rand((P, 1), dtype=torch.double)
    d_orbital_ij = torch.rand((P, 1), dtype=torch.double)
    pairlist = make_random_pairlist(N, P, include_self_pairs=False)
    idx_i, idx_j = pairlist
    mf_x_result, mf_y_result = mf_spookynet_interaction_module(x, pairlist, f_ij, f_ij_cutoff, p_orbital_ij,
                                                               d_orbital_ij)
    ref_x_result, ref_y_result = ref_spookynet_interaction_module(x, f_ij * f_ij_cutoff, p_orbital_ij,
                                                                  d_orbital_ij, idx_i, idx_j, 1, None, None)
    assert torch.equal(mf_x_result, ref_x_result)
    assert torch.equal(mf_y_result, ref_y_result)


def test_spookynet_bernstein_polynomial_equivalence():
    from spookynet.modules.exponential_bernstein_polynomials import ExponentialBernsteinPolynomials as RefExponentialBernsteinPolynomials
    from modelforge.potential.utils import ExponentialBernsteinRadialBasisFunction as MfExponentialBernSteinPolynomials

    num_basis_functions = 3
    ref_exp_bernstein_polynomials = RefExponentialBernsteinPolynomials(num_basis_functions, exp_weighting=True)
    mf_exp_bernstein_polynomials = MfExponentialBernSteinPolynomials(num_basis_functions, ini_alpha=1.0)  # TODO: put
    # the right number

    N = 5
    r_angstrom = torch.rand((N, 1))
    r_nanometer = r_angstrom * 0.1
    cutoff_values = torch.rand((N, 1))
    ref_exp_bernstein_polynomial_result = ref_exp_bernstein_polynomials(r_angstrom, cutoff_values)
    mf_exp_bernstein_polynomial_result = mf_exp_bernstein_polynomials(r_nanometer) * cutoff_values
    print(f"{ref_exp_bernstein_polynomial_result=}")
    print(f"{mf_exp_bernstein_polynomial_result=}")
    assert torch.allclose(ref_exp_bernstein_polynomial_result, mf_exp_bernstein_polynomial_result)

from modelforge.dataset.dataset import NNPInput
from modelforge.potential.spookynet import SpookyNet
from spookynet import SpookyNet as RefSpookyNet
from modelforge.tests.precalculated_values import (
    setup_single_methane_input,
)
import torch
from icecream import ic
import pytest

from modelforge.utils.units import _convert


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
    config["potential"]["core_parameter"]["number_of_atom_features"] = 11
    config["potential"]["core_parameter"]["number_of_radial_basis_functions"] = 7
    config["potential"]["core_parameter"]["number_of_residual_blocks"] = 1
    config["potential"]["core_parameter"]["number_of_interaction_modules"] = 1
    config["potential"]["core_parameter"]["cutoff"] = "2.4 meter"

    print(f"{config['potential']['core_parameter']}=")

    torch.manual_seed(1234)

    # initialize model
    spookynet = SpookyNet(
        **config["potential"]["core_parameter"],
        postprocessing_parameter=config["potential"]["postprocessing_parameter"],
    ).double()

    single_methane = setup_single_methane_input()["modelforge_methane_input"]
    double_methane = NNPInput(
        atomic_numbers=torch.cat([single_methane.atomic_numbers] * 2, dim=0),
        positions=torch.cat([single_methane.positions] * 2, dim=0),
        atomic_subsystem_indices=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        total_charge=torch.cat([single_methane.total_charge] * 2, dim=0),
    )
    print(f"{torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int64).dtype=}")
    print(f"test: {double_methane.atomic_subsystem_indices.dtype=}")
    ic(double_methane)
    single_methane.positions = single_methane.positions.double()
    single_methane.total_charge = single_methane.total_charge.double()
    double_methane.positions = double_methane.positions.double()
    double_methane.total_charge = double_methane.total_charge.double()

    print(f"test: {double_methane.atomic_subsystem_indices.dtype=}")
    spookynet.input_preparation._input_checks(single_methane)
    single_pairlist_output = spookynet.input_preparation.prepare_inputs(single_methane)

    spookynet.input_preparation._input_checks(double_methane)
    print(f"test: {double_methane.atomic_subsystem_indices.dtype=}")
    double_pairlist_output = spookynet.input_preparation.prepare_inputs(double_methane)
    print(f"test: {double_methane.atomic_subsystem_indices.dtype=}")
    calculated_results = spookynet.core_module.forward(double_methane, double_pairlist_output)

    ref_spookynet = RefSpookyNet(
        num_features=config["potential"]["core_parameter"]["number_of_atom_features"],
        num_basis_functions=config["potential"]["core_parameter"]["number_of_radial_basis_functions"],
        num_modules=config["potential"]["core_parameter"]["number_of_interaction_modules"],
        num_residual_electron=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_pre=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_local_x=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_local_s=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_local_p=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_local_d=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_local=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_nonlocal_q=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_nonlocal_k=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_nonlocal_v=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_post=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        num_residual_output=config["potential"]["core_parameter"]["number_of_residual_blocks"],
        use_zbl_repulsion=False,
        use_electrostatics=False,
        use_d4_dispersion=False,
        cutoff=_convert(config["potential"]["core_parameter"]["cutoff"]).m_as(unit.angstrom),
    ).double()

    for param in ref_spookynet.parameters():
        torch.nn.init.normal_(param, 5.0, 3.0)

    spookynet.core_module.atomic_shift = ref_spookynet.element_bias
    spookynet.core_module.atomic_embedding_module.element_embedding.weight = ref_spookynet.nuclear_embedding.element_embedding
    spookynet.core_module.atomic_embedding_module.config_linear = ref_spookynet.nuclear_embedding.config_linear.weight

    spookynet.core_module.charge_embedding_module.linear_q.weight = ref_spookynet.charge_embedding.linear_q.weight
    spookynet.core_module.charge_embedding_module.linear_q.bias = ref_spookynet.charge_embedding.linear_q.bias
    spookynet.core_module.charge_embedding_module.linear_k.weight = ref_spookynet.charge_embedding.linear_k.weight
    spookynet.core_module.charge_embedding_module.linear_v.weight = ref_spookynet.charge_embedding.linear_v.weight
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].activation1.alpha = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].activation1.alpha
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].activation1.beta = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].activation1.beta
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].linear1.weight = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].linear1.weight
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].activation2.alpha = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].activation2.alpha
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].activation2.beta = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].activation2.beta
    spookynet.core_module.charge_embedding_module.resmlp.residual.stack[0].linear2.weight = \
        ref_spookynet.charge_embedding.resblock.residual.stack[0].linear2.weight
    spookynet.core_module.charge_embedding_module.resmlp.activation.alpha = ref_spookynet.charge_embedding.resblock.activation.alpha
    spookynet.core_module.charge_embedding_module.resmlp.activation.beta = ref_spookynet.charge_embedding.resblock.activation.beta
    spookynet.core_module.charge_embedding_module.resmlp.linear.weight = ref_spookynet.charge_embedding.resblock.linear.weight

    spookynet.core_module.spookynet_representation_module.radial_symmetry_function_module.alpha = ref_spookynet.radial_basis_functions._alpha

    spookynet.core_module.interaction_modules[0].local_interaction.radial_s.weight = ref_spookynet.module[
        0].local_interaction.radial_s.weight
    spookynet.core_module.interaction_modules[0].local_interaction.radial_p.weight = ref_spookynet.module[
        0].local_interaction.radial_p.weight
    spookynet.core_module.interaction_modules[0].local_interaction.radial_d.weight = ref_spookynet.module[
        0].local_interaction.radial_d.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].local_interaction.resblock_x.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.activation.alpha = ref_spookynet.module[
        0].local_interaction.resblock_x.activation.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.activation.beta = ref_spookynet.module[
        0].local_interaction.resblock_x.activation.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.linear.weight = ref_spookynet.module[
        0].local_interaction.resblock_x.linear.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_x.linear.bias = ref_spookynet.module[
        0].local_interaction.resblock_x.linear.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].local_interaction.resblock_s.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.activation.alpha = ref_spookynet.module[
        0].local_interaction.resblock_s.activation.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.activation.beta = ref_spookynet.module[
        0].local_interaction.resblock_s.activation.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.linear.weight = ref_spookynet.module[
        0].local_interaction.resblock_s.linear.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_s.linear.bias = ref_spookynet.module[
        0].local_interaction.resblock_s.linear.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].local_interaction.resblock_p.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.activation.alpha = ref_spookynet.module[
        0].local_interaction.resblock_p.activation.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.activation.beta = ref_spookynet.module[
        0].local_interaction.resblock_p.activation.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.linear.weight = ref_spookynet.module[
        0].local_interaction.resblock_p.linear.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_p.linear.bias = ref_spookynet.module[
        0].local_interaction.resblock_p.linear.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].local_interaction.resblock_d.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.activation.alpha = ref_spookynet.module[
        0].local_interaction.resblock_d.activation.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.activation.beta = ref_spookynet.module[
        0].local_interaction.resblock_d.activation.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.linear.weight = ref_spookynet.module[
        0].local_interaction.resblock_d.linear.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_d.linear.bias = ref_spookynet.module[
        0].local_interaction.resblock_d.linear.bias
    spookynet.core_module.interaction_modules[0].local_interaction.projection_p.weight = ref_spookynet.module[
        0].local_interaction.projection_p.weight
    spookynet.core_module.interaction_modules[0].local_interaction.projection_d.weight = ref_spookynet.module[
        0].local_interaction.projection_d.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].local_interaction.resblock.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.activation.alpha = ref_spookynet.module[
        0].local_interaction.resblock.activation.alpha
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.activation.beta = ref_spookynet.module[
        0].local_interaction.resblock.activation.beta
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.linear.weight = ref_spookynet.module[
        0].local_interaction.resblock.linear.weight
    spookynet.core_module.interaction_modules[0].local_interaction.resmlp_l.linear.bias = ref_spookynet.module[
        0].local_interaction.resblock.linear.bias

    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.activation.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_q.activation.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.activation.beta = ref_spookynet.module[
        0].nonlocal_interaction.resblock_q.activation.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.linear.weight = ref_spookynet.module[
        0].nonlocal_interaction.resblock_q.linear.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_q.linear.bias = ref_spookynet.module[
        0].nonlocal_interaction.resblock_q.linear.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.activation.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_k.activation.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.activation.beta = ref_spookynet.module[
        0].nonlocal_interaction.resblock_k.activation.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.linear.weight = ref_spookynet.module[
        0].nonlocal_interaction.resblock_k.linear.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_k.linear.bias = ref_spookynet.module[
        0].nonlocal_interaction.resblock_k.linear.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.activation.alpha = \
        ref_spookynet.module[0].nonlocal_interaction.resblock_v.activation.alpha
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.activation.beta = ref_spookynet.module[
        0].nonlocal_interaction.resblock_v.activation.beta
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.linear.weight = ref_spookynet.module[
        0].nonlocal_interaction.resblock_v.linear.weight
    spookynet.core_module.interaction_modules[0].nonlocal_interaction.resmlp_v.linear.bias = ref_spookynet.module[
        0].nonlocal_interaction.resblock_v.linear.bias

    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].activation1.alpha = \
        ref_spookynet.module[0].residual_pre.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].activation1.beta = \
        ref_spookynet.module[0].residual_pre.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].linear1.weight = \
        ref_spookynet.module[0].residual_pre.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].linear1.bias = \
        ref_spookynet.module[0].residual_pre.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].activation2.alpha = \
        ref_spookynet.module[0].residual_pre.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].activation2.beta = \
        ref_spookynet.module[0].residual_pre.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].linear2.weight = \
        ref_spookynet.module[0].residual_pre.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].residual_pre.stack[0].linear2.bias = \
        ref_spookynet.module[0].residual_pre.stack[0].linear2.bias

    spookynet.core_module.interaction_modules[0].residual_post.stack[0].activation1.alpha = \
        ref_spookynet.module[0].residual_post.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].activation1.beta = \
        ref_spookynet.module[0].residual_post.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].linear1.weight = \
        ref_spookynet.module[0].residual_post.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].linear1.bias = \
        ref_spookynet.module[0].residual_post.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].activation2.alpha = \
        ref_spookynet.module[0].residual_post.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].activation2.beta = \
        ref_spookynet.module[0].residual_post.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].linear2.weight = \
        ref_spookynet.module[0].residual_post.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].residual_post.stack[0].linear2.bias = \
        ref_spookynet.module[0].residual_post.stack[0].linear2.bias

    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].activation1.alpha = \
        ref_spookynet.module[0].resblock.residual.stack[0].activation1.alpha
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].activation1.beta = \
        ref_spookynet.module[0].resblock.residual.stack[0].activation1.beta
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].linear1.weight = \
        ref_spookynet.module[0].resblock.residual.stack[0].linear1.weight
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].linear1.bias = \
        ref_spookynet.module[0].resblock.residual.stack[0].linear1.bias
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].activation2.alpha = \
        ref_spookynet.module[0].resblock.residual.stack[0].activation2.alpha
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].activation2.beta = \
        ref_spookynet.module[0].resblock.residual.stack[0].activation2.beta
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].linear2.weight = \
        ref_spookynet.module[0].resblock.residual.stack[0].linear2.weight
    spookynet.core_module.interaction_modules[0].resmlp.residual.stack[0].linear2.bias = \
        ref_spookynet.module[0].resblock.residual.stack[0].linear2.bias
    spookynet.core_module.interaction_modules[0].resmlp.activation.alpha = ref_spookynet.module[
        0].resblock.activation.alpha
    spookynet.core_module.interaction_modules[0].resmlp.activation.beta = ref_spookynet.module[
        0].resblock.activation.beta
    spookynet.core_module.interaction_modules[0].resmlp.linear.weight = ref_spookynet.module[0].resblock.linear.weight
    spookynet.core_module.interaction_modules[0].resmlp.linear.bias = ref_spookynet.module[0].resblock.linear.bias
    spookynet.core_module.energy_and_charge_readout.weight = ref_spookynet.output.weight

    ref_spookynet.train()

    # TODO: how are multiple systems passed into the reference SpookyNet
    print(f"test: {double_methane.atomic_subsystem_indices.dtype=}")
    energy, forces, dipole, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = ref_spookynet(
        single_methane.atomic_numbers,
        single_methane.total_charge,
        (single_methane.positions * unit.nanometer).m_as(unit.angstrom),
        single_pairlist_output.pair_indices[0],
        single_pairlist_output.pair_indices[1],
        batch_seg=None,
        num_batch=1,
    )

    ic(calculated_results["per_atom_energy"])
    ic(ea)


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
    from spookynet.modules.exponential_bernstein_polynomials import \
        ExponentialBernsteinPolynomials as RefExponentialBernsteinPolynomials
    from modelforge.potential.utils import ExponentialBernsteinRadialBasisFunction as MfExponentialBernSteinPolynomials

    num_basis_functions = 3
    ref_exp_bernstein_polynomials = RefExponentialBernsteinPolynomials(num_basis_functions, exp_weighting=True)
    mf_exp_bernstein_polynomials = MfExponentialBernSteinPolynomials(num_basis_functions)

    N = 5
    r_angstrom = torch.rand((N, 1))
    r_nanometer = r_angstrom * 0.1
    cutoff_values = torch.rand((N, 1))
    ref_exp_bernstein_polynomial_result = ref_exp_bernstein_polynomials(r_angstrom, cutoff_values)
    mf_exp_bernstein_polynomial_result = mf_exp_bernstein_polynomials(r_nanometer) * cutoff_values
    print(f"{ref_exp_bernstein_polynomial_result=}")
    print(f"{mf_exp_bernstein_polynomial_result=}")
    assert torch.allclose(ref_exp_bernstein_polynomial_result, mf_exp_bernstein_polynomial_result)

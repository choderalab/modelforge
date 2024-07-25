from .test_models import load_configs


def test_embedding(single_batch_with_batchsize_64):
    # test the input featurization, including:
    # - nuclear charge embedding
    # - total charge mixing
    
    import torch

    nnp_input = single_batch_with_batchsize_64.nnp_input
    model_name = "SchNet"
    # read default parameters and extract featurization
    config = load_configs(f"{model_name.lower()}", "qm9")
    featurization_config = config["potential"]["core_parameter"]["featurization"]

    # featurize the atomic input (default is only nuclear charge embedding)
    from modelforge.potential.utils import FeaturizeInput

    featurize_input_module = FeaturizeInput(featurization_config)

    # mixing module should be the identidy operation since only nuclear charge is used
    mixing_module = featurize_input_module.mixing
    assert mixing_module.__module__ == "torch.nn.modules.linear"
    mixing_module_name = str(mixing_module)

    # only nucreal charges embedded
    assert (
        "nuclear_charge_embedding"
        in featurize_input_module.registered_embedding_operations
    )
    assert len(featurize_input_module.registered_embedding_operations) == 1
    # no mixing
    assert "Identity()" in mixing_module_name

    # add total charge to the input
    featurization_config["properties_to_featurize"].append("per_molecule_total_charge")
    featurize_input_module = FeaturizeInput(featurization_config)

    # only nuclear charges embedded
    assert (
        "nuclear_charge_embedding"
        in featurize_input_module.registered_embedding_operations
    )
    assert len(featurize_input_module.registered_embedding_operations) == 1
    # total charge is added to feature vector
    assert "total_charge" in featurize_input_module.registered_appended_properties
    assert len(featurize_input_module.registered_appended_properties) == 1

    mixing_module = featurize_input_module.mixing
    assert (
        mixing_module.__module__ == "modelforge.potential.utils"
    )  # this is were Dense lives
    mixing_module_name = str(mixing_module)

    assert "Dense" in mixing_module_name

    # make a forward pass, embedd nuclear charges and add total charge (is expanded from per-molecule to per-atom property). Mix the properties then.
    out = featurize_input_module(nnp_input)
    assert out.shape == torch.Size(
        [557, 32]
    )  # nr_of_atoms, nr_of_per_atom_features (the total charge is mixed in)


def test_radial_symmetry_function():

    from modelforge.potential.utils import SchnetRadialBasisFunction, CosineCutoff
    import torch
    from openff.units import unit

    # set cutoff and radial symmetry function
    cutoff = CosineCutoff(cutoff=unit.Quantity(5.0, unit.angstrom))
    rbf_expension = SchnetRadialBasisFunction(
        number_of_radial_basis_functions=18,
        max_distance=unit.Quantity(5.0, unit.angstrom),
    )

    # calculate expension and cutoff
    d_ij = torch.tensor(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    )  # distances have the dimensions [nr_of_pairs, 1] (because displacement vectors have the dimensions [nr_of_pairs, 3])

    f_ij_cutoff = cutoff(d_ij)
    f_ij = rbf_expension(d_ij)
    vs = f_ij * f_ij_cutoff

    # make sure that this matches the output of SchNETRepresentation
    from modelforge.potential.schnet import SchNETRepresentation

    rep = SchNETRepresentation(
        radial_cutoff=5 * unit.angstrom,
        number_of_radial_basis_functions=18,
    )

    representation = rep(d_ij)
    f_ij_cutoff = representation["f_ij"] * representation["f_cutoff"]

    assert torch.allclose(vs, f_ij_cutoff)

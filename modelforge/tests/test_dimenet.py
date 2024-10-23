from typing import Optional
import pytest


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_dimenet_temp")
    return fn


def setup_dimenet_model(potential_seed: Optional[int] = None):
    from modelforge.tests.test_potentials import load_configs_into_pydantic_models
    from modelforge.potential import NeuralNetworkPotentialFactory

    # read default parameters
    config = load_configs_into_pydantic_models("dimenet", "qm9")

    model = NeuralNetworkPotentialFactory.generate_potential(
        use="inference",
        potential_parameter=config["potential"],
        training_parameter=config["training"],
        dataset_parameter=config["dataset"],
        runtime_parameter=config["runtime"],
        potential_seed=potential_seed,
        use_training_mode_neighborlist=True,
        jit=False,
    )
    return model


def test_init():
    """Test initialization of the Dimenet model."""
    potential = setup_dimenet_model()
    assert potential is not None, "Dimenet model should be initialized."


def test_forward(single_batch_with_batchsize, prep_temp_dir):
    import torch

    potential = setup_dimenet_model()
    print(potential)

    batch = single_batch_with_batchsize(
        batch_size=64, dataset_name="QM9", local_cache_dir=str(prep_temp_dir)
    )

    yhat = potential(batch.nnp_input.to_dtype(dtype=torch.float32))


def test_envelope():
    from modelforge.potential.dimenet import Envelope
    import torch

    # Create an instance of the Envelope class
    envelope = Envelope(exponent=5)

    # Sample input tensor
    inputs = torch.tensor([0.5, 0.8, 1.0, 1.2], dtype=torch.float32)

    # Forward pass
    outputs = envelope(inputs)
    assert outputs.shape == inputs.shape
    assert torch.allclose(
        outputs, torch.tensor([1.7109, 0.2539, 0.0000, 0.0000]), rtol=1e-3
    )

    # Script the model for optimization and deployment
    scripted_envelope = torch.jit.script(envelope)

    # Verify that the scripted model works
    outputs_scripted = scripted_envelope(inputs)
    print(outputs_scripted)
    assert torch.allclose(outputs, outputs_scripted)

    # ----------------------------------------- #
    # test for correct output computation

    exponent = (
        5 + 1
    )  # NOTE: Envelop function receives exponent = 5 but takes its increment and uses exponent = 6 FIXME: that seems strange ?
    # start with test for float:
    d_ij = 0.5
    # generate envelope function of d_ij value
    u_05 = (
        1
        - (exponent + 1) * (exponent + 2) / 2 * d_ij**exponent
        + exponent * (exponent + 2) * d_ij ** (exponent + 1)
        - exponent * (exponent + 1) / 2 * d_ij ** (exponent + 2)
    )
    u_05 /= d_ij  # NOTE: this is not in the paper, but in the DimNet++ implementation

    # NOTE: this test passes, but only if you divide by d_ij at the end, which is not in the paper, but in the DimNet++ implementation
    u_05 = torch.tensor([u_05], dtype=torch.float32)

    assert torch.allclose(u_05, outputs[0], rtol=1e-3)


def test_bessel_basis():
    import torch
    from modelforge.potential.dimenet import BesselBasisLayer

    # Create an instance of the BesselBasisLayer
    num_radial = 6
    radial_cutoff = 0.5
    bessel_layer = BesselBasisLayer(
        number_of_radial_bessel_functions=num_radial,
        radial_cutoff=radial_cutoff,
        envelope_exponent=5,
    )

    # Sample input tensor of distances
    num_pairs = 100
    d_ij = torch.linspace(0, radial_cutoff, steps=num_pairs).unsqueeze(
        -1
    )  # Shape: (100,1)

    # Forward pass
    outputs = bessel_layer(d_ij)  # Shape: (100, num_radial)
    shape_tensor = torch.randn(
        num_pairs, num_radial
    )  # output from besser_layer should have this size
    assert shape_tensor.shape == outputs.shape  # Should print: torch.Size([100, 6])


def test_representation():
    from modelforge.potential.dimenet import Representation
    from torch.nn import SiLU

    # Create an instance of the RepresentationBlock
    number_of_radial_bessel_functions = 5
    radial_cutoff = 0.5
    number_of_spherical_harmonics = 7
    envelope_exponent = 5
    activation_function = SiLU()
    embedding_size = 32

    rep = Representation(
        number_of_radial_bessel_functions=number_of_radial_bessel_functions,
        radial_cutoff=radial_cutoff,
        number_of_spherical_harmonics=number_of_spherical_harmonics,
        envelope_exponent=envelope_exponent,
        activation_function=activation_function,
        embedding_size=embedding_size,
    )

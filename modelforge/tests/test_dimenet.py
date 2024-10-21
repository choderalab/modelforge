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

    yhat = potential(batch.nnp_input.to(dtype=torch.float32))


def test_envelope():
    from modelforge.potential.dimenet import Envelope
    import torch

    # Create an instance of the Envelope class
    envelope = Envelope(exponent=5)

    # Sample input tensor
    inputs = torch.tensor([0.5, 0.8, 1.0, 1.2], dtype=torch.float32)

    # Forward pass
    outputs = envelope(inputs)
    print(outputs)

    # Script the model for optimization and deployment
    scripted_envelope = torch.jit.script(envelope)

    # Verify that the scripted model works
    outputs_scripted = scripted_envelope(inputs)
    print(outputs_scripted)


def test_bessel_basis():
    import torch
    from modelforge.potential.dimenet import BesselBasisLayer

    # Create an instance of the BesselBasisLayer
    num_radial = 6
    radial_cutoff = 0.5
    bessel_layer = BesselBasisLayer(
        number_of_radial_bessel_functions=num_radial, radial_cutoff=radial_cutoff, envelope_exponent=5
    )

    # Sample input tensor of distances
    d_ij = torch.linspace(0, radial_cutoff, steps=100).unsqueeze(-1)  # Shape: (100,1)

    # Forward pass
    outputs = bessel_layer(d_ij)  # Shape: (100, num_radial)
    print(outputs.shape)  # Should print: torch.Size([100, 6])

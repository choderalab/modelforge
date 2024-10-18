from typing import Optional
import pytest

@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_dimenet_temp")
    return fn



def setup_dimenet_model(potential_seed: Optional[int] = None):
    from modelforge.tests.test_models import load_configs_into_pydantic_models
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
        jit=False
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

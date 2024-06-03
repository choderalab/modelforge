from modelforge.potential import NeuralNetworkPotentialFactory
import torch
from modelforge.train.training import return_toml_config


def profile_network(model, data):

    result = model(data).E

    forces = -torch.autograd.grad(
        result.sum(), data.positions, create_graph=True, retain_graph=True
    )[0]


def setup(model_name: str):
    config = return_toml_config(
        f"../modelforge/tests/data/training_defaults/{model_name.lower()}_qm9.toml"
    )
    # Extract parameters
    potential_parameters = config["potential"].get("potential_parameters", {})

    model = NeuralNetworkPotentialFactory.create_nnp(
        use="inference",
        model_type=model_name,
        simulation_environment="PyTorch",
        model_parameters=potential_parameters,
    )

    from modelforge.dataset.dataset import DataModule

    # test the self energy calculation on the QM9 dataset
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy

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

    data = next(iter(dataset.train_dataloader())).nnp_input
    return model, data


if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    import torch

    model_name = "SchNet"

    model, data = setup(model_name)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True
        ),  # NOTE: https://github.com/pytorch/pytorch/issues/100253
    ) as prof:
        profile_network(model, data)

    print('######################################')
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
    print('######################################')
    print('######################################')
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    print('######################################')

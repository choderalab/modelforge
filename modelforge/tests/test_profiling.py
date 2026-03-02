import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profiling_function():
    from modelforge.tests.helper_functions import setup_potential_for_test
    import torch
    from modelforge.utils.profiling import (
        start_record_memory_history,
        export_memory_snapshot,
        stop_record_memory_history,
        setup_waterbox_testsystem,
    )

    # define the potential, device and precision
    potential_name = "AimNet2"
    precision = torch.float32
    device = "cuda"

    # setup the input and model
    nnp_input = setup_waterbox_testsystem(2.5, device=device, precision=precision)
    model = setup_potential_for_test(
        potential_name,
        "inference",
        potential_seed=42,
        use_training_mode_neighborlist=True,
        simulation_environment="PyTorch",
    ).to(device, precision)
    # Disable gradients for model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Set model to eval
    model.eval()

    # this is the function that will be profiled
    def loop_to_record():
        for _ in range(5):
            # perform the forward pass through each of the models
            r = model(nnp_input)["per_system_energy"]
            # Compute the gradient (forces) from the predicted energies
            grad = torch.autograd.grad(
                r,
                nnp_input.positions,
                grad_outputs=torch.ones_like(r),
                create_graph=False,
                retain_graph=False,
            )[0]

    # Start recording memory snapshot history
    start_record_memory_history()
    loop_to_record()
    # Create the memory snapshot file
    export_memory_snapshot()
    # Stop recording memory snapshot history
    stop_record_memory_history()

import torch
from loguru import logger as log
import socket
from datetime import datetime
from modelforge.dataset.dataset import NNPInput

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def setup_waterbox_testsystem(
    edge_size_in_nm: float,
    device: torch.device,
    precision: torch.dtype,
) -> NNPInput:
    from modelforge.utils.io import import_

    openmmtools = import_("openmmtools")
    from simtk import unit
    from modelforge.dataset.dataset import NNPInput

    test_system = openmmtools.testsystems.WaterBox(
        box_edge=edge_size_in_nm * unit.nanometer
    )
    positions = test_system.positions  # Positions in nanometers
    topology = test_system.topology

    # Extract atomic numbers and residue indices
    atomic_numbers = []
    residue_indices = []
    for residue_index, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            atomic_numbers.append(atom.element.atomic_number)
            residue_indices.append(residue_index)
    num_waters = len(list(topology.residues()))
    positions_in_nanometers = positions.value_in_unit(unit.nanometer)

    # Convert to torch tensors and move to GPU
    torch_atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
    torch_positions = torch.tensor(
        positions_in_nanometers, dtype=torch.float32, device=device, requires_grad=True
    )
    torch_atomic_subsystem_indices = torch.zeros_like(
        torch_atomic_numbers, dtype=torch.long, device=device
    )
    torch_total_charge = torch.zeros((1, 1), dtype=torch.float32, device=device)

    log.info(f"Waterbox system setup with {num_waters} waters")
    return NNPInput(
        atomic_numbers=torch_atomic_numbers,
        positions=torch_positions,
        atomic_subsystem_indices=torch_atomic_subsystem_indices,
        per_system_total_charge=torch_total_charge,
    ).to_dtype(dtype=precision)


from typing import List
import time


def measure_performance_for_edge_sizes(
    edge_sizes: List[float],
    potential_names: List[str],
):
    """
    Measures GPU memory utilization and computation time for force calculations
    for water boxes of different edge sizes across multiple potentials.
    Parameters
    ----------
    edge_sizes : List[float]
        A list of edge sizes (in nanometers) for the water boxes.
    potential_names : List[str]
        A list of potential names to use in the model setup.
    Returns
    -------
    List[dict]
        A list of dictionaries containing edge size, number of water molecules,
        potential name, memory usage in bytes, and computation time in seconds.
    """
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precicion = torch.float32
    for potential_name in potential_names:
        for edge_size in edge_sizes:

            nnp_input = setup_waterbox_testsystem(
                edge_size,
                device,
                precicion,
            )

            # Import your model setup function
            from modelforge.tests.helper_functions import setup_potential_for_test

            # Setup model
            model = setup_potential_for_test(
                potential_name,
                "inference",
                potential_seed=42,
                use_training_mode_neighborlist=False,
                simulation_environment="PyTorch",
            )

            model.to(device)
            model.to(precicion)
            total_params = sum(p.numel() for p in model.parameters())

            # Measure GPU memory usage and computation time
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.synchronize()

            # Run forward pass and time it
            start_time = time.perf_counter()
            try:
                output = model(nnp_input.as_namedtuple())["per_molecule_energy"]
            except:
                print("Out of memory error during forward pass")
                continue

            try:
                F_training = -torch.autograd.grad(
                    output.sum(),
                    nnp_input.positions,
                    create_graph=False,
                    retain_graph=False,
                )[0]
            except:
                print("Out of memory error during backward pass")
                continue
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            max_memory_allocated = torch.cuda.max_memory_allocated(device=device)
            computation_time = end_time - start_time

            results.append(
                {
                    "potential_name": f"{potential_name}: {total_params:.1e} params",
                    "edge_size_nm": edge_size,
                    "num_waters": num_waters,
                    "memory_usage_bytes": max_memory_allocated,
                    "computation_time_s": computation_time,
                }
            )

            # Clean up
            del (
                nnp_input,
                output,
                model,
            )
            try:
                del F_training
            except:
                pass
            torch.cuda.empty_cache()
            time.sleep(1)  # Sleep for a second to allow GPU memory to be freed

    return results


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_computation_time(results):
    """
    Plots computation time against the number of water molecules for multiple potentials.
    Parameters
    ----------
    results : List[dict]
        A list of dictionaries containing edge size, number of water molecules,
        potential name, memory usage in bytes, and computation time in seconds.
    """
    # Create a DataFrame for plotting
    df = pd.DataFrame(results)
    df["computation_time_ms"] = (
        df["computation_time_s"] * 1000
    )  # Convert seconds to milliseconds

    # Plot using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="num_waters",
        y="computation_time_ms",
        hue="potential_name",
        units="potential_name",
        estimator=None,  # Do not aggregate data
        marker="o",
        linewidth=2,
        markersize=8,
    )
    plt.title("Computation Time vs Number of Water Molecules for Different Potentials")
    plt.xlabel("Number of Water Molecules")
    plt.ylabel("Computation Time (ms)")
    plt.xticks(sorted(df["num_waters"].unique()))
    plt.legend(title="Potential Name")
    plt.tight_layout()
    plt.show()


def plot_gpu_memory_usage(results):
    """
    Plots GPU memory usage against the number of water molecules for multiple potentials.
    Parameters
    ----------
    results : List[dict]
        A list of dictionaries containing edge size, number of water molecules,
        potential name, and memory usage in bytes.
    """
    # Create a DataFrame for plotting
    df = pd.DataFrame(results)
    df["memory_usage_mb"] = df["memory_usage_bytes"] / 1e6  # Convert bytes to megabytes

    # Plot using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="num_waters",
        y="memory_usage_mb",
        units="potential_name",
        estimator=None,  # Do not aggregate data
        hue="potential_name",
        marker="o",
        linewidth=2,
        markersize=8,
    )
    plt.title(
        "Backward pass: GPU Memory Usage vs Number of Water Molecules for Different Potentials"
    )
    plt.xlabel("Number of Water Molecules")
    plt.ylabel("GPU Memory Usage (MB)")
    plt.xticks(sorted(df["num_waters"].unique()))
    plt.legend(title="Potential Name")
    plt.tight_layout()
    plt.show()


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not recording memory history")
        return

    log.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not stopping memory history")
        return

    log.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        log.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        log.error(f"Failed to capture memory snapshot {e}")
        return
    return f"{file_prefix}.pickle"

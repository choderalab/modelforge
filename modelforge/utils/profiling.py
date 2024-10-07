import torch
from loguru import logger as log
import socket
from datetime import datetime, timedelta

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def setup_waterbox_testsystem(
    edge_size_in_nm: float,
    device: torch.device,
    precision: torch.dtype,
) -> None:
    from openmmtools.testsystems import WaterBox
    from simtk import unit
    from modelforge.dataset.dataset import NNPInput

    test_system = WaterBox(box_edge=edge_size_in_nm * unit.nanometer)
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
    torch_total_charge = torch.zeros(num_waters, dtype=torch.float32, device=device)

    return NNPInput(
        atomic_numbers=torch_atomic_numbers,
        positions=torch_positions,
        atomic_subsystem_indices=torch_atomic_subsystem_indices,
        total_charge=torch_total_charge,
    ).to(dtype=precision)


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
        log.info("CUDA unavailable. Not recording memory history")
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

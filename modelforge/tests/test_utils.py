import numpy as np
import torch
import pytest
import platform
import os

ON_MACOS = platform.system() == "Darwin"
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("utils_test")
    return fn


@pytest.mark.parametrize(
    "partial_point_charges, atomic_subsystem_indices, total_charge",
    [
        (
            torch.zeros(6),
            torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64),
            torch.tensor([0.0, 1.0]),
        ),
        (
            torch.zeros(6),
            torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64),
            torch.tensor([-1.0, 2.0]),
        ),
        (
            torch.rand(6),
            torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int64),
            torch.tensor([-1.0, 2.0]),
        ),
    ],
)
def test_default_charge_conservation(
    partial_point_charges: torch.Tensor,
    atomic_subsystem_indices: torch.Tensor,
    total_charge: torch.Tensor,
):
    """
    Test the default_charge_conservation function with various test cases.
    """
    from modelforge.potential.processing import default_charge_conservation

    # test charge equilibration
    # ------------------------- #
    charges = default_charge_conservation(
        partial_point_charges,
        total_charge,
        atomic_subsystem_indices,
    )

    # Calculate the total charge per molecule after correction
    predicted_total_charge = torch.zeros_like(total_charge).scatter_add_(
        0, atomic_subsystem_indices, charges
    )

    # Assert that the predicted total charges match the desired total charges
    assert torch.allclose(predicted_total_charge, total_charge, atol=1e-6)


@pytest.mark.skipif(
    ON_MACOS,
    reason="Skipt Test on MacOS CI runners as it relies on spawning multiple threads. ",
)
def test_method_locking(tmp_path):
    """
    Test the lock_with_attribute decorator to ensure that it correctly serializes access
    to a critical section across multiple processes.
    """
    import multiprocessing
    from modelforge.utils.misc import lock_with_attribute
    import time

    # Define a class with a method decorated by lock_with_attribute
    class TestClass:
        def __init__(self, lock_file):
            self.method_lock = lock_file

        @lock_with_attribute("method_lock")
        def critical_section(self, shared_list):
            process_name = multiprocessing.current_process().name
            # Record entry into the critical section
            shared_list.append(f"{process_name} entered")
            # Simulate work
            time.sleep(1)
            # Record exit from the critical section
            shared_list.append(f"{process_name} exited")

    # Worker function to be executed by each process
    def worker(lock_file, shared_list):
        test_obj = TestClass(lock_file)
        test_obj.critical_section(shared_list)

    # Create a Manager to handle shared state across processes
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    # Path to the lock file within the pytest-provided temporary directory
    lock_file_path = tmp_path / "test.lock"

    # List to hold process objects
    processes = []

    # Create and start multiple processes
    processes = []
    for i in range(4):
        p = multiprocessing.Process(
            target=worker,
            args=(str(lock_file_path), shared_list),
            name=f"Process-{i+1}",
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Verify that only one process was in the critical section at any given time
    entered = set()
    for entry in shared_list:
        if "entered" in entry:
            process = entry.split()[0]
            # Ensure no other process is in the critical section
            assert (
                len(entered) == 0
            ), f"{process} entered while {entered} was in the critical section"
            entered.add(process)
        elif "exited" in entry:
            process = entry.split()[0]
            # Ensure the process that exits was the one that entered
            assert process in entered, f"{process} exited without entering"
            entered.remove(process)

    # Ensure all processes have exited the critical section
    assert len(entered) == 0, f"Processes left in critical section: {entered}"


def test_dense_layer():
    from modelforge.potential.utils import DenseWithCustomDist
    import torch

    # random 2x3 torch.Tensor
    x = torch.randn(2, 3)

    # create a Dense layer with 3 input features and 2 output features
    dense_layer = DenseWithCustomDist(in_features=3, out_features=2)
    out = dense_layer(x)

    # create a Dense layer with 3 input features and 2 output features
    for weight_init_fn, bias_init_fn in zip(
        [torch.nn.init.zeros_, torch.nn.init.xavier_normal_],
        [torch.nn.init.zeros_, torch.nn.init.ones_],
    ):
        # test the weight initialization and correct weight multiplication
        dense_layer = DenseWithCustomDist(
            in_features=3, out_features=2, bias=False, weight_init=weight_init_fn
        )

        # test that weights are handeled correctly
        out = dense_layer(x)
        manuel_out = dense_layer.weight @ x.T
        assert torch.allclose(out, manuel_out.T)
        # test bias
        dense_layer = DenseWithCustomDist(
            in_features=3,
            out_features=2,
            bias=True,
            weight_init=weight_init_fn,
            bias_init=bias_init_fn,
        )
        out = dense_layer(x)
        manuel_out = dense_layer.weight @ x.T + dense_layer.bias
        assert torch.allclose(out, manuel_out.T)


def test_ase_dataclass():

    from modelforge.potential.processing import AtomicSelfEnergies
    from openff.units import unit

    # Example usage
    atomic_self_energies = AtomicSelfEnergies(
        energies={
            "H": 13.6 * unit.kilojoule_per_mole,
            "He": 24.6 * unit.kilojoule_per_mole,
            "Li": 5.4 * unit.kilojoule_per_mole,
        }
    )

    # Access by element name
    assert np.isclose(atomic_self_energies["H"], 13.6)

    # Access by atomic number
    assert np.isclose(atomic_self_energies[1], 13.6)

    # Iterate over the atomic self energies
    for idx, (atom_index, ase) in enumerate(atomic_self_energies):
        print(atom_index, ase)
        assert atom_index == idx + 1


def test_cosine_cutoff():
    """
    Test the cosine cutoff implementation.
    """
    from modelforge.potential import CosineAttenuationFunction

    # Define inputs
    x = torch.Tensor([1, 2, 3])
    y = torch.Tensor([4, 5, 6])
    from openff.units import unit

    # Calculate expected output
    cutoff = 6
    d_ij = torch.linalg.norm(x - y)
    expected_output = 0.5 * (torch.cos(d_ij * np.pi / cutoff) + 1.0)

    # Calculate actual output
    cutoff = 0.6
    cutoff_module = CosineAttenuationFunction(cutoff)
    actual_output = cutoff_module(d_ij / 10)

    # Check if the results are equal NOTE: Cutoff function doesn't care about
    # the units as long as they are the same
    assert np.isclose(actual_output, expected_output)


def test_cosine_cutoff_module():
    # Test CosineAttenuationFunction module
    from modelforge.potential import CosineAttenuationFunction
    from openff.units import unit

    # test the cutoff on this distance vector
    d_ij_angstrom = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
    # the expected outcome is that entry 1 and 2 become zero
    # and entry 0 becomes 0.5 (since the cutoff is 2.0 angstrom)
    # input in angstrom
    cutoff = 2.0

    expected_output = torch.tensor([0.5, 0.0, 0.0]).unsqueeze(1)
    cosine_cutoff_module = CosineAttenuationFunction(cutoff / 10)

    output = cosine_cutoff_module(d_ij_angstrom / 10)  # input is in nanometer

    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_PhysNetAttenuationFunction():
    from modelforge.potential.representation import PhysNetAttenuationFunction
    from openff.units import unit
    import torch

    # test the cutoff on this distance vector (NOTE: it is in angstrom)
    d_ij_angstrom = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
    # the expected outcome is that entry 1 and 2 become zero
    # and entry 0 becomes 0.5 (since the cutoff is 2.0 angstrom)
    # input in angstrom
    cutoff = 2.0 * unit.angstrom

    expected_output = torch.tensor([0.5, 0.0, 0.0]).unsqueeze(1)
    physnet_cutoff_module = PhysNetAttenuationFunction(cutoff.to(unit.nanometer).m)

    output = physnet_cutoff_module(d_ij_angstrom / 10)  # input is in nanometer

    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_scatter_add():
    """
    Test the scatter_add utility function.
    """

    dim_size = 3
    dim = 0
    x = torch.tensor([1, 4, 3, 2], dtype=torch.float32)
    idx_i = torch.tensor([0, 2, 2, 1], dtype=torch.int64)
    # Using PyTorch's native scatter_add function
    shape = list(x.shape)
    shape[dim] = dim_size
    native_result = torch.zeros(shape, dtype=x.dtype)
    native_result.scatter_add_(dim, idx_i, x)


def test_scatter_softmax():
    from modelforge.potential.utils import scatter_softmax

    index = torch.tensor([0, 0, 1, 1, 2])
    src = 1.1 ** torch.arange(10).reshape(2, 5)
    expanded_index = index.expand_as(src)
    util_out = scatter_softmax(src, expanded_index, dim=1, dim_size=3)
    correct_out = torch.tensor(
        [
            [0.4750208259, 0.5249791741, 0.4697868228, 0.5302131772, 1.0000000000],
            [0.4598240554, 0.5401759744, 0.4514355958, 0.5485643744, 1.0000000000],
        ]
    )
    assert torch.allclose(util_out, correct_out)


def test_energy_readout():
    from modelforge.potential.processing import FromAtomToMoleculeReduction
    import torch

    # the EnergyReadout module performs a linear pass to reduce the
    # nr_of_atom_basis to 1 and then performs a scatter add operation to return
    # a tensor with size [nr_of_molecules,]

    # the input for the EnergyReadout module is vector (E_i) that will be scatter_added, and
    # a second tensor supplying the indixes for the summation

    r = {
        "per_atom_energy": torch.tensor([3, 3, 1, 1, 1, 1, 1, 1], dtype=torch.float32),
        "atomic_subsystem_index": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
    }
    energy_readout = FromAtomToMoleculeReduction()
    E = energy_readout(r["atomic_subsystem_index"], r["per_atom_energy"])

    # check that output has length of total number of molecules in batch
    assert E.size() == torch.Size(
        [
            2,
        ]
    )
    # check that the correct values were summed
    assert torch.isclose(E[0], torch.tensor([6.0], dtype=torch.float32))
    assert torch.isclose(E[1], torch.tensor([6.0], dtype=torch.float32))


def test_welford():
    """
    Test the Welford's algorithm implementation.
    """
    from modelforge.utils.misc import Welford
    import torch
    import numpy as np

    torch.manual_seed(0)
    target_mean = 1000
    target_stddev = 50
    target_variance = target_stddev**2

    online_estimator = Welford()

    for i in range(0, 5):
        batch = torch.normal(target_mean, target_stddev, size=(1000,))
        online_estimator.update(batch)

        assert np.isclose(online_estimator.mean / target_mean, 1.0, rtol=1e-1)
        assert np.isclose(online_estimator.variance / target_variance, 1.0, rtol=1e-1)
        assert np.isclose(online_estimator.stddev / target_stddev, 1.0, rtol=1e-1)


@pytest.mark.skipif(
    ON_MACOS and IN_GITHUB_ACTIONS,
    reason="Test is flaky on the MacOS CI runners as it relies on spawning multiple threads. ",
)
def test_filelocking(prep_temp_dir):
    from modelforge.utils.misc import lock_file, unlock_file, check_file_lock

    filepath = str(prep_temp_dir) + "/test.txt"

    import threading

    class thread(threading.Thread):
        def __init__(self, thread_name, thread_id, filepath):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            self.name = thread_name
            self.filepath = filepath
            self.did_I_lock_it = None

        def run(self):
            import time

            with open(self.filepath, "w") as f:
                if not check_file_lock(f):
                    lock_file(f)
                    self.did_I_lock_it = True
                    time.sleep(3)
                    unlock_file(f)

                else:
                    self.did_I_lock_it = False

    # the first thread should lock the file and set "did_I_lock_it" to True
    thread1 = thread("lock_file_here", "Thread-1", filepath)
    # the second thread should check if locked, and set "did_I_lock_it" to False
    # the second thread should also set "status" to True, because it waits
    # for the first thread to unlock the file
    thread2 = thread("encounter_locked_file", "Thread-2", filepath)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # this thread should lock the file, since it will be executed after the others complete
    # this will ensure that we can unlock the file
    thread3 = thread("lock_file_here", "Thread-3", filepath)
    thread3.start()
    thread3.join()
    assert thread1.did_I_lock_it == True

    assert thread2.did_I_lock_it == False
    assert thread3.did_I_lock_it == True

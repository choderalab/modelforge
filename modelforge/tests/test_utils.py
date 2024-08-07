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
    from modelforge.potential.utils import CosineAttenuationFunction

    # Define inputs
    x = torch.Tensor([1, 2, 3])
    y = torch.Tensor([4, 5, 6])
    from openff.units import unit

    cutoff = 6

    # Calculate expected output
    d_ij = torch.linalg.norm(x - y)
    expected_output = 0.5 * (torch.cos(d_ij * np.pi / cutoff) + 1.0)
    cutoff = 0.6 * unit.nanometer

    # Calculate actual output
    cutoff_module = CosineAttenuationFunction(cutoff)
    actual_output = cutoff_module(d_ij / 10)

    # Check if the results are equal
    # NOTE: Cutoff function doesn't care about the units as long as they are the same
    assert np.isclose(actual_output, expected_output)

    # input in angstrom
    cutoff = 2.0 * unit.angstrom
    expected_output = torch.tensor([0.5, 0.0, 0.0])
    cosine_cutoff_module = CosineAttenuationFunction(cutoff)


def test_cosine_cutoff_module():
    # Test CosineCutoff module
    from modelforge.potential.utils import CosineAttenuationFunction
    from openff.units import unit

    # test the cutoff on this distance vector (NOTE: it is in angstrom)
    d_ij_angstrom = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1)
    # the expected outcome is that entry 1 and 2 become zero
    # and entry 0 becomes 0.5 (since the cutoff is 2.0 angstrom)
    # input in angstrom
    cutoff = 2.0 * unit.angstrom

    expected_output = torch.tensor([0.5, 0.0, 0.0]).unsqueeze(1)
    cosine_cutoff_module = CosineAttenuationFunction(cutoff)

    output = cosine_cutoff_module(d_ij_angstrom / 10)  # input is in nanometer

    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_radial_symmetry_function_implementation():
    """
    Test the Radial Symmetry function implementation.
    """
    import torch
    from openff.units import unit
    import numpy as np
    from modelforge.potential.utils import (
        CosineAttenuationFunction,
        GaussianRadialBasisFunctionWithScaling,
    )

    cutoff_module = CosineAttenuationFunction(cutoff=unit.Quantity(5.0, unit.angstrom))

    class RadialSymmetryFunctionTest(GaussianRadialBasisFunctionWithScaling):
        @staticmethod
        def calculate_radial_basis_centers(
            number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            dtype,
        ):
            centers = torch.linspace(
                _min_distance_in_nanometer,
                _max_distance_in_nanometer,
                number_of_radial_basis_functions,
                dtype=dtype,
            )
            return centers

        @staticmethod
        def calculate_radial_scale_factor(
            number_of_radial_basis_functions,
            _max_distance_in_nanometer,
            _min_distance_in_nanometer,
            dtype,
        ):
            scale_factors = torch.full(
                (number_of_radial_basis_functions,),
                (_min_distance_in_nanometer - _max_distance_in_nanometer)
                / number_of_radial_basis_functions,
            )
            scale_factors = (scale_factors * -15_000) ** -0.5
            return scale_factors

    RSF = RadialSymmetryFunctionTest(
        number_of_radial_basis_functions=18,
        max_distance=unit.Quantity(5.0, unit.angstrom),
    )
    # test a single distance
    d_ij = torch.tensor([[0.2]])
    radial_expension = RSF(d_ij)

    expected_output = np.array(
        [
            5.7777413e-08,
            5.4214674e-06,
            2.4740110e-04,
            5.4905377e-03,
            5.9259072e-02,
            3.1104434e-01,
            7.9399312e-01,
            9.8568588e-01,
            5.9509689e-01,
            1.7472850e-01,
            2.4949821e-02,
            1.7326004e-03,
            5.8513560e-05,
            9.6104134e-07,
            7.6763511e-09,
            2.9819147e-11,
            5.6333109e-14,
            5.1755549e-17,
        ],
        dtype=np.float32,
    )

    assert np.allclose(radial_expension.numpy().flatten(), expected_output, rtol=1e-3)

    # test multiple distances with cutoff
    d_ij = torch.tensor([[r] for r in np.linspace(0, 0.5, 10)])
    radial_expension = RSF(d_ij) * cutoff_module(d_ij)

    expected_output = np.array(
        [
            [
                1.00000000e00,
                6.97370611e-01,
                2.36512753e-01,
                3.90097089e-02,
                3.12909145e-03,
                1.22064879e-04,
                2.31574554e-06,
                2.13657562e-08,
                9.58678574e-11,
                2.09196141e-13,
                2.22005077e-16,
                1.14577532e-19,
                2.87583090e-23,
                3.51038337e-27,
                2.08388175e-31,
                6.01615362e-36,
                8.44679753e-41,
                5.76756600e-46,
            ],
            [
                2.68038176e-01,
                7.29490887e-01,
                9.65540222e-01,
                6.21510012e-01,
                1.94559846e-01,
                2.96200218e-02,
                2.19303227e-03,
                7.89645189e-05,
                1.38275834e-06,
                1.17757010e-08,
                4.87703136e-11,
                9.82316969e-14,
                9.62221521e-17,
                4.58380155e-20,
                1.06194951e-23,
                1.19649050e-27,
                6.55604552e-32,
                1.74703654e-36,
            ],
            [
                5.15165267e-03,
                5.47178933e-02,
                2.82643788e-01,
                7.10030194e-01,
                8.67443988e-01,
                5.15386799e-01,
                1.48919812e-01,
                2.09266151e-02,
                1.43012111e-03,
                4.75305832e-05,
                7.68248035e-07,
                6.03888967e-09,
                2.30855409e-11,
                4.29190731e-14,
                3.88050222e-17,
                1.70629005e-20,
                3.64875837e-24,
                3.79458837e-28,
            ],
            [
                7.05512776e-06,
                2.92447055e-04,
                5.89544925e-03,
                5.77981439e-02,
                2.75573882e-01,
                6.38983424e-01,
                7.20556963e-01,
                3.95161266e-01,
                1.05392022e-01,
                1.36699807e-02,
                8.62294776e-04,
                2.64527563e-05,
                3.94651201e-07,
                2.86340809e-09,
                1.01036987e-11,
                1.73382336e-14,
                1.44696036e-17,
                5.87267193e-21,
            ],
            [
                6.79841545e-10,
                1.09978970e-07,
                8.65244557e-06,
                3.31051436e-04,
                6.15997825e-03,
                5.57430086e-02,
                2.45317579e-01,
                5.25042257e-01,
                5.46496226e-01,
                2.76635027e-01,
                6.81011682e-02,
                8.15322217e-03,
                4.74713206e-04,
                1.34419004e-05,
                1.85104660e-07,
                1.23965647e-09,
                4.03750130e-12,
                6.39515861e-15,
            ],
            [
                4.50275565e-15,
                2.84275808e-12,
                8.72828077e-10,
                1.30330158e-07,
                9.46429271e-06,
                3.34240505e-04,
                5.74059467e-03,
                4.79492711e-02,
                1.94775558e-01,
                3.84781601e-01,
                3.69675978e-01,
                1.72725113e-01,
                3.92479574e-02,
                4.33716512e-03,
                2.33089213e-04,
                6.09208166e-06,
                7.74348707e-08,
                4.78668403e-10,
            ],
            [
                1.95755731e-21,
                4.82320349e-18,
                5.77941614e-15,
                3.36790471e-12,
                9.54470642e-10,
                1.31550705e-07,
                8.81760261e-06,
                2.87432047e-04,
                4.55666577e-03,
                3.51307040e-02,
                1.31720486e-01,
                2.40185684e-01,
                2.12994423e-01,
                9.18579329e-02,
                1.92660386e-02,
                1.96514909e-03,
                9.74823310e-05,
                2.35170925e-06,
            ],
            [
                5.02685557e-29,
                4.83367095e-25,
                2.26039866e-21,
                5.14067897e-18,
                5.68568509e-15,
                3.05825078e-12,
                7.99999982e-10,
                1.01773376e-07,
                6.29659424e-06,
                1.89454631e-04,
                2.77224261e-03,
                1.97280685e-02,
                6.82755520e-02,
                1.14914067e-01,
                9.40607618e-02,
                3.74430407e-02,
                7.24871542e-03,
                6.82461743e-04,
            ],
            [
                5.43174696e-38,
                2.03835481e-33,
                3.72003749e-29,
                3.30173621e-25,
                1.42516199e-21,
                2.99167362e-18,
                3.05415190e-15,
                1.51633227e-12,
                3.66121676e-10,
                4.29917177e-08,
                2.45510656e-06,
                6.81841165e-05,
                9.20923191e-04,
                6.04910223e-03,
                1.93234986e-02,
                3.00198084e-02,
                2.26807491e-02,
                8.33362963e-03,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
        ]
    )

    assert np.allclose(radial_expension.numpy(), expected_output, rtol=1e-3)


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


def test_embedding():
    """
    Test the Embedding module.
    """
    from modelforge.potential.utils import Embedding

    maximum_atomic_number = 100
    embedding_dim = 7

    # Create Embedding instance
    embedding = Embedding(maximum_atomic_number, embedding_dim)

    # Test embedding_dim property
    assert embedding.embedding_dim == embedding_dim

    # Test forward pass
    input_tensor = torch.randint(0, 99, (5,))

    output = embedding(input_tensor)
    assert output.shape == (5, embedding_dim)


def test_energy_readout():
    from modelforge.potential.processing import FromAtomToMoleculeReduction
    import torch

    # the EnergyReadout module performs a linear pass to reduce the nr_of_atom_basis to 1
    # and then performs a scatter add operation to return a tensor with size [nr_of_molecules,]

    # the input for the EnergyReadout module is vector (E_i) that will be scatter_added, and
    # a second tensor supplying the indixes for the summation

    r = {
        "per_atom_energy": torch.tensor([3, 3, 1, 1, 1, 1, 1, 1], dtype=torch.float32),
        "atomic_subsystem_index": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
    }
    energy_readout = FromAtomToMoleculeReduction(
        per_atom_property_name="per_atom_energy",
        index_name="atomic_subsystem_index",
        output_name="per_molecule_energy",
    )
    E = energy_readout(r)["per_molecule_energy"]

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
    # the second thread should also set "status" to True, because it waits for the first thread to unlock the file
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

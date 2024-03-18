import numpy as np
import torch
import pytest

from modelforge.potential.utils import CosineCutoff, RadialSymmetryFunction


def test_ase_dataclass():
    from modelforge.potential.utils import AtomicSelfEnergies

    # Example usage
    atomic_self_energies = AtomicSelfEnergies(
        energies={"H": 13.6, "He": 24.6, "Li": 5.4}
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
    cutoff_module = CosineCutoff(cutoff)
    actual_output = cutoff_module(d_ij / 10)

    # Check if the results are equal
    # NOTE: Cutoff function doesn't care about the units as long as they are the same
    assert np.isclose(actual_output, expected_output)


def test_cosine_cutoff_module():
    # Test CosineCutoff module
    from openff.units import unit

    # test the cutoff on this distance vector (NOTE: it is in angstrom)
    d_ij_angstrom = torch.tensor([1.0, 2.0, 3.0])
    # the expected outcome is that entry 1 and 2 become zero
    # and entry 0 becomes 0.5 (since the cutoff is 2.0 angstrom)

    cutoff = unit.Quantity(2.0, unit.angstrom)
    expected_output = torch.tensor([0.5, 0.0, 0.0])
    cosine_cutoff_module = CosineCutoff(cutoff)

    output = cosine_cutoff_module(d_ij_angstrom / 10)  # input is in nanometer

    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_radial_symmetry_function_implementation():
    """
    Test the Radial Symmetry function implementation.
    """
    from modelforge.potential.utils import SchnetRadialSymmetryFunction, CosineCutoff
    import torch
    from openff.units import unit
    import numpy as np

    cutoff_module = CosineCutoff(cutoff=unit.Quantity(5.0, unit.angstrom))
    RSF = SchnetRadialSymmetryFunction(
        number_of_radial_basis_functions=18,
        max_distance=unit.Quantity(5.0, unit.angstrom),
    )
    # test a single distance
    d_ij = torch.tensor([[0.2]])
    radial_expension = RSF(d_ij)

    expected_output = np.array(
        [
            5.5345993e-12,
            6.4639507e-09,
            2.4604744e-06,
            3.0524534e-04,
            1.2342081e-02,
            1.6264367e-01,
            6.9854599e-01,
            9.7782737e-01,
            4.4610667e-01,
            6.6332147e-02,
            3.2145421e-03,
            5.0771905e-05,
            2.6135859e-07,
            4.3849102e-10,
            2.3976884e-13,
            4.2730126e-17,
            2.4819276e-21,
            4.6983385e-26,
        ],
        dtype=np.float32,
    )

    print(radial_expension.numpy().flatten())

    assert np.allclose(radial_expension.numpy().flatten(), expected_output, rtol=1e-3)

    # test multiple distances with cutoff
    d_ij = torch.tensor([[r] for r in np.linspace(0, 0.5, 10)])
    radial_expension = RSF(d_ij) * cutoff_module(d_ij)

    expected_output = np.array(
        [
            [
                1.00000000e00,
                5.70892913e-01,
                1.06223011e-01,
                6.44157068e-03,
                1.27313493e-04,
                8.20098165e-07,
                1.72173816e-09,
                1.17808801e-12,
                2.62722782e-16,
                1.90952293e-20,
                4.52338298e-25,
                3.49229730e-30,
                8.78756240e-36,
                7.20667607e-42,
                1.92624244e-48,
                1.67801662e-55,
                4.76421044e-63,
                4.40854197e-71,
            ],
            [
                1.31254429e-01,
                6.22803495e-01,
                9.63157741e-01,
                4.85459571e-01,
                7.97476857e-02,
                4.26964819e-03,
                7.45033363e-05,
                4.23710002e-07,
                7.85364158e-10,
                4.74439730e-13,
                9.34118334e-17,
                5.99421213e-21,
                1.25363667e-25,
                8.54516847e-31,
                1.89836159e-36,
                1.37450592e-42,
                3.24357284e-49,
                2.49465253e-56,
            ],
            [
                2.96220990e-04,
                1.16824875e-02,
                1.50163411e-01,
                6.29074751e-01,
                8.58913915e-01,
                3.82213690e-01,
                5.54334913e-02,
                2.62027933e-03,
                4.03675429e-05,
                2.02686637e-07,
                3.31686906e-10,
                1.76905324e-13,
                3.07512550e-17,
                1.74218453e-21,
                3.21687974e-26,
                1.93590867e-31,
                3.79703211e-37,
                2.42724179e-43,
            ],
            [
                1.14224879e-08,
                3.74423091e-06,
                4.00012761e-04,
                1.39281827e-02,
                1.58060810e-01,
                5.84606177e-01,
                7.04712145e-01,
                2.76865906e-01,
                3.54516593e-02,
                1.47949056e-03,
                2.01232338e-05,
                8.92057220e-08,
                1.28883410e-10,
                6.06890989e-14,
                9.31394622e-18,
                4.65871485e-22,
                7.59465234e-27,
                4.03514395e-32,
            ],
            [
                7.43164095e-15,
                2.02473905e-11,
                1.79788829e-08,
                5.20314337e-06,
                4.90769884e-04,
                1.50868891e-02,
                1.51157750e-01,
                4.93594719e-01,
                5.25315188e-01,
                1.82212606e-01,
                2.05990104e-02,
                7.58968339e-04,
                9.11402737e-06,
                3.56702673e-08,
                4.55000385e-11,
                1.89158791e-14,
                2.56301214e-18,
                1.13183772e-22,
            ],
            [
                7.96913570e-23,
                1.80458766e-18,
                1.33184612e-14,
                3.20360984e-11,
                2.51150528e-08,
                6.41709524e-06,
                5.34381690e-04,
                1.45035401e-02,
                1.28293849e-01,
                3.69868314e-01,
                3.47534103e-01,
                1.06428293e-01,
                1.06224850e-02,
                3.45544931e-04,
                3.66346706e-06,
                1.26587091e-08,
                1.42559419e-11,
                5.23253063e-15,
            ],
            [
                1.34504798e-32,
                2.53155264e-27,
                1.55290701e-22,
                3.10465609e-18,
                2.02297418e-14,
                4.29612912e-11,
                2.97353307e-08,
                6.70776111e-06,
                4.93164455e-04,
                1.18172354e-02,
                9.22887052e-02,
                2.34904093e-01,
                1.94868652e-01,
                5.26869117e-02,
                4.64272405e-03,
                1.33337517e-04,
                1.24807655e-06,
                3.80749596e-09,
            ],
            [
                3.21547233e-44,
                5.03009727e-38,
                2.56458615e-32,
                4.26155230e-27,
                2.30795335e-22,
                4.07377097e-18,
                2.34355107e-14,
                4.39401565e-11,
                2.68508564e-08,
                5.34767608e-06,
                3.47120387e-04,
                7.34352396e-03,
                5.06335382e-02,
                1.13784054e-01,
                8.33362386e-02,
                1.98927925e-02,
                1.54762942e-03,
                3.92416776e-05,
            ],
            [
                7.75617510e-58,
                1.00846686e-50,
                4.27351237e-44,
                5.90225508e-38,
                2.65680416e-32,
                3.89772973e-27,
                1.86368237e-22,
                2.90429976e-18,
                1.47509621e-14,
                2.44180021e-11,
                1.31736924e-08,
                2.31640384e-06,
                1.32748738e-04,
                2.47945306e-03,
                1.50935274e-02,
                2.99457344e-02,
                1.93637094e-02,
                4.08085825e-03,
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


def test_embedding():
    """
    Test the Embedding module.
    """
    from modelforge.potential.utils import Embedding

    max_Z = 100
    embedding_dim = 7

    # Create Embedding instance
    embedding = Embedding(max_Z, embedding_dim)

    # Test embedding_dim property
    assert embedding.embedding_dim == embedding_dim

    # Test forward pass
    input_tensor = torch.randint(0, 99, (5,))

    output = embedding(input_tensor)
    assert output.shape == (5, embedding_dim)


def test_energy_readout():
    from modelforge.potential.utils import FromAtomToMoleculeReduction
    import torch

    # the EnergyReadout module performs a linear pass to reduce the nr_of_atom_basis to 1
    # and then performs a scatter add operation to return a tensor with size [nr_of_molecules,]

    # the input for the EnergyReadout module is vector (E_i) that will be scatter_added, and
    # a second tensor supplying the indixes for the summation

    E_i = torch.tensor([3, 3, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
    atomic_subsystem_indices = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])

    energy_readout = FromAtomToMoleculeReduction()
    E = energy_readout(E_i, atomic_subsystem_indices)

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

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


@pytest.mark.parametrize("RadialSymmetryFunction", [RadialSymmetryFunction])
def test_rbf(RadialSymmetryFunction):
    """
    Test the Gaussian Radial Basis Function (RBF) implementation.
    """
    from modelforge.dataset import QM9Dataset

    from .helper_functions import prepare_pairlist_for_single_batch, return_single_batch

    batch = return_single_batch(QM9Dataset)
    pairlist = prepare_pairlist_for_single_batch(batch)
    from openff.units import unit

    radial_symmetry_function_module = RadialSymmetryFunction(
        number_of_radial_basis_functions=20,
        radial_cutoff=unit.Quantity(5.0, unit.angstrom),
    )
    output = radial_symmetry_function_module(
        pairlist["d_ij"]
    )  # Shape: [n_pairs, number_of_radial_basis_functions]
    # Add assertion to check the shape of the output
    assert output.shape[2] == 20  # number_of_radial_basis_functions dimension


@pytest.mark.parametrize("RadialSymmetryFunction", [RadialSymmetryFunction])
def test_radial_symmetry_functions(RadialSymmetryFunction):
    # Check dimensions of output and output
    from openff.units import unit

    number_of_radial_basis_functions = 5
    cutoff = unit.Quantity(10.0, unit.angstrom)
    start = unit.Quantity(0.0, unit.angstrom)

    radial_symmetry_function_module = RadialSymmetryFunction(
        number_of_radial_basis_functions=number_of_radial_basis_functions,
        radial_cutoff=cutoff,
        radial_start=start,
    )

    # Test that the number of radial basis functions is correct
    assert (
        radial_symmetry_function_module.number_of_radial_basis_functions
        == number_of_radial_basis_functions
    )

    # Test that the cutoff distance is correct
    assert (
        radial_symmetry_function_module.radial_cutoff.to(unit.nanometer).m
        == cutoff.to(unit.nanometer).m
    )

    # Test that the widths and offsets are correct
    expected_offsets = torch.linspace(
        start.to(unit.nanometer).m,
        cutoff.to(unit.nanometer).m,
        number_of_radial_basis_functions,
    )
    expected_widths = torch.abs(
        expected_offsets[1] - expected_offsets[0]
    ) * torch.ones_like(expected_offsets)
    assert torch.allclose(radial_symmetry_function_module.R_s, expected_offsets)

    # Test that the forward pass returns the expected output
    d_ij = torch.tensor([[1.0], [2.0], [3.0]])
    expected_output = radial_symmetry_function_module(d_ij)
    assert expected_output.shape == (3, 1, number_of_radial_basis_functions)


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

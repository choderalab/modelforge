import numpy as np
import torch

from modelforge.potential.utils import CosineCutoff, cosine_cutoff


def test_cosine_cutoff():
    """
    Test the cosine cutoff implementation.
    """
    # Define inputs
    x = torch.Tensor([1, 2, 3])
    y = torch.Tensor([4, 5, 6])
    cutoff = 2.5

    # Calculate expected output
    d_ij = torch.linalg.norm(x - y)
    expected_output = 0.5 * (np.cos(np.pi * d_ij / cutoff) + 1) if d_ij <= cutoff else 0

    # Calculate actual output
    actual_output = cosine_cutoff(d_ij, cutoff)

    # Check if the results are equal
    assert np.isclose(actual_output, expected_output)
    # Test cosine_cutoff function
    d_ij = torch.tensor([1.0, 2.0, 3.0])
    cutoff = 2.0
    expected_output = torch.tensor([0.5, 0.0, 0.0])
    output = cosine_cutoff(d_ij, cutoff)
    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_cosine_cutoff_module():
    # Test CosineCutoff module
    d_ij = torch.tensor([1.0, 2.0, 3.0])
    cutoff = 2.0
    expected_output = torch.tensor([0.5, 0.0, 0.0])
    cosine_cutoff_module = CosineCutoff(cutoff)
    output = cosine_cutoff_module(d_ij)
    assert torch.allclose(output, expected_output, rtol=1e-3)


def test_rbf():
    """
    Test the Gaussian Radial Basis Function (RBF) implementation.
    """
    from modelforge.dataset import QM9Dataset
    from modelforge.potential import GaussianRBF

    from .helper_functions import prepare_pairlist_for_single_batch, return_single_batch

    batch = return_single_batch(QM9Dataset, mode="fit")
    pairlist = prepare_pairlist_for_single_batch(batch)

    radial_basis = GaussianRBF(n_rbf=20, cutoff=5.0)
    output = radial_basis(pairlist["d_ij"])  # Shape: [n_pairs, n_rbf]
    # Add assertion to check the shape of the output
    assert output.shape[1] == 20  # n_rbf dimension


from modelforge.potential import GaussianRBF


def test_gaussian_rbf():
    n_rbf = 5
    cutoff = 10.0
    start = 0.0
    trainable = False
    gaussian_rbf = GaussianRBF(n_rbf, cutoff, start, trainable)

    # Test that the number of radial basis functions is correct
    assert gaussian_rbf.n_rbf == n_rbf

    # Test that the cutoff distance is correct
    assert gaussian_rbf.cutoff == cutoff

    # Test that the widths and offsets are correct
    expected_offsets = torch.linspace(start, cutoff, n_rbf)
    expected_widths = torch.abs(
        expected_offsets[1] - expected_offsets[0]
    ) * torch.ones_like(expected_offsets)
    assert torch.allclose(gaussian_rbf.offsets, expected_offsets)
    assert torch.allclose(gaussian_rbf.widths, expected_widths)

    # Test that the forward pass returns the expected output
    d_ij = torch.tensor([1.0, 2.0, 3.0])
    expected_output = gaussian_rbf(d_ij)
    assert expected_output.shape == (3, n_rbf)


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


def test_GaussianRBF():
    """
    Test the GaussianRBF layer.
    """

    from modelforge.potential import GaussianRBF

    n_rbf = 10
    dim_of_x = 3
    layer = GaussianRBF(10, 5.0)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = layer(x)  # Shape: [dim_of_x, n_rbf]

    # Add assertion to check the shape of the output
    assert y.shape == (dim_of_x, n_rbf)

def test_scatter_softmax():
    from modelforge.potential.utils import scatter_softmax

    index = torch.tensor([0, 0, 1, 1, 2])
    src = 1.1 ** torch.arange(10).reshape(2, 5)
    util_out = scatter_softmax(src, index, dim=1, dim_size=3)
    correct_out = torch.tensor([[0.4750208259, 0.5249791741, 0.4697868228, 0.5302131772, 1.0000000000],
        [0.4598240554, 0.5401759744, 0.4514355958, 0.5485643744, 1.0000000000]])
    assert(torch.allclose(util_out, correct_out))

def test_sliced_embedding():
    """
    Test the SlicedEmbedding module.
    """
    from modelforge.potential.utils import SlicedEmbedding
    from torch.nn import Embedding

    max_Z = 100
    embedding_dim = 7
    sliced_dim = 0

    # Create SlicedEmbedding instance
    sliced_embedding = SlicedEmbedding(max_Z, embedding_dim, sliced_dim)
    normal_embedding = Embedding(max_Z, embedding_dim)

    # Test embedding_dim property
    assert sliced_embedding.embedding_dim == embedding_dim

    # Test forward pass
    input_tensor = torch.randint(0, 99, (5, 1))

    sliced_output = sliced_embedding(input_tensor)
    normal_output = normal_embedding(input_tensor)

    assert sliced_output.shape == (5, embedding_dim)
    assert normal_output.shape == (5, 1, embedding_dim)
    
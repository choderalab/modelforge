# BEGIN: ed8c6549bwf9
import numpy as np
from modelforge.potential.utils import cosine_cutoff

def test_cosine_cutoff():
    """
    Test the cosine cutoff implementation.
    """
    # Define inputs
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    cutoff = 2.5

    # Calculate expected output
    r = np.linalg.norm(x - y)
    expected_output = 0.5 * (np.cos(np.pi * r / cutoff) + 1) if r <= cutoff else 0

    # Calculate actual output
    actual_output = cosine_cutoff(x, y, cutoff)

    # Check if the results are equal
    assert np.isclose(actual_output, expected_output)
# END: ed8c6549bwf9
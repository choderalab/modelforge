"""
Unit and regression test for the modelforge package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import modelforge


def test_modelforge_imported():
    """Sample test, will always pass so long as import statement worked."""
    print("importing ", modelforge.__name__)
    assert "modelforge" in sys.modules


# Assert that a certain exception is raised
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()

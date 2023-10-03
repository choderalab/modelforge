from modelforge.potential.pain import PaiNN

from .helper_functinos import generate_methane_input

from modelforge.potential.utils import CosineCutoff


def test_PaiNN_init():
    """Test initialization of the PaiNN neural network potential."""
    painn = PaiNN(128, 6, 10, cutoff_fn=CosineCutoff(5.0))
    assert painn is not None, "PaiNN model should be initialized."

from modelforge.potential.pain import PaiNN

from .helper_functinos import generate_methane_input

from modelforge.potential.utils import CosineCutoff

from torch.nn import functional as F


def test_PaiNN_init():
    """Test initialization of the PaiNN neural network potential."""
    painn = PaiNN(128, 6, 10, CosineCutoff(5.0), F.silu)
    assert painn is not None, "PaiNN model should be initialized."

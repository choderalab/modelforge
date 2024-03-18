import pyro
from modelforge.potential import SchNet
from modelforge.potential.bayesian_models import BayesianAutoNormalPotential
from .helper_functions import SIMPLIFIED_INPUT_DATA


def test_bayesian_model():
    # initialize a vanilla SchNet model
    schnet = SchNet()

    # make a Bayesian model from the SchNet
    schnet = BayesianAutoNormalPotential(schnet, log_sigma=1e-2)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(schnet)
    assert guide is not None

    # run SVI using the Bayesian model
    svi = pyro.infer.SVI(
        model=schnet,
        guide=guide,
        optim=pyro.optim.Adam({"lr": 1e-3}),
        loss=pyro.infer.Trace_ELBO(),
    )

    # calculate VI loss
    svi.step(SIMPLIFIED_INPUT_DATA, y=0.0)

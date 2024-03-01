import pyro
from modelforge.potential import CosineCutoff, GaussianRBF
from modelforge.potential.utils import SlicedEmbedding
from modelforge.potential.schnet import SchNET
from modelforge.potential.bayesian_models import BayesianAutoNormalPotential
from .helper_functions import SIMPLIFIED_INPUT_DATA

def test_bayesian_model():
    # initialize a vanilla SchNet model
    embedding = SlicedEmbedding(8, 16, sliced_dim=0)
    rbf = GaussianRBF(n_rbf=8, cutoff=5.0)
    cutoff = CosineCutoff(5.0)
    schnet = SchNET(
        embedding=embedding,
        cutoff=cutoff,
        nr_interaction_blocks=8,
        radial_basis=rbf,
    )

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





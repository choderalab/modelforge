import torch
import pyro
from pyro.nn.module import to_pyro_module_

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def init_log_sigma(model, value):
    """Initializes the log_sigma parameters of a model

    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize

    value : float
        The value to initialize the log_sigma parameters to

    """
    params = {
        name: pyro.nn.PyroSample(
            pyro.distributions.Normal(
                torch.zeros(param.shape),
                torch.ones(param.shape) * value,
            )
        )
        for name, param in model.named_parameters()
    }

    for name, param in model.named_parameters():
        rsetattr(model, name, params[name])


class BayesianAutoNormalPotential(pyro.nn.PyroModule):
    """A Bayesian model with a normal prior and likelihood.

    Parameters
    ----------
    log_sigma : float, optional
        The initial value of the log_sigma parameters. Default is 0.0.

    Methods
    -------
    model
        The model function. If no `y` argument is provided, 
        provide the prior; if `y` is provided, provide the likelihood.
    """
    def __init__(
            self, base_model,
            *args, **kwargs,
    ):
        super().__init__()
        to_pyro_module_(base_model)
        self.base_model = base_model
        log_sigma = kwargs.pop("log_sigma", 0.0)
        init_log_sigma(self.base_model, log_sigma)

    def forward(self, *args, **kwargs):
        """The model function. If no `y` argument is provided, 
        provide the prior; if `y` is provided, provide the likelihood.
        """
        y = kwargs.pop("y", None)
        y_hat = self.base_model(*args, **kwargs)["E_predict"]
        pyro.sample(
            "obs", 
            pyro.distributions.Delta(y_hat),
            obs=y
        )

    

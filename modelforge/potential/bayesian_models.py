import torch
import pyro
from pyro.nn.module import to_pyro_module_

def init_log_sigma(model, value):
    """Initializes the log_sigma parameters of a model

    Parameters
    ----------
    model : torch.nn.Module
        The model to initialize

    value : float
        The value to initialize the log_sigma parameters to

    """
    log_sigma_params = {
        name + "_log_sigma": pyro.nn.PyroParam(
            torch.ones(param.shape) * value,
        )
        for name, param in model.named_parameters()
    }

    for name, param in log_sigma_params.items():
        setattr(model, name, param)

class BayesianAutoNormalPotential(torch.nn.Module):
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
            self,
            *args, **kwargs,
    ):
        super().__init__()
        log_sigma = kwargs.pop("log_sigma", 0.0)
        init_log_sigma(self, log_sigma)

    def model(self, *args, **kwargs):
        """The model function. If no `y` argument is provided, 
        provide the prior; if `y` is provided, provide the likelihood.
        """
        y = kwargs.pop("y", None)
        y_hat = self(*args, **kwargs)
        pyro.sample(
            "obs", 
            pyro.distributions.Delta(y_hat),
            obs=y
        )

    

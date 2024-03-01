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
    def __init__(
            self,
            *args, **kwargs,
    ):
        super().__init__()
        log_sigma = kwargs.pop("log_sigma", 0.0)
        init_log_sigma(self, log_sigma)

    def model(self, *args, **kwargs):
        y = kwargs.pop("y", None)
        y_hat = self(*args, **kwargs)
        pyro.sample(
            "obs", 
            pyro.distributions.Delta(y_hat),
            obs=y
        )

    

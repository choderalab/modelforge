from modelforge.potential import AllModelNames, AllModelClasses
from typing import Union, Tuple, Callable, NamedTuple
import numpy as np


class JAXModel:
    """A model wrapper that facilitates calling a JAX function with predefined parameters and buffers.

    Attributes
    ----------
    jax_fn : Callable
        The JAX function to be called.
    parameter : jax.
        Parameters required by the JAX function.
    buffer : Any
        Buffers required by the JAX function.
    name : str
        Name of the model.
    """

    def __init__(
        self, jax_fn: Callable, parameter: np.ndarray, buffer: np.ndarray, name: str
    ):
        self.jax_fn = jax_fn
        self.parameter = parameter
        self.buffer = buffer
        self.name = name

    def __call__(self, data: NamedTuple):
        """Calls the JAX function using the stored parameters and buffers along with additional data.

        Parameters
        ----------
        data : NamedTuple
            Data to be passed to the JAX function.

        Returns
        -------
        Any
            The result of the JAX function.
        """

        return self.jax_fn(self.parameter, self.buffer, data)

    def __repr__(self):
        return f"{self.__class__.__name__} wrapping {self.name}"


class PyTorch2JAXConverter:

    def convert_to_jax_model(self, nnp_instance: AllModelClasses) -> JAXModel:
        """Converts a PyTorch neural network potential instance to a JAXModel.

        Parameters
        ----------
        nnp_instance : Any
            The neural network potential instance to convert.

        Returns
        -------
        JAXModel
            The converted JAX model.
        """

        jax_fn, params, buffers = self._convert_pytnn_to_jax(nnp_instance)
        return JAXModel(jax_fn, params, buffers, nnp_instance.__class__.__name__)

    @staticmethod
    def _convert_pytnn_to_jax(
        nnp_instance: AllModelClasses,
    ) -> Tuple[Callable, np.ndarray, np.ndarray]:
        """Internal method to convert PyTorch neural network parameters and buffers to JAX format.

        Parameters
        ----------
        nnp_instance : Any
            The PyTorch neural network instance.

        Returns
        -------
        Tuple[Callable, Any, Any]
            A tuple containing the JAX function, parameters, and buffers.
        """

        import jax
        from jax import custom_vjp
        from pytorch2jax.pytorch2jax import convert_to_jax, convert_to_pyt
        import functorch
        from functorch import make_functional_with_buffers

        # skip input checks
        nnp_instance.mode = "fast"

        # Convert the PyTorch model to a functional representation and extract the model function and parameters
        model_fn, model_params, model_buffer = make_functional_with_buffers(
            nnp_instance
        )

        # Convert the model parameters from PyTorch to JAX representations
        model_params = jax.tree_map(convert_to_jax, model_params)
        # Convert the model buffer from PyTorch to JAX representations
        model_buffer = jax.tree_map(convert_to_jax, model_buffer)

        # Define the apply function using a custom VJP
        @custom_vjp
        def apply(params, *args, **kwargs):
            # Convert the input data from PyTorch to JAX representations
            params, args, kwargs = map(
                lambda x: jax.tree_map(convert_to_pyt, x), (params, args, kwargs)
            )
            # Apply the model function to the input data
            out = model_fn(params, *args, **kwargs)
            # Convert the output data from JAX to PyTorch representations
            out = jax.tree_map(convert_to_jax, out)
            return out

        # Define the forward and backward passes for the VJP
        def apply_fwd(params, *args, **kwargs):
            return apply(params, *args, **kwargs), (params, args, kwargs)

        def apply_bwd(res, grads):
            params, args, kwargs = res
            params, args, kwargs = map(
                lambda x: jax.tree_map(convert_to_pyt, x), (params, args, kwargs)
            )
            grads = jax.tree_map(convert_to_pyt, grads)
            # Compute the gradients using the model function and convert them from JAX to PyTorch representations
            grads = functorch.vjp(model_fn, params, *args, **kwargs)[1](grads)
            return jax.tree_map(convert_to_jax, grads)

        apply.defvjp(apply_fwd, apply_bwd)

        # Return the apply function and the converted model parameters
        return apply, model_params, model_buffer


def convert_pytorch_models_to_flex(model: AllModelClasses):

    import ivy
    import torch
    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule
    from loguru import logger as log

    ivy.set_backend("jax")

    # generate trace input
    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(data, batch_size=32)

    dataset.prepare_data(remove_self_energies=True, normalize=False)
    batch = dataset.train_dataloader().__iter__().__next__()

    torch_result = model(batch.nnp_input)
    # Transpile it into a hk.Module with the corresponding parameters
    jax_model = ivy.transpile(model, source="torch", to="jax", args=[batch.nnp_input])

    import jax.numpy as jnp
    from jax import random

    jax_input = batch.nnp_input.as_jax_namedtuple()
    variables = jax_model.init(random.key(0), jax_input)

    jax_result = jax_model.apply(variables, jax_input)
    if jnp.allclose(torch_result.E.detach().numpy(), jax_result.E):
        log.info("Model transposing was sucessfull")
    else:
        raise RuntimeError(
            "Model transposing was not successful. The results from the PyTorch and JAX models do not match."
        )

    return jax_model

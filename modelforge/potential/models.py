import torch.nn as nn
import torch
from typing import Dict, List, Optional


class Transform(nn.Module):
    pass


class PotentialBase(nn.Module):
    """
    Base class for all atomistic models.

    Parameters
    ----------
    input_dtype_str : str, optional
        Data type of real inputs, default is "float32".

    Attributes
    ----------
    required_derivatives : Optional[List[str]]
        List of required derivatives.
    model_outputs : Optional[List[str]]
        List of model outputs.
    """

    def __init__(
        self,
        input_dtype_str: str = "float32",
    ):
        super().__init__()
        self.input_dtype_str = input_dtype_str
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        """
        Collect required derivatives from submodules.

        Returns
        -------
        List[str]
            List of required derivatives.
        """
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        """
        Collect model outputs from submodules.

        Returns
        -------
        List[str]
            List of model outputs.
        """
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize derivatives for response properties.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Dictionary of input tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of input tensors with required gradients enabled.
        """

        for p in self.required_derivatives:
            if p in inputs.keys():
                inputs[p].requires_grad_()  # set in place
        return inputs

    def extract_outputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        results = {k: inputs[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(PotentialBase):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module, and a list of output modules.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        input_dtype_str: str = "float32",
    ):
        """
        Initialize the NeuralNetworkPotential class.

        Parameters
        ----------
        representation : nn.Module
            The module that builds representation from inputs.
        input_modules : List[nn.Module], optional
            Modules applied before representation, default is None.
        output_modules : List[nn.Module], optional
            Modules that predict output properties from the representation, default is None.
        input_dtype_str : str, optional
            The dtype of real inputs, default is "float32".
        """
        super().__init__(
            input_dtype_str=input_dtype_str,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules or [])
        self.output_modules = nn.ModuleList(output_modules or [])

        self.collect_derivatives()
        self.collect_outputs()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the neural network potential.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            Dictionary of input tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of output tensors.
        """
        inputs = self.initialize_derivatives(inputs)

        for m in self.input_modules:
            inputs = m(inputs)

        inputs = self.representation(inputs)

        for m in self.output_modules:
            inputs = m(inputs)

        results = self.extract_outputs(inputs)

        return results

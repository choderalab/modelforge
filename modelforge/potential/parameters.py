"""
This module contains pydantic models for storing the parameters of the potentials.
"""

from enum import Enum
from typing import List, Optional, Type, Union

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from modelforge.utils.units import _convert_str_or_unit_to_unit_length


class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


# To avoid having to set config parameters for each class,
# we will just create a parent class for all the parameters classes.
class ParametersBase(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )


# for the activation functions we have defined alpha and negative slope are the
# two parameters that are possible
class ActivationFunctionParamsAlpha(BaseModel):
    alpha: Optional[float] = None


class ActivationFunctionParamsNegativeSlope(BaseModel):
    negative_slope: Optional[float] = None


class AtomicNumber(BaseModel):
    maximum_atomic_number: int = 101
    number_of_per_atom_features: int = 32


class Featurization(BaseModel):
    properties_to_featurize: List[str]
    atomic_number: AtomicNumber = Field(default_factory=AtomicNumber)


class ActivationFunctionName(CaseInsensitiveEnum):
    ReLU = "ReLU"
    CeLU = "CeLU"
    Sigmoid = "Sigmoid"
    Softmax = "Softmax"
    ShiftedSoftplus = "ShiftedSoftplus"
    SiLU = "SiLU"
    Tanh = "Tanh"
    LeakyReLU = "LeakyReLU"
    ELU = "ELU"


# this enum will tell us if we need to pass additional parameters to the activation function
class ActivationFunctionParamsEnum(CaseInsensitiveEnum):
    ReLU = "None"
    CeLU = ActivationFunctionParamsAlpha
    Sigmoid = "None"
    Softmax = "None"
    ShiftedSoftplus = "None"
    SiLU = "None"
    Tanh = "None"
    LeakyReLU = ActivationFunctionParamsNegativeSlope
    ELU = ActivationFunctionParamsAlpha


class CoreParameterBase(ParametersBase):
    # Ensure that both lists (properties and sizes) have the same length
    @model_validator(mode="after")
    def validate_predicted_properties(self):
        if len(self.predicted_properties) != len(self.predicted_dim):
            raise ValueError(
                "The length of 'predicted_properties' and 'predicted_dim' must be the same."
            )
        return self


class ActivationFunctionConfig(ParametersBase):

    activation_function_name: ActivationFunctionName
    activation_function_arguments: Optional[
        Union[ActivationFunctionParamsAlpha, ActivationFunctionParamsNegativeSlope]
    ] = None

    @model_validator(mode="after")
    def validate_activation_function_arguments(self) -> "ActivationFunctionConfig":
        if ActivationFunctionParamsEnum[self.activation_function_name].value != "None":
            if self.activation_function_arguments is None:
                raise ValueError(
                    f"Activation function {self.activation_function_name} requires additional arguments."
                )
        else:
            if self.activation_function_arguments is not None:
                raise ValueError(
                    f"Activation function {self.activation_function_name} does not require additional arguments."
                )
        return self

    def return_activation_function(self):
        from modelforge.potential.utils import ACTIVATION_FUNCTIONS

        if self.activation_function_arguments is not None:
            return ACTIVATION_FUNCTIONS[self.activation_function_name](
                **self.activation_function_arguments.model_dump(exclude_unset=True)
            )
        return ACTIVATION_FUNCTIONS[self.activation_function_name]()

    @computed_field
    @property
    def activation_function(self) -> Type[torch.nn.Module]:
        return self.return_activation_function()


# these will all be set by default to false such that we do not need to define
# unused post processing operations in the datafile
class GeneralPostProcessingOperation(ParametersBase):
    calculate_molecular_self_energy: bool = False
    calculate_atomic_self_energy: bool = False


class PerAtomEnergy(ParametersBase):
    normalize: bool = False
    from_atom_to_system_reduction: bool = False
    keep_per_atom_property: bool = False


class PerAtomCharge(ParametersBase):
    conserve: bool = True
    conserve_strategy: str = "default"


class ElectrostaticPotential(ParametersBase):
    electrostatic_strategy: str = "coulomb"
    maximum_interaction_radius: float = 0.5

    converted_units = field_validator(
        "maximum_interaction_radius",
        mode="before",
    )(_convert_str_or_unit_to_unit_length)


class PostProcessingParameter(ParametersBase):
    properties_to_process: List[str]
    per_atom_energy: PerAtomEnergy = PerAtomEnergy()
    per_atom_charge: PerAtomCharge = PerAtomCharge()
    electrostatic_potential: ElectrostaticPotential = ElectrostaticPotential()
    general_postprocessing_operation: GeneralPostProcessingOperation = (
        GeneralPostProcessingOperation()
    )


class AimNet2Parameters(ParametersBase):
    class CoreParameter(CoreParameterBase):
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        activation_function_parameter: ActivationFunctionConfig
        featurization: Featurization
        predicted_properties: List[str]
        predicted_dim: List[int]
        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_or_unit_to_unit_length
        )

    potential_name: str = "AimNet2"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class ANI2xParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):
        angle_sections: int
        maximum_interaction_radius: float
        minimum_interaction_radius: float
        number_of_radial_basis_functions: int
        maximum_interaction_radius_for_angular_features: float
        minimum_interaction_radius_for_angular_features: float
        angular_dist_divisions: int
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator(
            "maximum_interaction_radius",
            "minimum_interaction_radius",
            "maximum_interaction_radius_for_angular_features",
            "minimum_interaction_radius_for_angular_features",
            mode="before",
        )(_convert_str_or_unit_to_unit_length)

    potential_name: str = "ANI2x"
    only_unique_pairs: bool = True
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class SchNetParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        number_of_filters: int
        shared_interactions: bool
        activation_function_parameter: ActivationFunctionConfig
        featurization: Featurization
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_or_unit_to_unit_length
        )

    potential_name: str = "SchNet"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: int = -1


class TensorNetParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):
        number_of_per_atom_features: int
        number_of_interaction_layers: int
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        minimum_interaction_radius: float
        maximum_atomic_number: int
        equivariance_invariance_group: str
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator(
            "maximum_interaction_radius", "minimum_interaction_radius", mode="before"
        )(_convert_str_or_unit_to_unit_length)

    potential_name: str = "TensorNet"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class PaiNNParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        shared_interactions: bool
        shared_filters: bool
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_or_unit_to_unit_length
        )

    potential_name: str = "PaiNN"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class PhysNetParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_residual: int
        number_of_modules: int
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_or_unit_to_unit_length
        )

    potential_name: str = "PhysNet"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class SAKEParameters(ParametersBase):
    class CoreParameter(CoreParameterBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        number_of_spatial_attention_heads: int
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[str]
        predicted_dim: List[int]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_or_unit_to_unit_length
        )

    potential_name: str = "SAKE"
    only_unique_pairs: bool = False
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None

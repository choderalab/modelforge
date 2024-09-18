"""
This module contains pydantic models for storing the parameters of the potentials.
"""

from enum import Enum
from typing import List, Optional, Type, Union

import torch
from openff.units import unit
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from modelforge.utils.units import _convert_str_to_unit_length


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
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
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


class OutputTypeEnum(CaseInsensitiveEnum):
    scalar = "scalar"
    vector = "vector"


class PredictedPropertiesParameter(ParametersBase):
    name: str
    type: OutputTypeEnum


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
    from_atom_to_molecule_reduction: bool = False
    keep_per_atom_property: bool = False


class CoulombPotential(ParametersBase):
    electrostatic_strategy: str = "default"
    maximum_interaction_radius: float

    converted_units = field_validator(
        "maximum_interaction_radius",
        mode="before",
    )(_convert_str_to_unit_length)


class PostProcessingParameter(ParametersBase):
    properties_to_process: List[str]
    per_atom_energy: PerAtomEnergy = PerAtomEnergy()
    coulomb_potential: Optional[CoulombPotential] = None
    general_postprocessing_operation: GeneralPostProcessingOperation = (
        GeneralPostProcessingOperation()
    )


class AimNet2Parameters(ParametersBase):
    class CoreParameter(ParametersBase):
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        activation_function_parameter: ActivationFunctionConfig
        featurization: Featurization

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_to_unit_length
        )

    potential_name: str = "AimNet2"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class ANI2xParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        angle_sections: int
        maximum_interaction_radius: float
        minimum_interaction_radius: float
        number_of_radial_basis_functions: int
        maximum_interaction_radius_for_angular_features: float
        minimum_interaction_radius_for_angular_features: float
        angular_dist_divisions: int
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator(
            "maximum_interaction_radius",
            "minimum_interaction_radius",
            "maximum_interaction_radius_for_angular_features",
            "minimum_interaction_radius_for_angular_features",
            mode="before",
        )(_convert_str_to_unit_length)

    potential_name: str = "ANI2x"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class SchNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        number_of_filters: int
        shared_interactions: bool
        activation_function_parameter: ActivationFunctionConfig
        featurization: Featurization
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_to_unit_length
        )

    potential_name: str = "SchNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: int = -1


class TensorNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        number_of_per_atom_features: int
        number_of_interaction_layers: int
        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        minimum_interaction_radius: float
        maximum_atomic_number: int
        equivariance_invariance_group: str
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator(
            "maximum_interaction_radius", "minimum_interaction_radius", mode="before"
        )(_convert_str_to_unit_length)

    potential_name: str = "TensorNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class PaiNNParameters(ParametersBase):
    class CoreParameter(ParametersBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        shared_interactions: bool
        shared_filters: bool
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_to_unit_length
        )

    potential_name: str = "PaiNN"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class PhysNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_residual: int
        number_of_modules: int
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_to_unit_length
        )

    potential_name: str = "PhysNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None


class SAKEParameters(ParametersBase):
    class CoreParameter(ParametersBase):

        number_of_radial_basis_functions: int
        maximum_interaction_radius: float
        number_of_interaction_modules: int
        number_of_spatial_attention_heads: int
        featurization: Featurization
        activation_function_parameter: ActivationFunctionConfig
        predicted_properties: List[PredictedPropertiesParameter]

        converted_units = field_validator("maximum_interaction_radius", mode="before")(
            _convert_str_to_unit_length
        )

    potential_name: str = "SAKE"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter
    potential_seed: Optional[int] = None

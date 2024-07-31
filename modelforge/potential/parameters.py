from pydantic import BaseModel, field_validator, ConfigDict
from openff.units import unit
from typing import Union, List
from modelforge.utils.units import _convert_str_to_unit

"""
This module contains pydantic models for storing the parameters of 
"""


# To avoid having to set config parameters for each class,
# we will just create a parent class for all the parameters classes.
class ParametersBase(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
    )


# these will all be set by default to false
# such that we do not need to define unused post processing operations in the datafile
class GeneralPostProcessingOperation(ParametersBase):
    calculate_molecular_self_energy: bool = False
    calculate_atomic_self_energy: bool = False


class PerAtomEnergy(ParametersBase):
    normalize: bool = False
    from_atom_to_molecule_reduction: bool = False
    keep_per_atom_property: bool = False


class ANI2xParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        angle_sections: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        minimum_interaction_radius: Union[str, unit.Quantity]
        number_of_radial_basis_functions: int
        maximum_interaction_radius_for_angular_features: Union[str, unit.Quantity]
        minimum_interaction_radius_for_angular_features: Union[str, unit.Quantity]
        angular_dist_divisions: int
        activation_function: str

        converted_units = field_validator(
            "maximum_interaction_radius",
            "minimum_interaction_radius",
            "maximum_interaction_radius_for_angular_features",
            "minimum_interaction_radius_for_angular_features",
        )(_convert_str_to_unit)

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "ANI2x"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class SchNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        class Featurization(ParametersBase):
            properties_to_featurize: List[str]
            highest_atomic_number: int
            number_of_per_atom_features: int

        number_of_radial_basis_functions: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        number_of_filters: int
        shared_interactions: bool
        activation_function: str
        featurization: Featurization

        converted_units = field_validator("maximum_interaction_radius")(
            _convert_str_to_unit
        )

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "SchNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class TensorNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        # class Featurization(ParametersBase):
        #     properties_to_featurize: List[str]
        #     max_Z: int
        #     number_of_per_atom_features: int

        number_of_per_atom_features: int
        number_of_interaction_layers: int
        number_of_radial_basis_functions: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        minimum_interaction_radius: Union[str, unit.Quantity]
        highest_atomic_number: int
        equivariance_invariance_group: str

        converted_units = field_validator("maximum_interaction_radius")(_convert_str_to_unit)
        converted_units = field_validator("minimum_interaction_radius")(_convert_str_to_unit)

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "TensorNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class PaiNNParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        class Featurization(ParametersBase):
            properties_to_featurize: List[str]
            highest_atomic_number: int
            number_of_per_atom_features: int

        number_of_radial_basis_functions: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        shared_interactions: bool
        shared_filters: bool
        featurization: Featurization
        activation_function: str

        converted_units = field_validator("maximum_interaction_radius")(
            _convert_str_to_unit
        )

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "PaiNN"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class PhysNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        class Featurization(ParametersBase):
            properties_to_featurize: List[str]
            highest_atomic_number: int
            number_of_per_atom_features: int

        number_of_radial_basis_functions: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        number_of_interaction_residual: int
        number_of_modules: int
        featurization: Featurization
        activation_function: str

        converted_units = field_validator("maximum_interaction_radius")(
            _convert_str_to_unit
        )

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "PhysNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class SAKEParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        class Featurization(ParametersBase):
            properties_to_featurize: List[str]
            highest_atomic_number: int
            number_of_per_atom_features: int

        number_of_radial_basis_functions: int
        maximum_interaction_radius: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        number_of_spatial_attention_heads: int
        featurization: Featurization
        activation_function: str

        converted_units = field_validator("maximum_interaction_radius")(
            _convert_str_to_unit
        )

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "SAKE"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter

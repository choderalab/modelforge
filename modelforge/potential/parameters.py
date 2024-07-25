from pydantic import BaseModel, field_validator, ConfigDict
from openff.units import unit
from typing import Union

"""
This module contains pydantic models for storing the parameters of 
"""


# To avoid having to set use_enum_values = True in every subclass of BaseModel,
# we will just create a parent class for all the parameters classes.
class ParametersBase(BaseModel):
    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)


# define a reusable validator for converting string to unit.Quantity
def convert_str_to_unit(value: Union[str, unit.Quantity]) -> unit.Quantity:
    if isinstance(value, str):
        return unit.Quantity(value)
    return value


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
        radial_max_distance: Union[str, unit.Quantity]
        radial_min_distance: Union[str, unit.Quantity]
        number_of_radial_basis_functions: int
        angular_max_distance: Union[str, unit.Quantity]
        angular_min_distance: Union[str, unit.Quantity]
        angular_dist_divisions: int

        converted_units = field_validator(
            "radial_max_distance",
            "radial_min_distance",
            "angular_max_distance",
            "angular_min_distance",
        )(convert_str_to_unit)
        # def convert_to_unit(cls, value) -> unit.Quantity:
        #
        #     if isinstance(value, str):
        #         return unit.Quantity(value)
        #     return value

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
        max_Z: int
        number_of_atom_features: int
        number_of_radial_basis_functions: int
        cutoff: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        number_of_filters: int
        shared_interactions: bool

        converted_units = field_validator("cutoff")(convert_str_to_unit)

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "SchNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class PaiNNParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        max_Z: int
        number_of_atom_features: int
        number_of_radial_basis_functions: int
        cutoff: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        shared_interactions: bool
        shared_filters: bool

        converted_units = field_validator("cutoff")(convert_str_to_unit)

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
        max_Z: int
        number_of_atom_features: int
        number_of_radial_basis_functions: int
        cutoff: Union[str, unit.Quantity]
        number_of_interaction_residual: int
        number_of_modules: int

        converted_units = field_validator("cutoff")(convert_str_to_unit)

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
        max_Z: int
        number_of_atom_features: int
        number_of_radial_basis_functions: int
        cutoff: Union[str, unit.Quantity]
        number_of_interaction_modules: int
        number_of_spatial_attention_heads: int

        converted_units = field_validator("cutoff")(convert_str_to_unit)

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy()
        general_postprocessing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation()
        )

    potential_name: str = "SAKE"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter

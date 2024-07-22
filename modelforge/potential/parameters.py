from pydantic import BaseModel


# To avoid having to set use_enum_values = True in every subclass of BaseModel,
# we will just create a parent class for all the parameters classes.
class ParametersBase(BaseModel):
    class Config:
        use_enum_values = True


class GeneralPostProcessingOperation(ParametersBase):
    calculate_molecular_self_energy: bool = False
    calculate_atomic_self_energy: bool = False


class PerAtomEnergy(ParametersBase):
    normalize: bool = False
    from_atom_to_molecule_reduction: bool = False
    keep_per_atom_property: bool = False


class ANI2xParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        angle_sections: int = 4
        radial_max_distance: str = "5.1 angstrom"
        radial_min_distance: str = "0.8 angstrom"
        number_of_radial_basis_functions: int = 16
        angular_max_distance: str = "3.5 angstrom"
        angular_min_distance: str = "0.8 angstrom"
        angular_dist_divisions: int = 8

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy(
            normalize=True,
            from_atom_to_molecule_reduction=True,
            keep_per_atom_property=True,
        )
        general_post_processing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation(
                calculate_molecular_self_energy=False,
                calculate_atomic_self_energy=False,
            )
        )

    model_name: str = "ANI2x"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class SchNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        max_Z: int = 101
        number_of_atom_features: int = 32
        number_of_radial_basis_functions: int = 20
        cutoff: str = "5.0 angstrom"
        number_of_interaction_modules: int = 3
        number_of_filters: int = 32
        shared_interactions: bool = False

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy(
            normalize=True,
            from_atom_to_molecule_reduction=True,
            keep_per_atom_property=True,
        )
        general_post_processing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation(
                calculate_molecular_self_energy=True,
                calculate_atomic_self_energy=False,
            )
        )

    model_name: str = "SchNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class PaiNNParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        max_Z: int = 101
        number_of_atom_features: int = 32
        number_of_radial_basis_functions: int = 20
        cutoff: str = "5.0 angstrom"
        number_of_interaction_modules: int = 3
        shared_interactions: bool = False
        shared_filters: bool = False

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy(
            normalize=True,
            from_atom_to_molecule_reduction=True,
            keep_per_atom_property=True,
        )
        general_post_processing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation(
                calculate_molecular_self_energy=False,
                calculate_atomic_self_energy=False,
            )
        )

    model_name: str = "PaiNN"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class PhysNetParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        max_Z: int = 101
        number_of_atom_features: int = 64
        number_of_radial_basis_functions: int = 16
        cutoff: str = "5.0 angstrom"
        number_of_interaction_residual: int = 3
        number_of_modules: int = 5

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy(
            normalize=True,
            from_atom_to_molecule_reduction=True,
            keep_per_atom_property=True,
        )
        general_post_processing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation(
                calculate_molecular_self_energy=False,
                calculate_atomic_self_energy=False,
            )
        )

    model_name: str = "PhysNet"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter


class SAKEParameters(ParametersBase):
    class CoreParameter(ParametersBase):
        max_Z: int = 101
        number_of_atom_features: int = 64
        number_of_radial_basis_functions: int = 50
        cutoff: str = "5.0 angstrom"
        number_of_interaction_modules: int = 6
        number_of_spatial_attention_heads: int = 4

    class PostProcessingParameter(ParametersBase):
        per_atom_energy: PerAtomEnergy = PerAtomEnergy(
            normalize=True,
            from_atom_to_molecule_reduction=True,
            keep_per_atom_property=True,
        )
        general_post_processing_operation: GeneralPostProcessingOperation = (
            GeneralPostProcessingOperation(
                calculate_molecular_self_energy=False,
                calculate_atomic_self_energy=False,
            )
        )

    model_name: str = "SAKE"
    core_parameter: CoreParameter
    postprocessing_parameter: PostProcessingParameter

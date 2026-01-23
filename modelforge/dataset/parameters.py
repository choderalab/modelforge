from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class CaseInsensitiveEnum(str, Enum):
    """
    Enum class that allows case-insensitive comparison of its members.
    """

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


class DataSetName(CaseInsensitiveEnum):
    QM9 = "QM9"
    ANI1X = "ANI1X"
    ANI2X = "ANI2X"
    SPICE1 = "SPICE1"
    SPICE2 = "SPICE2"
    SPICE1_OPENFF = "SPICE1_OPENFF"
    SPICE2_OPENFF = "SPICE2_OPENFF"
    PHALKETHOH = "PhAlkEthOH"
    TMQM = "tmQM"
    TMQM_XTB = "TMQM_XTB"
    FE_II = "FE_II"


class PropertiesDefinition(ParametersBase):
    atomic_numbers: str  # atomic numbers of the atoms in the system
    positions: str  # positions of the atoms in the system
    E: str  # total energy of the system
    F: Optional[str] = None  # forces on each atom in the system
    dipole_moment: Optional[str] = None  # dipole moment of the system
    total_charge: Optional[str] = None  # total charge of the system
    spin_multiplicity: Optional[str] = None  # spin multiplicity of the system
    partial_charges: Optional[str] = None  # partial charges on each atom in the system
    quadrupole_moment: Optional[str] = None  # quadrupole moment of the system
    # spin_multiplicity_per_atom: Optional[str] = None  # spin multiplicity per atom


class DatasetParameters(BaseModel):
    """
    Class to hold the dataset parameters.

    Attributes
    ----------
    dataset_name : DataSetName
        The name of the dataset.
    version_select : str
        The version of the dataset to use.
    num_workers : int
        The number of workers to use for the DataLoader.
    pin_memory : bool
        Whether to pin memory for the DataLoader.
    regenerate_processed_cache : bool
        Whether to regenerate the processed cache.
    properties_of_interest : List[str]
        The properties of interest to load from the hdf5 file.
    regression_ase : Optional[bool]= False
        If true, self-energies will be regressed from the dataset. defaults to False.
    properties_assignment : PropertiesDefinition
        Association between the properties of interest and the internal naming convention
    element_filter : List[tuple], optional
        A list of tuples containing the atomic numbers and their corresponding stoichiometry.
        This is used to filter the dataset to only include elements of interest.
    local_yaml_file : Optional[str]
        The path to the local yaml file with parameters for the dataset.
        If not provided, the dataset_name will be used to load an modelforge dataset.
    dataset_cache_dir : Optional[str]= "./"
        The directory where the dataset cache is stored.
        If not provided, the default cache directory will be used.

    """

    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    dataset_name: str
    version_select: str
    num_workers: int = Field(gt=0)
    pin_memory: bool
    regenerate_processed_cache: bool = False
    regression_ase: Optional[bool] = False
    regression_ase: Optional[bool] = False
    properties_of_interest: List[str]
    properties_assignment: PropertiesDefinition
    element_filter: List[tuple] = None
    local_yaml_file: Optional[str] = None
    dataset_cache_dir: Optional[str] = "./"

    # we are going to check if the datasetname is in the DataSetName enum
    # if not, local_yaml_file should be set to the path of the yaml file
    @model_validator(mode="after")
    def validate_dataset_name(self) -> Self:
        """
        Validate that the dataset name is in the DataSetName enum.
        """
        if self.local_yaml_file is None:

            try:
                DataSetName(self.dataset_name)
            except ValueError:
                msg = (
                    f"Dataset name {self.dataset_name} is not available in modelforge."
                )

                msg += "Please provide a path to the yaml file with parameters by setting local_yaml_file."
                raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_properties(self) -> Self:
        """
        Validate that the properties of interest are in the properties assignment.

        Note, datasets will validate the properties_of_interest against available properties in the dataset,
        so we do not need additional validation here.
        """
        for prop in self.properties_assignment.model_dump().values():
            if prop not in self.properties_of_interest:
                if prop is not None:
                    raise ValueError(
                        f"Property {prop} is not in the properties_of_interest."
                    )

        return self

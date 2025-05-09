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
    PHALKETHOH = "PhAlkEthOH"
    TMQM = "tmQM"
    TMQM_XTB = "TMQM_XTB"
    FE_II = "FE_II"


class PropertiesDefinition(ParametersBase):
    atomic_numbers: str
    positions: str
    E: str
    F: Optional[str] = None
    dipole_moment: Optional[str] = None
    total_charge: Optional[str] = None
    S: Optional[str] = None


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
    properties_assignment : PropertiesDefinition
        Association between the properties of interest and the internal naming convention
    """

    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
    )

    dataset_name: DataSetName
    version_select: str
    num_workers: int = Field(gt=0)
    pin_memory: bool
    regenerate_processed_cache: bool = False
    properties_of_interest: List[str]
    properties_assignment: PropertiesDefinition
    element_filter: List[tuple] = None

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

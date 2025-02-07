from typing_extensions import Annotated
from pydantic import BeforeValidator, PlainSerializer

import numpy as np
from typing import Union
from openff.units import unit

__all__ = ["NdArray"]


# Define a serializer for numpy ndarrays for pydantic
def nd_array_validator(v):
    return v


def nd_array_serializer(v):
    return str(v)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_validator),
    PlainSerializer(nd_array_serializer, return_type=str),
]


# if units are given as openff.unit compatible strings, convert them to openff unit.Unit objects
def _convert_unit_str_to_unit_unit(value: Union[str, unit.Unit]):
    """
    This will convert a string representation of a unit to an openff unit.Unit object

    If the input is a unit.Unit, nothing will be changed.

    Parameters
    ----------
    value: Union[str, unit.Unit]
        The value to convert to a unit.Unit object

    Returns
    -------
        unit.Unit

    """
    if isinstance(value, str):
        return unit.Unit(value)
    return value


def _convert_list_to_ndarray(value: Union[list, np.ndarray]):
    """
    This will convert a list to a numpy ndarray

    If the input is a numpy ndarray, nothing will be changed.

    Parameters
    ----------
    value: Union[list, np.ndarray]
        The value to convert to a numpy ndarray

    Returns
    -------
        np.ndarray

    """
    if isinstance(value, list):
        return np.array(value)
    return value

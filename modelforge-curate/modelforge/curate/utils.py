from typing_extensions import Annotated
from pydantic import BeforeValidator, PlainSerializer

import numpy as np
from typing import Union

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

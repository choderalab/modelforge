import numpy as np

from .utils import pad_molecules, pad_to_max_length

default_transformation = {
    "geometry": pad_molecules,
    "atomic_numbers": pad_to_max_length,
    "all": np.array,
}

import h5py
import pint
from openff.units import unit, Quantity
import numpy as np
from tqdm import tqdm

# define new context for converting energy to energy/mol

c = unit.Context("chem")
c.add_transformation(
    "[force] * [length]",
    "[force] * [length]/[substance]",
    lambda unit, x: x * unit.avogadro_constant,
)
c.add_transformation(
    "[force] * [length]/[substance]",
    "[force] * [length]",
    lambda unit, x: x / unit.avogadro_constant,
)
unit.add_context(c


def dict_to_hdf5(file_name: str, data: list, id_key: str) -> None:
    """
    Writes hdf5 file from dict.

    Parameters
    ----------
    file_name: str, required
        Name and path of hdf5 file to write.
    data: list of dicts, required
        List that contains dictionaries of properties for each molecule to write to file.
    id_key: str, required
        Name of the key in each dict that uniquely describes each molecule.

    Examples
    --------
    > dict_to_hdf5(file_name='qm9.hdf5', data=data, id_key='name')
    """
    assert file_name.endswith(".hdf5")

    dt = h5py.special_dtype(vlen=str)

    with h5py.File(file_name, "w") as f:
        for datapoint in tqdm(data):
            record_name = datapoint[id_key]
            group = f.create_group(record_name)
            for key, val in datapoint.items():
                if isinstance(val, pint.Quantity):
                    val_m = val.m
                    val_u = str(val.u)
                else:
                    val_m = val
                    val_u = None
                if isinstance(val_m, str):
                    group.create_dataset(name=key, data=val_m, dtype=dt)
                elif isinstance(val_m, (float, int)):
                    group.create_dataset(name=key, data=val_m)
                elif isinstance(val_m, np.ndarray):
                    group.create_dataset(name=key, data=val_m, shape=val_m.shape)
                if not val_u is None:
                    group[key].attrs["units"] = val_u

import pytest

from pydantic import ValidationError
from modelforge.potential import _Implemented_NNPs


def test_dataset_parameter_model():

    from modelforge.dataset.dataset import DatasetParameters

    # test to ensure we can properly initialize
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": 4,
        "pin_memory": True,
    }

    dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test the validator on num_workers that asserts it must be greater than 0
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": -1,
        "pin_memory": True,
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test to ensure error is raised if we do not provide all parameters
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": 4,
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test to ensure error is raised if we set a wrong type
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": 4,
        "pin_memory": "totally_true",
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # we should raise an error if we assign a wrong type to dataset_name
    with pytest.raises(ValidationError):
        dataset_parameters.dataset_name = 4

    # check the validator that asserts number of workers must be greater than 0 during assignment
    with pytest.raises(ValidationError):
        dataset_parameters.num_workers = 0


def test_convert_str_to_unit():
    # Test the validator that will automatically convert a string formated like "1.0 angstrom" to a unit.Quantity

    from modelforge.potential.parameters import convert_str_to_unit
    from openff.units import unit

    assert convert_str_to_unit("1.0 angstrom") == unit.Quantity("1.0 angstrom")
    assert convert_str_to_unit(unit.Quantity("1.0 angstrom")) == unit.Quantity(
        "1.0 angstrom"
    )


@pytest.mark.parametrize(
    "potential_name", _Implemented_NNPs.get_all_neural_network_names()
)
def test_potential_parameter_model(potential_name):
    from modelforge.tests.data import potential_defaults

    from importlib import resources
    import toml

    potential_path = (
        resources.files(potential_defaults) / f"{potential_name.lower()}.toml"
    )
    potential_config_dict = toml.load(potential_path)

    from modelforge.potential import _Implemented_NNP_Parameters

    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )

    # test to ensure we can properly initialize
    potential_parameters = PotentialParameters(**potential_config_dict["potential"])

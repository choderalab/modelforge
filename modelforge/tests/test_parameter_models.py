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
        "properties_of_interest": [
            "atomic_numbers",
            "geometry",
            "internal_energy_at_0K",
            "dipole_moment",
        ],
        "properties_assignment": {
            "atomic_numbers": "atomic_numbers",
            "positions": "geometry",
            "E": "internal_energy_at_0K",
        },
    }

    dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test the validator on num_workers that asserts it must be greater than 0
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": -1,
        "pin_memory": True,
        "properties_of_interest": [
            "atomic_numbers",
            "geometry",
            "internal_energy_at_0K",
            "dipole_moment",
        ],
        "properties_assignment": {
            "atomic_numbers": "atomic_numbers",
            "positions": "geometry",
            "E": "internal_energy_at_0K",
        },
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test to ensure error is raised if we do not provide all parameters
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": 4,
        "properties_of_interest": [
            "atomic_numbers",
            "geometry",
            "internal_energy_at_0K",
            "dipole_moment",
        ],
        "properties_assignment": {
            "atomic_numbers": "atomic_numbers",
            "positions": "geometry",
            "E": "internal_energy_at_0K",
        },
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # test to ensure error is raised if we set a wrong type
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": 4,
        "pin_memory": "totally_true",
        "properties_of_interest": [
            "atomic_numbers",
            "geometry",
            "internal_energy_at_0K",
            "dipole_moment",
        ],
        "properties_assignment": {
            "atomic_numbers": "atomic_numbers",
            "positions": "geometry",
            "E": "internal_energy_at_0K",
        },
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)

    # we should raise an error if we assign a wrong type to dataset_name
    with pytest.raises(ValidationError):
        dataset_parameters.dataset_name = 4

    # check the validator that asserts number of workers must be greater than 0 during assignment
    with pytest.raises(ValidationError):
        dataset_parameters.num_workers = 0

    # test the validator to  ensure properties_assignment has values in properties_of_interest
    dataset_parameter_dict = {
        "dataset_name": "QM9",
        "version_select": "latest",
        "num_workers": -1,
        "pin_memory": True,
        "properties_of_interest": [
            "atomic_numbers",
            "geometry",
            "internal_energy_at_0K",
            "dipole_moment",
        ],
        "properties_assignment": {
            "atomic_numbers": "atomic_numbers",
            "positions": "geometry",
            "E": "internal_energy_at_300K",
        },
    }

    with pytest.raises(ValidationError):
        dataset_parameters = DatasetParameters(**dataset_parameter_dict)


def test_convert_str_to_unit():
    # Test the validator that will automatically convert a string formated like "1.0 angstrom" to a unit.Quantity

    from modelforge.utils.units import _convert_str_to_unit
    from openff.units import unit

    assert _convert_str_to_unit("1.0 angstrom") == unit.Quantity("1.0 angstrom")
    assert _convert_str_to_unit(unit.Quantity("1.0 angstrom")) == unit.Quantity(
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


def test_runtime_parameter_model():
    from modelforge.train.parameters import RuntimeParameters
    from modelforge.tests.data import runtime_defaults

    from importlib import resources
    import toml

    runtime_path = resources.files(runtime_defaults) / "runtime.toml"
    runtime_config_dict = toml.load(runtime_path)

    # test to ensure we can properly initialize
    runtime_parameters = RuntimeParameters(**runtime_config_dict["runtime"])

    with pytest.raises(ValidationError):
        runtime_parameters.number_of_nodes = -1

    with pytest.raises(ValidationError):
        runtime_parameters.devices = -1

    with pytest.raises(ValidationError):
        runtime_parameters.devices = [-1, 0]

    with pytest.raises(ValidationError):
        runtime_parameters.accelerator = "not_a_valid_accelerator"


def test_training_parameter_model():
    from modelforge.train.parameters import TrainingParameters
    from modelforge.tests.data import training_defaults

    from importlib import resources
    import toml

    training_path = resources.files(training_defaults) / "default.toml"
    training_config_dict = toml.load(training_path)

    # test to ensure we can properly initialize
    training_parameters = TrainingParameters(**training_config_dict["training"])

    # this will throw an error because the split should sum to 1
    with pytest.raises(ValidationError):
        training_parameters.splitting_strategy.dataset_split = [0.1, 0.1, 0.1]

    # this will throw an error because the split should be of length 3
    with pytest.raises(ValidationError):
        training_parameters.splitting_strategy.dataset_split = [0.7, 0.1, 0.1, 0.1]

    # this will throw an error because the datafile has 1 entries for the loss_components dictionary
    with pytest.raises(ValidationError):
        training_parameters.loss_parameter.loss_components = [
            "per_system_energy",
            "per_atom_force",
        ]

"""
This will define a pydantic model for the training parameters, meant to take the dictionary defined in the
associated yaml file. This model will be used to validate the input dictionary and to provide default values.

The main TrainingParameters model has the following arguments:
- nr_of_epochs: int, default 1000
- remove_self_energies: bool, default True
- batch_size: int, default 128
- learning_rate: float, default 1e-3
- lr_scheduler_config: SchedulerConfig
- loss_parameter: LossParameter
- early_stopping: EarlyStopping
- splitting_strategy: SplittingStrategy

This model will have the following sub-models:
- SchedulerConfig
- LossParameter
- EarlyStopping
- SplittingStrategy

The SchedulerConfig model has the following parameters:
- frequency: int, default 1
- mode: SchedulerMode, default "min"
- factor: float, default 0.1
- patience: int, default 10
- cooldown: int, default 5
- min_learning_rate: float, default 1e-8
- threshold: float, default 0.1
- threshold_mode: ThresholdMode, default "abs"
- monitor: str, default "val/per_molecule_energy/rmse"
- interval: str, default "epoch"

The LossParameter model has the following parameters:
- loss_property: List[str], default ["per_molecule_energy", "per_atom_force"]
- loss_weight: Dict[str,float], default {"per_molecule_energy": 0.999, "per_atom_force": 0.001}

The EarlyStopping model has the following parameters:
- verbose: bool, default True
- monitor: str, default "val/per_molecule_energy/rmse"
- min_delta: float, default 0.001
- patience: int, default 50

The SplittingStrategy model will have the following parameters:
- name: SplittingStrategyName, default SplittingStrategyName.random_record_splitting_strategy
- data_split: List[float], default [0.8, 0.1, 0.1]
- seed: int, default 42


"""
from pydantic import BaseModel, root_validator, validator
from enum import Enum
from typing import List, Union, Dict, Optional, Callable


# So we  do not need to set use_enum_values = True for each model
# we can create a base class that will be inherited by all models
class ParametersBase(BaseModel):
    class Config:
        use_enum_values = True


class SchedulerMode(str, Enum):
    """
    Enum class for the scheduler mode, allowing "min" ond "max"

    args:
        min (str): The minimum mode
        max (str): The maximum mode
    """

    min = "min"
    max = "max"


class ThresholdMode(str, Enum):
    """
    Enum class for the threshold mode, allowing "abs" and "rel"

    args:
        abs (str): The absolute mode
        rel (str): The relative mode
    """

    abs = "abs"
    rel = "rel"


# generate an enum class for each registered splitting strategy
# NOTE: not sure if it is best to allow this to dynamically update
# or to simply hardcode in the values.  Hardcoded values would be better for
# documentation, which is really the aim of adding in the dataclasses
# from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES
#
# SplittingStrategyName = Enum(
#     "SplittingStrategyName", {name: name for name in REGISTERED_SPLITTING_STRATEGIES}
# )


class SplittingStrategyName(str, Enum):
    """
    Enum class for the splitting strategy name

    args:
        first_come_first_serve (str): The first come first serve strategy
        random_record_splitting_strategy (str): Split based on records randomly (i.e., all conformers of a molecule are in the same set)
        random_conformer_splitting_strategy (str): Split based on conformers randomly (i.e., conformers of a molecule can be in different sets)
    """

    first_come_first_serve = "first_come_first_serve"
    random_record_splitting_strategy = "random_record_splitting_strategy"
    random_conformer_splitting_strategy = "random_conformer_splitting_strategy"


class AnnealingStrategy(str, Enum):
    """
    Enum class for the annealing strategy
    """

    cos = "cos"
    linear = "linear"


class TrainingParameters(ParametersBase):
    """
    A class to hold the training parameters that inherits from the pydantic BaseModel

    args:
        nr_of_epochs (int): The number of epochs, default is 1000
        remove_self_energies (bool): Whether to remove self energies, default is True
        batch_size (int): The batch size, default is 128
        lr (float): The learning rate, default is 1e-3
        lr_scheduler_config (SchedulerConfig): The learning rate scheduler configuration.
        loss_parameter (LossParameter): The loss parameter
        early_stopping (EarlyStopping): The early stopping parameters, Optional, default is None
        splitting_strategy (SplittingStrategy): The splitting strategy
    """

    class SchedulerConfig(ParametersBase):
        """
        Scheduler configuration class

        args:
            frequency (int): The frequency of the scheduler, default is 1
            mode (SchedulerMode): The mode of the scheduler (options: "min", "max"), default is "min"
            factor (float): The factor of the scheduler, default is 0.1
            patience (int): The patience of the scheduler, default is 10
            cooldown (int): The cooldown of the scheduler, default is 5
            min_learning_rate (float): The minimum learning rate of the scheduler, default is 1e-8
            threshold (float): The threshold of the scheduler, default is 0.1
            threshold_mode (ThresholdMode): The threshold mode of the scheduler (options: "abs", "rel"), default is "abs"
            monitor (str): The monitor of the scheduler, default is "val/per_molecule_energy/rmse"
            interval (str): The interval of the scheduler, default is "epoch"

        """

        frequency: int = 1
        mode: SchedulerMode = SchedulerMode.min
        factor: float = 0.1
        patience: int = 10
        cooldown: int = 5
        min_learning_rate: float = 1e-8
        threshold: float = 0.1
        threshold_mode: ThresholdMode = ThresholdMode.abs
        monitor: str = "val/per_molecule_energy/rmse"
        interval: str = "epoch"

    class LossParameter(ParametersBase):
        """
        class to hold the loss properties

        args:
            loss_property (List[str]): The loss property, default is ["per_molecule_energy", "per_atom_force"].
                The length of this list must match the length of loss_weight
            weight (Dict[str,float]): The loss weight, default is {"per_molecule_energy": 0.999, "per_atom_force": 0.001}

        """

        loss_property: List = ["per_molecule_energy", "per_atom_force"]
        weight: Dict[str, float] = {
            "per_molecule_energy": 0.999,
            "per_atom_force": 0.001,
        }

        @root_validator
        def ensure_length_match(cls, values):
            loss_property = values.get("loss_property")
            loss_weight = values.get("weight")
            if len(loss_property) != len(loss_weight):
                raise ValueError(
                    f"The length of loss_property ({len(loss_property)}) and weight ({len(loss_weight)}) must match."
                )
            return values

    class EarlyStopping(ParametersBase):
        """
        class to hold the early stopping parameters

        args:
            verbose (bool): Whether to print the early stopping information, default is True
            monitor (str): The monitor of the early stopping, default is "val/per_molecule_energy/rmse"
            min_delta (float): The minimum delta of the early stopping, default is 0.001
            patience (int): The patience of the early stopping, default is 50
        """

        verbose: bool = True
        monitor: str = "val/per_molecule_energy/rmse"
        min_delta: float = 0.001
        patience: int = 50

    class SplittingStrategy(ParametersBase):
        """
        class to hold the splitting strategy

        args:
            name (SplittingStrategyName): The name of the splitting strategy, default is SplittingStrategyName.random_record_splitting_strategy
                Options are: first_come_first_serve, random_record_splitting_strategy, random_conformer_splitting_strategy
            data_split (List[float]): The data split, default is [0.8, 0.1, 0.1]
            seed (int): The seed, default is 42

        """

        name: SplittingStrategyName = (
            SplittingStrategyName.random_record_splitting_strategy
        )
        data_split = [0.8, 0.1, 0.1]
        seed: int = 42

    class StochasticWeightAveraging(ParametersBase):
        swa_lrs: Union[float, List[float]] = 0.05
        swa_epoch_start: float = 0.8
        annealing_epoch: int = 10
        annealing_strategy: AnnealingStrategy = AnnealingStrategy.cos
        avg_fn: Optional[Callable] = None

    nr_of_epochs: int = 1000
    remove_self_energies: bool = True
    batch_size: int = 128
    lr: float = 1e-3
    lr_scheduler_config: SchedulerConfig
    loss_parameter: LossParameter
    early_stopping: Optional[EarlyStopping] = None
    splitting_strategy: SplittingStrategy
    stochastic_weight_averaging: Optional[StochasticWeightAveraging] = None


class Accelerator(str, Enum):
    """
    Enum class for the accelerator, allowing "cpu", "gpu" and "tpu".

    args:
        cpu (str): The cpu accelerator
        gpu (str): The gpu accelerator
        tpu (str): The tpu accelerator
    """

    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"


class SimulationEnvironment(str, Enum):
    """
    Enum class for the simulation environment, allowing "PyTorch"  and "JAX".

    args:
        PyTorch (str): The torch environment
        JAX (str): The jax environment
    """

    JAX = "JAX"
    PyTorch = "PyTorch"


class RuntimeParameters(ParametersBase):
    """
    A class to hold the runtime parameters that inherits from the pydantic BaseModel

    args:
        save_dir (str): The save directory, default is "test"
        experiment_name (str): The experiment name, default is "my_experiment"
        accelerator (Accelerator): The accelerator, default is Accelerator.cpu
        number_of_nodes (int): The number of nodes, default is 1
        devices (int or List[int]): The index/indices of the device, default is 1
        local_cache_dir (str): The local cache directory, default is "./cache"
        checkpoint_path (str): The checkpoint path, default is None
        simulation_environment (SimulationEnvironment):
            The simulation environment, default is SimulationEnvironment.PyTorch
        log_every_n_steps (int): The logging frequency, default is 50

    """

    save_dir: str = "test"
    experiment_name: str = "my_experiment"
    accelerator: Accelerator = Accelerator.cpu
    number_of_nodes: int = 1
    devices: Union[int, List[int]] = 1
    local_cache_dir: str = "./cache"
    checkpoint_path: str = None
    simulation_environment: SimulationEnvironment = SimulationEnvironment.PyTorch
    log_every_n_steps: int = 50

    @validator("number_of_nodes")
    def number_of_nodes_must_be_positive(cls, v):
        if v < 1:
            raise ValueError("number_of_nodes must be positive and greater than 0")
        return v

    @validator("devices")
    def device_index_must_be_positive(cls, v):
        if isinstance(v, list):
            for device in v:
                if device < 0:
                    raise ValueError("device_index must be positive")
        if v < 0:
            raise ValueError("device_index must be positive")
        return v

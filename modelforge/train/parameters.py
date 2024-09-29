"""
This defines pydantic models for the training parameters and runtime parameters.
"""

from enum import Enum
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# So we  do not need to set Config parameters in each model
# we can create a base class that will be inherited by all models
class ParametersBase(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True, arbitrary_types_allowed=True, validate_assignment=True
    )


# for enums over strings, we likely do not want things to be case sensitive
# this will return the appropriate enum value regardless of case
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return super()._missing_(value)


class SchedulerMode(CaseInsensitiveEnum):
    """
    Enum class for the scheduler mode, allowing "min" ond "max"

    args:
        min (str): The minimum mode
        max (str): The maximum mode
    """

    min = "min"
    max = "max"


class ThresholdMode(CaseInsensitiveEnum):
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


class SplittingStrategyName(CaseInsensitiveEnum):
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


class AnnealingStrategy(CaseInsensitiveEnum):
    """
    Enum class for the annealing strategy
    """

    cos = "cos"
    linear = "linear"


#
class Loggers(CaseInsensitiveEnum):
    """
    Enum class for the experiment logger
    """

    wandb = "wandb"
    tensorboard = "tensorboard"


class TensorboardConfig(BaseModel):
    save_dir: str


class WandbConfig(BaseModel):
    save_dir: str
    project: str
    group: str
    log_model: bool
    job_type: Optional[str]
    tags: Optional[List[str]]
    notes: Optional[str]


class TrainingParameters(ParametersBase):
    """
    A class to hold the training parameters that inherits from the pydantic BaseModel

    args:
        nr_of_epochs (int): The number of epochs
        remove_self_energies (bool): Whether to remove self energies
        batch_size (int): The batch size,
        lr (float): The learning rate
        lr_scheduler (SchedulerConfig): The learning rate scheduler configuration.
        loss_parameter (LossParameter): The loss parameter
        early_stopping (EarlyStopping): The early stopping parameters, Optional
        splitting_strategy (SplittingStrategy): The splitting strategy
    """

    class SchedulerConfig(ParametersBase):
        """
        Scheduler configuration class

        args:
            frequency (int): The frequency of the scheduler
            mode (SchedulerMode): The mode of the scheduler (options: "min", "max")
            factor (float): The factor of the scheduler
            patience (int): The patience of the scheduler
            cooldown (int): The cooldown of the scheduler
            min_learning_rate (float): The minimum learning rate of the scheduler
            threshold (float): The threshold of the scheduler
            threshold_mode (ThresholdMode): The threshold mode of the scheduler (options: "abs", "rel")
            monitor (str): The monitor of the scheduler
            interval (str): The interval of the scheduler

        """

        frequency: int
        mode: SchedulerMode
        factor: float
        patience: int
        cooldown: int
        min_lr: float
        threshold: float
        threshold_mode: ThresholdMode
        monitor: str  # note this might be better as an enum
        interval: str

    class LossParameter(ParametersBase):
        """
        class to hold the loss properties

        args:
            loss_property (List[str]): The loss property.
                The length of this list must match the length of loss_weight
            weight (Dict[str,float]): The loss weight.
                The keys must correspond to entries in the loss_property list.

        """

        loss_property: List
        weight: Dict[str, float]

        @model_validator(mode="after")
        def ensure_length_match(self) -> "LossParameter":
            loss_property = self.loss_property
            loss_weight = self.weight
            if len(loss_property) != len(loss_weight):
                raise ValueError(
                    f"The length of loss_property ({len(loss_property)}) and weight ({len(loss_weight)}) must match."
                )
            return self

    class EarlyStopping(ParametersBase):
        """
        class to hold the early stopping parameters

        args:
            verbose (bool): Whether to print the early stopping information
            monitor (str): The monitor of the early stopping
            min_delta (float): The minimum delta of the early stopping
            patience (int): The patience of the early stopping
        """

        verbose: bool
        monitor: str
        min_delta: float
        patience: int

    class SplittingStrategy(ParametersBase):
        """
        class to hold the splitting strategy

        args:
            name (SplittingStrategyName): The name of the splitting strategy, default is SplittingStrategyName.random_record_splitting_strategy
                Options are: first_come_first_serve, random_record_splitting_strategy, random_conformer_splitting_strategy
            data_split (List[float]): The data split, must be a list of length of 3 that sums to 1.
            seed (int): The seed

        """

        name: SplittingStrategyName = (
            SplittingStrategyName.random_record_splitting_strategy
        )
        data_split: List[float]
        seed: int

        @field_validator("data_split")
        def data_split_must_sum_to_one_and_length_three(cls, v) -> List[float]:

            if len(v) != 3:
                raise ValueError("data_split must have length of 3")
            if sum(v) != 1:
                raise ValueError("data_split must sum to 1")
            return v

    class StochasticWeightAveraging(ParametersBase):
        """
        class to hold the stochastic weight averaging parameters

        args:
            swa_lrs (Union[float, List[float]]): The learning rate for stochastic weight averaging
            swa_epoch_start (float): The epoch start for stochastic weight averaging
            annealing_epoch (int): The annealing epoch
            annealing_strategy (AnnealingStrategy): The annealing strategy
            avg_fn (Optional[Callable]): The average function
        """

        swa_lrs: Union[float, List[float]]
        swa_epoch_start: float
        annealing_epoch: int
        annealing_strategy: AnnealingStrategy
        avg_fn: Optional[Callable] = None

    class ExperimentLogger(ParametersBase):
        logger_name: Loggers
        tensorboard_configuration: Optional[TensorboardConfig] = None
        wandb_configuration: Optional[WandbConfig] = None

        @model_validator(mode="after")
        def ensure_logger_configuration(self) -> "ExperimentLogger":
            if (
                self.logger_name == Loggers.tensorboard
                and self.tensorboard_configuration is None
            ):
                raise ValueError("tensorboard_configuration must be provided")
            if self.logger_name == Loggers.wandb and self.wandb_configuration is None:
                raise ValueError("wandb_configuration must be provided")
            return self

    number_of_epochs: int
    remove_self_energies: bool
    batch_size: int
    lr: float
    monitor_for_checkpoint: str
    lr_scheduler: Optional[SchedulerConfig] = None
    loss_parameter: LossParameter
    early_stopping: Optional[EarlyStopping] = None
    splitting_strategy: SplittingStrategy
    stochastic_weight_averaging: Optional[StochasticWeightAveraging] = None
    experiment_logger: ExperimentLogger
    verbose: bool = False
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW
    min_number_of_epochs: Union[int, None] = None


### Runtime Parameters


class Accelerator(CaseInsensitiveEnum):
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


class SimulationEnvironment(CaseInsensitiveEnum):
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
        experiment_name (str): The experiment name
        accelerator (Accelerator): The accelerator, options are: "cpu", "gpu", "tpu"
        number_of_nodes (int): The number of nodes
        devices (int or List[int]): The index/indices of the device
        local_cache_dir (str): The local cache directory
        checkpoint_path (str,None): The checkpoint path
        simulation_environment (SimulationEnvironment):
            The simulation environment options are: PyTorch or JAX
        log_every_n_steps (int): The logging frequency

    """

    experiment_name: str
    accelerator: Accelerator
    number_of_nodes: int
    devices: Union[int, List[int]]
    local_cache_dir: str
    checkpoint_path: Union[str, None]
    simulation_environment: SimulationEnvironment
    log_every_n_steps: int

    @field_validator("number_of_nodes")
    @classmethod
    def number_of_nodes_must_be_positive(cls, v) -> int:
        if v < 1:
            raise ValueError("number_of_nodes must be positive and greater than 0")
        return v

    @field_validator("devices")
    @classmethod
    def device_index_must_be_positive(cls, v) -> Union[int, List[int]]:
        if isinstance(v, list):
            for device in v:
                if device < 0:
                    raise ValueError("device_index must be positive")
        else:
            if v < 0:
                raise ValueError("device_index must be positive")
        return v

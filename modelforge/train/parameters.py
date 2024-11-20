"""
This defines pydantic models for the training parameters and runtime parameters.
"""

from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Literal,
    Annotated,
)

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field
from loguru import logger as log


# So we do not need to set Config parameters in each model
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
    Enum class for the scheduler mode, allowing "min" and "max"
    """

    min = "min"
    max = "max"


class ThresholdMode(CaseInsensitiveEnum):
    """
    Enum class for the threshold mode, allowing "abs" and "rel"
    """

    abs = "abs"
    rel = "rel"


class SchedulerName(CaseInsensitiveEnum):
    """
    Enum class for the scheduler names
    """

    ReduceLROnPlateau = "ReduceLROnPlateau"
    CosineAnnealingLR = "CosineAnnealingLR"
    CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    OneCycleLR = "OneCycleLR"
    CyclicLR = "CyclicLR"


class SplittingStrategyName(CaseInsensitiveEnum):
    """
    Enum class for the splitting strategy name
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


class Loggers(CaseInsensitiveEnum):
    """
    Enum class for the experiment logger
    """

    wandb = "wandb"
    tensorboard = "tensorboard"


class TensorboardConfig(ParametersBase):
    save_dir: str


class WandbConfig(ParametersBase):
    save_dir: str
    project: str
    group: str
    log_model: Union[str, bool]
    job_type: Optional[str]
    tags: Optional[List[str]]
    notes: Optional[str]


class SchedulerConfigBase(ParametersBase):
    """
    Base class for scheduler configurations
    """

    scheduler_name: SchedulerName
    frequency: int
    interval: str
    monitor: Optional[str] = None


class ReduceLROnPlateauConfig(SchedulerConfigBase):
    """
    Configuration for ReduceLROnPlateau scheduler
    """

    scheduler_name: Literal[SchedulerName.ReduceLROnPlateau] = (
        SchedulerName.ReduceLROnPlateau
    )
    mode: SchedulerMode  # 'min' or 'max'
    factor: float
    patience: int
    threshold: float
    threshold_mode: ThresholdMode  # 'rel' or 'abs'
    cooldown: int
    min_lr: float
    eps: float = 1e-8


class CosineAnnealingLRConfig(SchedulerConfigBase):
    """
    Configuration for CosineAnnealingLR scheduler
    """

    scheduler_name: Literal[SchedulerName.CosineAnnealingLR] = (
        SchedulerName.CosineAnnealingLR
    )
    T_max: int
    eta_min: float = 0.0
    last_epoch: int = -1


class CosineAnnealingWarmRestartsConfig(SchedulerConfigBase):
    """
    Configuration for CosineAnnealingWarmRestarts scheduler
    """

    scheduler_name: Literal[SchedulerName.CosineAnnealingWarmRestarts] = (
        SchedulerName.CosineAnnealingWarmRestarts
    )
    T_0: int
    T_mult: int = 1
    eta_min: float = 0.0
    last_epoch: int = -1


class CyclicLRMode(CaseInsensitiveEnum):
    """
    Enum class for the CyclicLR modes
    """

    triangular = "triangular"
    triangular2 = "triangular2"
    exp_range = "exp_range"


class ScaleMode(CaseInsensitiveEnum):
    """
    Enum class for the scale modes
    """

    cycle = "cycle"
    iterations = "iterations"


class OneCycleLRConfig(SchedulerConfigBase):
    """
    Configuration for OneCycleLR scheduler
    """

    scheduler_name: Literal[SchedulerName.OneCycleLR] = SchedulerName.OneCycleLR
    max_lr: Union[float, List[float]]
    epochs: int  # required
    pct_start: float = 0.3
    anneal_strategy: AnnealingStrategy = AnnealingStrategy.cos
    cycle_momentum: bool = True
    base_momentum: Union[float, List[float]] = 0.85
    max_momentum: Union[float, List[float]] = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    three_phase: bool = False
    last_epoch: int = -1

    @model_validator(mode="after")
    def validate_epochs(self):
        if self.epochs is None:
            raise ValueError("OneCycleLR requires 'epochs' to be set.")
        if self.interval != "step":
            raise ValueError("OneCycleLR requires 'interval' to be set to 'step'.")

        return self


class CyclicLRConfig(SchedulerConfigBase):
    """
    Configuration for CyclicLR scheduler
    """

    scheduler_name: Literal[SchedulerName.CyclicLR] = SchedulerName.CyclicLR
    base_lr: Union[float, List[float]]
    max_lr: Union[float, List[float]]
    epochs_up: float  # Duration of the increasing phase in epochs
    epochs_down: Optional[float] = (
        None  # Duration of the decreasing phase in epochs (optional)
    )
    mode: CyclicLRMode = CyclicLRMode.triangular
    gamma: float = 1.0
    scale_mode: ScaleMode = ScaleMode.cycle
    cycle_momentum: bool = True
    base_momentum: Union[float, List[float]] = 0.8
    max_momentum: Union[float, List[float]] = 0.9
    last_epoch: int = -1

    @model_validator(mode="after")
    def validate_epochs(self):
        if self.epochs_up is None:
            raise ValueError("CyclicLR requires 'epochs_up' to be set.")

        if self.interval != "step":
            raise ValueError("OneCycleLR requires 'interval' to be set to 'step'.")
        return self


SchedulerConfig = Annotated[
    Union[
        ReduceLROnPlateauConfig,
        CosineAnnealingLRConfig,
        CosineAnnealingWarmRestartsConfig,
        OneCycleLRConfig,
        CyclicLRConfig,
    ],
    Field(discriminator="scheduler_name"),
]


class TrainingParameters(ParametersBase):
    """
    A class to hold the training parameters
    """

    class LossParameter(ParametersBase):
        """
        Class to hold the loss properties and mixing scheme
        """

        loss_components: List[str]
        weight: Dict[str, float]
        target_weight: Optional[Dict[str, float]] = None
        mixing_steps: Optional[Dict[str, float]] = None

        @model_validator(mode="before")
        def set_target_weight_defaults(cls, values):

            # if no target_weight is provided, set target_weight to be the same
            # as weight and mixing_steps to be 1.0
            if "target_weight" not in values:
                # set target_weight to be the same as weight
                values["target_weight"] = values["weight"]
                d = {}
                for key in values["target_weight"]:
                    d[key] = 1.0
                values["mixing_steps"] = d
                return values

            # if target weight is not provided for all properties, set the rest
            # to be the same as weight
            for key in values["weight"]:
                # if only a subset of the loss components are provided in target_weight, set the rest to be the same as weight
                if key not in values["target_weight"]:
                    values["target_weight"][key] = values["weight"][key]
                    values["mixing_steps"][key] = 1.0
            return values

        @model_validator(mode="after")
        def ensure_length_match(self) -> "LossParameter":
            loss_components = self.loss_components
            weight = self.weight
            if len(loss_components) != len(weight):
                raise ValueError(
                    f"The length of loss_components ({len(loss_components)}) and weight ({len(weight)}) must match."
                )
            if set(loss_components) != set(weight.keys()):
                raise ValueError("The keys of weight must match loss_components")

            if self.target_weight or self.mixing_steps:
                if not (self.target_weight and self.mixing_steps):
                    raise ValueError(
                        "If using mixing scheme, target_weight and mixing_steps must all be provided"
                    )
                if set(self.target_weight.keys()) != set(loss_components):
                    raise ValueError(
                        "The keys of target_weight must match loss_components"
                    )
            return self

    class EarlyStopping(ParametersBase):
        """
        Class to hold the early stopping parameters
        """

        verbose: bool
        monitor: Optional[str] = None
        min_delta: float
        patience: int

    class SplittingStrategy(ParametersBase):
        """
        Class to hold the splitting strategy
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
        Class to hold the stochastic weight averaging parameters
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

    monitor: str
    number_of_epochs: int
    remove_self_energies: bool
    shift_center_of_mass_to_origin: bool
    batch_size: int
    lr: float
    plot_frequency: int = 5  # how often to log regression and error histograms
    lr_scheduler: Optional[SchedulerConfig] = None
    loss_parameter: LossParameter
    early_stopping: Optional[EarlyStopping] = None
    splitting_strategy: SplittingStrategy
    stochastic_weight_averaging: Optional[StochasticWeightAveraging] = None
    experiment_logger: ExperimentLogger
    verbose: bool = False
    log_norm: bool = False
    limit_train_batches: Union[float, int, None] = None
    limit_val_batches: Union[float, int, None] = None
    limit_test_batches: Union[float, int, None] = None
    profiler: Optional[str] = None
    optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW
    min_number_of_epochs: Union[int, None] = None

    @model_validator(mode="after")
    def validate_dipole_and_shift_com(self):
        if "dipole_moment" in self.loss_parameter.loss_components:
            if not self.shift_center_of_mass_to_origin:
                raise ValueError(
                    "Use of dipole_moment in the loss requires shift_center_of_mass_to_origin to be True"
                )
        return self

    # Validator to set default monitors
    @model_validator(mode="after")
    def set_default_monitors(self) -> "TrainingParameters":
        if self.lr_scheduler and self.lr_scheduler.monitor is None:
            self.lr_scheduler.monitor = self.monitor
        if self.early_stopping and self.early_stopping.monitor is None:
            self.early_stopping.monitor = self.monitor
        return self


class Accelerator(CaseInsensitiveEnum):
    """
    Enum class for the accelerator, allowing "cpu", "gpu" and "tpu".
    """

    cpu = "cpu"
    gpu = "gpu"
    tpu = "tpu"


class SimulationEnvironment(CaseInsensitiveEnum):
    """
    Enum class for the simulation environment, allowing "PyTorch" and "JAX".
    """

    JAX = "JAX"
    PyTorch = "PyTorch"


class RuntimeParameters(ParametersBase):
    """
    A class to hold the runtime parameters
    """

    experiment_name: str
    accelerator: Accelerator
    number_of_nodes: int
    devices: Union[int, List[int]]
    local_cache_dir: str
    checkpoint_path: Union[str, None]
    simulation_environment: SimulationEnvironment
    log_every_n_steps: int
    verbose: bool

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

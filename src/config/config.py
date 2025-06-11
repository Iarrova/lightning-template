from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt

from src.datasets.enums import Dataset
from src.exceptions import ConfigurationError
from src.model.optimizers import Optimizer
from src.model.schedulers import Scheduler
from src.networks.enums import Network


class TrainingConfig(BaseModel):
    learning_rate: Annotated[PositiveFloat, Field(default=0.001)]
    num_epochs: Annotated[PositiveInt, Field(default=10)]
    early_stopping_patience: Annotated[PositiveInt, Field(default=10)]
    optimizer: Annotated[Optimizer, Field(default=Optimizer.ADAM)]
    scheduler: Annotated[Scheduler, Field(default=Scheduler.REDUCE_ON_PLATEAU)]
    scheduler_patience: Annotated[PositiveInt, Field(default=5)]
    scheduler_factor: Annotated[PositiveFloat, Field(default=0.1)]


class DatasetConfig(BaseModel):
    dataset: Dataset
    batch_size: Annotated[PositiveInt, Field(default=32)]
    validation_size: Annotated[float, Field(default=0.2, gt=0, lt=1)]
    augment: Annotated[bool, Field(default=True)]
    num_workers: Annotated[PositiveInt, Field(default=15)]
    data_dir: Annotated[Path, Field(default=Path("./data"))]

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        import os

        max_workers = os.cpu_count() or 1
        if v > max_workers:
            raise ConfigurationError(f"num_workers ({v}) cannot exceed CPU count ({max_workers})")
        return v


class NetworkConfig(BaseModel):
    network: Network
    include_top: Annotated[bool, Field(default=True)]


class WeightsConfig(BaseModel):
    pretrained_weights: Annotated[Optional[Path | Literal["ImageNet"]], Field(default=None)]
    save_weights_path: Annotated[Path, Field(default=Path("./weights/model.ckpt"))]

    @field_validator("pretrained_weights")
    @classmethod
    def validate_pretrained_weights(cls, v):
        if isinstance(v, Path) and not v.exists():
            raise ConfigurationError(f"Pretrained weights {v} does not exist")
        return v

    @model_validator(mode="after")
    def create_save_weights_path(self) -> "WeightsConfig":
        if not self.save_weights_path.parent.exists():
            self.save_weights_path.parent.mkdir(parents=True, exist_ok=True)
        return self


class LoggingConfig(BaseModel):
    tensorboard: Annotated[bool, Field(default=True)]
    csv: Annotated[bool, Field(default=True)]
    log_dir: Annotated[Path, Field(default=Path("./logs"))]
    log_level: Annotated[str, Field(default="INFO")]

    @model_validator(mode="after")
    def create_log_dir(self) -> "LoggingConfig":
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


class Config(BaseModel):
    training: TrainingConfig
    dataset: DatasetConfig
    network: NetworkConfig
    weights: WeightsConfig
    logging: LoggingConfig
    seed: Annotated[PositiveInt, Field(default=42)]
    mixed_precision: Annotated[bool, Field(default=True)]

import json
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt

from src.datasets import Dataset
from src.exceptions import ConfigurationError
from src.model.criterion import Criterion
from src.model.metrics import Metric
from src.model.optimizers import Optimizer
from src.model.schedulers import Scheduler
from src.networks import Network


class TrainingConfig(BaseModel):
    learning_rate: Annotated[PositiveFloat, Field(default=0.001)]
    num_epochs: Annotated[PositiveInt, Field(default=10)]
    early_stopping_patience: Annotated[PositiveInt, Field(default=10)]
    optimizer: Annotated[Optimizer, Field(default=Optimizer.ADAM)]
    scheduler: Annotated[Scheduler, Field(default=Scheduler.REDUCE_ON_PLATEAU)]
    scheduler_patience: Annotated[PositiveInt, Field(default=5)]
    scheduler_factor: Annotated[PositiveFloat, Field(default=0.1)]
    metrics: Annotated[List[Metric], Field(default_factory=lambda: [Metric.ALL])]
    criterion: Annotated[Criterion, Field(default=Criterion.CrossEntropy)]
    resume_training: Annotated[bool, Field(default=False)]


class DatasetConfig(BaseModel):
    dataset: Dataset
    augment: Annotated[bool, Field(default=True)]
    num_workers: Annotated[PositiveInt, Field(default=15)]
    batch_size: Annotated[PositiveInt, Field(default=32)]
    validation_size: Annotated[float, Field(default=0.2, gt=0, lt=1)]

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        max_workers = os.cpu_count() or 1
        if v > max_workers:
            raise ConfigurationError(f"num_workers ({v}) cannot exceed CPU count ({max_workers})")
        return v


class NetworkConfig(BaseModel):
    network: Network
    include_top: Annotated[bool, Field(default=True)]
    pytorch_weights: Annotated[Optional[Path | Literal["ImageNet"]], Field(default=None)]
    lightning_checkpoint: Annotated[Optional[Path], Field(default=None)]

    @field_validator("pytorch_weights")
    @classmethod
    def validate_pytorch_weights(cls, v):
        if v is None:
            return v
        if isinstance(v, str) and v == "ImageNet":
            return v
        if isinstance(v, Path) and v.exists():
            return v
        raise ValueError(f"Invalid 'pytorch_weights': file {v} does not exist or is not 'ImageNet'.")

    @field_validator("lightning_checkpoint")
    @classmethod
    def validate_lightning_checkpoint(cls, v):
        if v is None:
            return v
        if isinstance(v, Path) and v.exists():
            return v
        raise ValueError(f"Invalid 'lightning_checkpoint': file {v} does not exist.")


class LoggingConfig(BaseModel):
    tensorboard: Annotated[bool, Field(default=True)]
    csv: Annotated[bool, Field(default=True)]
    log_dir: Annotated[Path, Field(default=Path("./logs"))]

    @model_validator(mode="after")
    def create_directories(self) -> "LoggingConfig":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


class Config(BaseModel):
    training: TrainingConfig
    dataset: DatasetConfig
    network: NetworkConfig
    logging: LoggingConfig
    save_weights_path: Annotated[Path, Field(default=Path("./weights/model.ckpt"))]
    seed: Annotated[PositiveInt, Field(default=42)]

    @model_validator(mode="after")
    def create_directories(self) -> "Config":
        self.save_weights_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_json(cls, path: str) -> "Config":
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as file:
            try:
                data: Dict[str, Any] = json.load(file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in configuration file: {str(e)}")

        return cls(**data)

from typing import Annotated, Any, Dict, Optional

from pydantic import BaseModel, Field

from datasets import Datasets
from networks import Networks


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, Field(strict=True, gt=0, default=32)]
    validation_size: Annotated[float, Field(strict=True, gt=0, lt=1, default=0.2)]
    learning_rate: Annotated[float, Field(strict=True, gt=0, default=0.001)]
    num_epochs: Annotated[int, Field(strict=True, gt=0, default=10)]
    early_stopping_patience: Annotated[int, Field(strict=True, gt=0, default=10)]
    optimizer: Annotated[str, Field(default="adam")]
    scheduler: Annotated[str, Field(default="reduce_on_plateau")]
    scheduler_patience: Annotated[int, Field(strict=True, gt=0, default=5)]
    scheduler_factor: Annotated[float, Field(strict=True, gt=0, default=0.1)]


class DatasetConfig(BaseModel):
    dataset: Datasets
    augment: Annotated[bool, Field(default=True)]
    num_workers: Annotated[int, Field(strict=True, gt=0, default=15)]


class NetworkConfig(BaseModel):
    network: Networks
    include_top: Annotated[bool, Field(default=True)]
    pytorch_weights: Annotated[Optional[str], Field(default=None)]


class LoggingConfig(BaseModel):
    tensorboard: Annotated[bool, Field(default=True)]
    csv: Annotated[bool, Field(default=True)]
    log_dir: Annotated[str, Field(default="logs")]
    weights_dir: Annotated[str, Field(default="weights")]
    weights_path: Annotated[str, Field(default="model.ckpt")]


class Config(BaseModel):
    training: TrainingConfig
    dataset: DatasetConfig
    network: NetworkConfig
    logging: LoggingConfig
    seed: int = 42
    mixed_precission: bool = True
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)

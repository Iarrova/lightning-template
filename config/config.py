from typing import Annotated

from pydantic import BaseModel, Field

from datasets.constants import Datasets
from networks.constants import Networks


class Config(BaseModel):
    dataset: Datasets
    batch_size: int
    validation_size: Annotated[float, Field(strict=True, gt=0, lt=1)]
    augment: bool
    network: Networks
    include_top: bool
    weights: str
    learning_rate: Annotated[float, Field(strict=True, gt=0)]
    num_epochs: int
    weights_dir: str
    weights_path: str

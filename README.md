# PyTorch Training Template

A flexible PyTorch training framework with modular components for datasets, networks, metrics, optimizers, and schedulers.

## Getting Started

1. Install dependencies: `uv sync`
2. Copy `config.example.json` to `config.json` and modify as needed
3. Run training: `python train.py --config-path config.json`
4. Run testing: `python test.py --config-path config.json`

## Extending the Framework

### Custom Datasets

Create a new dataset by inheriting from `BaseDataset`:

```python
# src/datasets/my_dataset.py
from src.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    NUM_CLASSES: int = 10  # Set number of classes

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = "./data/my_dataset"

    def get_transforms(self):
        # Return (train_transform, test_transform)
        pass

    def get_train_dataset(self, transform):
        # Return PyTorch Dataset for training
        pass

    def get_test_dataset(self, transform):
        # Return PyTorch Dataset for testing
        pass

    def get_class_mapping(self):
        # Return dict mapping class names to indices
        pass
```

Register in `src/datasets/__init__.py`:

```python
from src.datasets.my_dataset import MyDataset

class Dataset(StrEnum):
    MY_DATASET = "MyDataset"

DATASET_MAPPING = {
    Dataset.MY_DATASET: MyDataset,
}
```

### Custom Networks

Create a new network by inheriting from `BaseNetwork`:

```python
# src/networks/my_network.py
from src.networks.base import BaseNetwork

class MyNetwork(BaseNetwork):
    def _create_model(self):
        # Create and return your model
        # Access config via self.network_config
        # Access num_classes via self.num_classes
        pass
```

Register in `src/networks/__init__.py`:

```python
from src.networks.my_network import MyNetwork

class Network(StrEnum):
    MY_NETWORK = "MyNetwork"

NETWORK_MAPPING = {
    Network.MY_NETWORK: MyNetwork,
}
```

### Custom Metrics

Add new metrics in `src/model/metrics.py`:

```python
class Metric(StrEnum):
    MY_METRIC = "my_metric"

# In MetricFactory.create():
metric_map = {
    Metric.MY_METRIC: MyTorchMetric(num_classes=num_classes),
}
```

### Custom Optimizers

Add new optimizers in `src/model/optimizers.py`:

```python
class Optimizer(StrEnum):
    MY_OPTIMIZER = "MyOptimizer"

# In OptimizerFactory.create():
match optimizer:
    case Optimizer.MY_OPTIMIZER:
        return optim.MyOptimizer(params, lr=learning_rate)
```

### Custom Schedulers

Add new schedulers in `src/model/schedulers.py`:

```python
class Scheduler(StrEnum):
    MY_SCHEDULER = "MyScheduler"

# In SchedulerFactory.create():
match scheduler:
    case Scheduler.MY_SCHEDULER:
        return optim.lr_scheduler.MyScheduler(optimizer, **kwargs)
```

## Configuration

The framework uses JSON configuration files. See `config.example.json` for all available options:

- `training`: Learning rate, epochs, optimizer, scheduler, metrics, etc.
- `dataset`: Dataset type, augmentation, batch size, workers, etc.
- `network`: Network architecture, pretrained weights, etc.
- `logging`: TensorBoard, CSV logging, log directory

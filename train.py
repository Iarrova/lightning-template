from argparse import ArgumentParser, Namespace

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from config.config import Config
from config.json_parser import parse_json
from datasets.utils.load_dataset import load_dataset
from model import Model
from networks.utils.load_network import load_network

L.seed_everything(42)

parser: ArgumentParser = ArgumentParser(description="Template for PyTorch Lightning prototyping.")
parser.add_argument(
    "--config-path",
    help="The path to the configuration file to use in training/testing.",
)
args: Namespace = parser.parse_args()

config: Config = parse_json(args.config_path)

if torch.cuda.is_available():
    print("[INFO] CUDA is available! Training on GPU...")
else:
    print("[INFO] CUDA is not available. Training on CPU...")

dataset = load_dataset(config)
train_loader, validation_loader = dataset.generate_train_loaders()

network = load_network(config, dataset.num_classes)
model = Model(network, config, dataset.num_classes)

callbacks = [
    ModelCheckpoint(
        dirpath=config.weights_dir,
        filename=config.weights_path,
        monitor="val_loss",
        mode="min",
        verbose=True,
    ),
    EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True),
    LearningRateMonitor(),
]

trainer = L.Trainer(
    max_epochs=config.num_epochs,
    callbacks=callbacks,
    accelerator="auto",
    devices="auto",
    logger=[
        CSVLogger(save_dir=f"logs/{config.weights_path}"),
        TensorBoardLogger(save_dir=f"logs/{config.weights_path}"),
    ],
)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

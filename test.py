import argparse

import torch
import torch.nn as nn

import lightning as L

from torchmetrics.classification import Accuracy

from utils.json_parser import parse_json

parser = argparse.ArgumentParser(
    description="Template for PyTorch Lightning prototyping."
)
parser.add_argument(
    "--config_path",
    help="The path to the configuration file to use in training/testing.",
)
args = parser.parse_args()

config = parse_json(args.config_path)

L.seed_everything(config.seed)
if torch.cuda.is_available():
    print("[INFO] CUDA is available! Training on GPU...")
else:
    print("[INFO] CUDA is not available. Training on CPU...")


# Dataset
if config.dataset == "CIFAR10":
    from datasets.CIFAR10 import generate_CIFAR10 as generate_dataset

    num_classes = 10
else:
    print("[ERROR] Currently only CIFAR10 dataset is supported. Exiting...")
    exit(1)

train_loader, validation_loader, test_loader, classes = generate_dataset(
    batch_size=config.batch_size,
    validation_size=config.validation_size,
    augment=config.augment,
)


# Network
if config.network == "ResNet50":
    from networks.resnet50 import ResNet50 as network
else:
    print("[ERROR] Currently only ResNet50 network is supported. Exiting...")

model = network(include_top=False, weights="imagenet", num_classes=num_classes)


# PyTorch Lightning
class LitModel(L.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

        self.test_accuracy = Accuracy(task="multiclass", num_classes=len(classes))

    def step(self, batch):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.criterion(output, target)
        return loss, output, target

    def test_step(self, batch, batch_idx):
        loss, output, target = self.step(batch)
        self.test_accuracy.update(output, target)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )


criterion = nn.CrossEntropyLoss()
weights_path = f"{config.weights_dir}/{config.weights_path}.ckpt"
litmodel = LitModel.load_from_checkpoint(weights_path, model=model, criterion=criterion)

trainer = L.Trainer()
trainer.test(model=litmodel, dataloaders=test_loader)

from argparse import ArgumentParser, Namespace

import lightning as L
import torch

# import seaborn as sns
# import matplotlib.pyplot as plt
from config.config import Config
from config.json_parser import parse_json
from model import Model
from utils.load_dataset import load_dataset
from utils.load_network import load_network

L.seed_everything(42)

parser: ArgumentParser = ArgumentParser(
    description="Template for PyTorch Lightning prototyping."
)
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
test_loader = dataset.generate_test_loader()

network = load_network(config, dataset.num_classes)

weights_path = f"{config.weights_dir}/{config.weights_path}.ckpt"
model = Model.load_from_checkpoint(
    weights_path, network=network, config=config, num_classes=dataset.num_classes
)

trainer = L.Trainer(devices="auto", logger=False)
trainer.test(model=model, dataloaders=test_loader)

"""
for metric_name, metric_instance in model.metrics.items():
    print(f"{metric_name}: {metric_instance.compute()}")

fig, ax = plt.subplots(figsize=(8, 6))
class_names = list(classes.keys())
sns.heatmap(
    model.confusion_matrix.compute(),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
)
ax.tick_params(axis="x", labelrotation=45)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close(fig)
"""

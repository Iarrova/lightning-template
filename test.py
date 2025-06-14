import os
from argparse import ArgumentParser

import lightning as L
import torch

from src.config import Config
from src.datasets import create_dataset
from src.model.model import Model


def test(config: Config):
    L.seed_everything(config.seed)

    if torch.cuda.is_available():
        print("[INFO] CUDA is available! Testing on GPU...")
    else:
        print("[INFO] CUDA is not available. Testing on CPU...")

    dataset = create_dataset(config.dataset)
    test_loader = dataset.generate_test_loader()

    if not os.path.exists(f"./weights/{config.project_name}.ckpt"):
        raise FileNotFoundError(f"Checkpoint not found at ./weights/{config.project_name}.ckpt")

    model = Model.load_from_checkpoint(
        checkpoint_path=f"./weights/{config.project_name}.ckpt",
        config=config,
        num_classes=dataset.num_classes,
    )

    trainer = L.Trainer(accelerator="auto", devices="auto", logger=False)

    print(f"[INFO] Testing model from ./weights/{config.project_name}.ckpt")
    trainer.test(model=model, dataloaders=test_loader)


def main():
    parser = ArgumentParser(description="Testing script for image classification.")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()

    config = Config.from_json(args.config_path)

    test(config)


if __name__ == "__main__":
    main()

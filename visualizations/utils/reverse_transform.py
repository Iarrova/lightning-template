import numpy as np
import torch


def reverse_transform(input: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    input_np = input.numpy().transpose((1, 2, 0))

    input_np = std * input_np + mean
    input_np = np.clip(input_np, 0, 1)

    return input_np

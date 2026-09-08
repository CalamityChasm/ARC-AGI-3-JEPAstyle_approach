"""Shared CUDA-if-available device selection for training/eval scripts."""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

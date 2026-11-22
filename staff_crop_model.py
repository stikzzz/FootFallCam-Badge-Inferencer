"""Architecture and crop preprocessing shared by training and inference."""

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


def to_tensor(rgb, size):
    scaled = ImageOps.contain(rgb, (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(scaled, ((size - scaled.width) // 2, (size - scaled.height) // 2))
    return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div_(255)


def build_model(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(48 * 4, num_classes),
    )

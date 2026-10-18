"""Minimal stand-in for the two timm helpers modeling_florence2.py imports,
so the Furrence2 worker doesn't need timm - which pulls in torchvision,
and torchvision from PyPI can replace the cu130 torch."""
import torch
from torch import nn
from torch.nn.init import trunc_normal_  # noqa: F401


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        if keep > 0.0 and self.scale_by_keep:
            mask.div_(keep)
        return x * mask

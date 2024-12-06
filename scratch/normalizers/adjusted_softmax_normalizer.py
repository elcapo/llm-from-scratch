import torch
from .base_normalizer import BaseNormalizer

class AdjustedSoftmaxNormalizer(BaseNormalizer):
    def __init__(self, keys_dim: int):
        super().__init__()
        self.keys_dim = keys_dim

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        return torch.softmax(inputs / self.keys_dim**0.5, dim=-1)

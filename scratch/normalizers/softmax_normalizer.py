import torch
from .base_normalizer import BaseNormalizer

class SoftmaxNormalizer(BaseNormalizer):
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        return torch.softmax(inputs, dim=-1)

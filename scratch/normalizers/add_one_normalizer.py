import torch
from .base_normalizer import BaseNormalizer

class AddOneNormalizer(BaseNormalizer):
    def forward(self, inputs: torch.tensor) -> torch.tensor:
        return inputs / inputs.sum()

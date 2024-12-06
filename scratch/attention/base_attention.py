from typing import Optional
import torch
from scratch.normalizers.base_normalizer import BaseNormalizer
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

class BaseAttention(torch.nn.Module):
    def __init__(self, normalizer: Optional[BaseNormalizer] = None):
        super().__init__()
        self.normalizer = SoftmaxNormalizer()

    def forward(self, x: torch.tensor) -> torch.tensor:
        pass

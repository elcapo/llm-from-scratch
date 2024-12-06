import torch
from scratch.normalizers.base_normalizer import BaseNormalizer
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

class BaseAttention(torch.nn.Module):
    def __init__(self, normalizer: BaseNormalizer = SoftmaxNormalizer()):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x: torch.tensor) -> torch.tensor:
        pass

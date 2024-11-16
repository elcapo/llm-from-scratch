from torch import tensor, softmax
from .base_normalizer import BaseNormalizer

class SoftmaxNormalizer(BaseNormalizer):
    def normalize(self, input: tensor) -> tensor:
        return softmax(input, dim=-1)

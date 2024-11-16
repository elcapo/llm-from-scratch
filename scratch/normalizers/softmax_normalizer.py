from torch import tensor, softmax
from .base_normalizer import BaseNormalizer

class SoftmaxNormalizer(BaseNormalizer):
    def normalize(self, inputs: tensor) -> tensor:
        return softmax(inputs, dim=-1)

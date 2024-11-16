from torch import tensor
from .base_normalizer import BaseNormalizer

class AddOneNormalizer(BaseNormalizer):
    def normalize(self, input: tensor) -> tensor:
        return input / input.sum()

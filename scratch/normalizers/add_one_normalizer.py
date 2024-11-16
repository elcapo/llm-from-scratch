from torch import tensor
from .base_normalizer import BaseNormalizer

class AddOneNormalizer(BaseNormalizer):
    def normalize(self, inputs: tensor) -> tensor:
        return inputs / inputs.sum()

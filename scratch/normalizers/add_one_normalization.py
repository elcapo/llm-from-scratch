from torch import tensor
from .base_normalization import BaseNormalization

class AddOneNormalization(BaseNormalization):
    def normalize(self, input: tensor) -> tensor:
        return input / input.sum()

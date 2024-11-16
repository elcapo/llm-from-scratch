from torch import tensor, softmax
from .base_normalization import BaseNormalization

class SoftmaxNormalization(BaseNormalization):
    def normalize(self, input: tensor) -> tensor:
        return softmax(input, dim=0)

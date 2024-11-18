from torch.nn import Module
from scratch.normalizers.base_normalizer import BaseNormalizer
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

class BaseAttention(Module):
    def __init__(self, normalizer: BaseNormalizer = SoftmaxNormalizer()):
        super().__init__()
        self.normalizer = normalizer

    def forward(self, x):
        pass

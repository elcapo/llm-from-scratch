from torch import tensor
from scratch.normalizers.base_normalizer import BaseNormalizer
from scratch.normalizers.softmax_normalizer import SoftmaxNormalizer

class BaseAttention:
    def __init__(self, normalizer: BaseNormalizer = SoftmaxNormalizer()):
        self.normalizer = normalizer

    def get_scores(self, inputs: tensor) -> tensor:
        pass
    
    def get_weights(self, scores: tensor) -> tensor:
        pass
    
    def get_context_vectors(self, weights: tensor) -> tensor:
        pass

    def compute(self, inputs: tensor) -> tensor:
        pass
